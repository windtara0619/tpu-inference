# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe, ref_moe

jax.config.parse_flags_with_absl()


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_bias=False,
):
    key = jax.random.key(seed)
    k0, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

    a = (jax.random.normal(k0, (num_tokens, hidden_size),
                           dtype=jnp.bfloat16).astype(dtype) / 10)

    w1 = (jax.random.normal(
        k1,
        (num_experts, 2, hidden_size, intermediate_size),
        dtype=jnp.bfloat16,
    ) / 10).astype(dtype)
    w2 = (jax.random.normal(k2, (num_experts, intermediate_size, hidden_size),
                            dtype=jnp.bfloat16) / 10).astype(dtype)

    if has_bias:
        b1 = (jax.random.normal(k3, (num_experts, 2, 1, intermediate_size),
                                dtype=jnp.bfloat16) / 10).astype(dtype)
        b2 = (jax.random.normal(k4, (num_experts, 1, hidden_size),
                                dtype=jnp.bfloat16) / 10).astype(dtype)
    else:
        b1 = b2 = None

    gating_output = (
        jax.random.normal(k5, (num_tokens, num_experts), dtype=jnp.bfloat16) +
        jnp.arange(num_tokens * num_experts, dtype=jnp.bfloat16).reshape(
            num_tokens, num_experts) / 100)

    # To generate unique top-k!
    top_k_indices = jax.random.randint(k6, (num_tokens, top_k),
                                       minval=0,
                                       maxval=num_experts - 1,
                                       dtype=jnp.int32)

    one_hot = (jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.bfloat16),
        axis=1,
    ) * 30)

    gating_output = (gating_output + one_hot).astype(dtype)

    return a, w1, w2, b1, b2, gating_output


def sub_channel_quantize(x, quant_dtype, wsz=256):
    """Quantizes x with sub-channel quantization on the 2nd minor."""
    if jnp.issubdtype(quant_dtype, jnp.floating):
        dtype_info = jnp.finfo(quant_dtype)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)
    w_lst, scale_lst = [], []
    assert len(x.shape) >= 2
    assert x.shape[-2] % wsz == 0
    for i in range(0, x.shape[-2], wsz):
        y = x[..., i:i + wsz, :]
        abs_max = jnp.abs(y).max(axis=-2, keepdims=True)
        scale = (abs_max / dtype_max).astype(jnp.float32)
        w = (y / scale).astype(quant_dtype)
        w_lst.append(w)
        scale = jnp.expand_dims(scale, axis=-2)
        scale_lst.append(scale)
    return jnp.concat(w_lst, axis=-2), jnp.concat(scale_lst, axis=-3)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoEKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        self.mesh_devices = sorted(
            jax.devices(),
            key=lambda x: (
                x.coords[0],
                (-1 if x.coords[0] % 2 else 1) * x.coords[1],
            ),
        )
        self.mesh = Mesh(np.array(self.mesh_devices).reshape(1, -1),
                         axis_names=("data", "model"))

    def _test_moe(
        self,
        dtype,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens,
        seed,
        renormalize_topk_logits,
        bt,
        bf,
        bd1,
        bd2,
        btc,
        bfc,
        bd1c,
        bd2c,
        act_fn="silu",
        scoring_fn="softmax",
        w_dtype=None,
        subc_quant_w1_sz=None,
        subc_quant_w2_sz=None,
        has_bias=False,
        atol=2e-1,
        rtol=2e-1,
    ):
        a, w1, w2, b1, b2, gating_output = gen_moe_inputs(
            dtype,
            top_k,
            num_experts,
            hidden_size,
            intermediate_size,
            num_tokens,
            seed=seed,
            has_bias=has_bias,
        )
        w1_scale = None
        w2_scale = None
        if w_dtype is not None:
            if subc_quant_w1_sz is None:
                subc_quant_w1_sz = 256
            if subc_quant_w2_sz is None:
                subc_quant_w2_sz = 256
            w1, w1_scale = sub_channel_quantize(w1, w_dtype, subc_quant_w1_sz)
            w2, w2_scale = sub_channel_quantize(w2, w_dtype, subc_quant_w2_sz)

        actual = fused_ep_moe(
            mesh=self.mesh,
            tokens=a,
            w1=w1,
            w2=w2,
            gating_output=gating_output,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            scoring_fn=scoring_fn,
            subc_quant_w1_sz=subc_quant_w1_sz,
            subc_quant_w2_sz=subc_quant_w2_sz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            b1=b1,
            b2=b2,
            bt=bt,
            bf=bf,
            bd1=bd1,
            bd2=bd2,
            btc=btc,
            bfc=bfc,
            bd1c=bd1c,
            bd2c=bd2c,
        )
        expected = ref_moe(
            a,
            w1,
            w2,
            gating_output,
            top_k,
            b1=b1,
            b2=b2,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            scoring_fn=scoring_fn,
            subc_quant_w1_sz=subc_quant_w1_sz,
            subc_quant_w2_sz=subc_quant_w2_sz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
        self.assertAllClose(actual, expected, atol=atol, rtol=rtol)

    @parameterized.product(renormalize_topk_logits=[True, False], )
    def test_basic(self, renormalize_topk_logits):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )

    @parameterized.product(act_fn=["silu", "gelu", "swigluoai"], )
    def test_activation(self, act_fn):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            act_fn=act_fn,
            bt=32,
            bf=512,
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )

    @parameterized.product(scoring_fn=["softmax", "sigmoid"])
    def test_scoring_fn(self, scoring_fn):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            scoring_fn=scoring_fn,
            bt=32,
            bf=512,
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            atol=
            4e-1,  # loosen tolerance as jax.lax.top_k and get_top_k aren't identical on ties (related: https://github.com/jax-ml/jax/issues/34620)
        )

    def test_benchmark_qwen_235(self):
        num_experts = 128
        top_k = 8
        hidden_size = 4096
        intermediate_size = 1536
        dtype = jnp.bfloat16
        num_tokens = 8 * 64
        seed = 54321
        renormalize_topk_logits = True
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=64,
            bf=768,
            bd1=2048,
            bd2=2048,
            btc=64,
            bfc=768,
            bd1c=2048,
            bd2c=2048,
            act_fn="silu",
            atol=5e-2,
            rtol=5e-2,
        )

    def test_benchmark_qwen_30b_a3b(self):
        num_experts = 128
        top_k = 8
        hidden_size = 2048
        intermediate_size = 768
        dtype = jnp.bfloat16
        num_tokens = 512
        seed = 54321
        renormalize_topk_logits = True
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=16,
            bf=384,
            bd1=512,
            bd2=512,
            btc=16,
            bfc=384,
            bd1c=256,
            bd2c=256,
            act_fn="silu",
            atol=5e-2,
            rtol=5e-2,
        )

    @parameterized.product(
        w_dtype=[jnp.int8, jnp.float8_e5m2, jnp.float4_e2m1fn], )
    def test_sub_channel_quantization(self, w_dtype):
        if w_dtype in (
                jnp.float8_e5m2,
                jnp.float4_e2m1fn,
        ) and not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            w_dtype=w_dtype,
            subc_quant_w1_sz=256,
            subc_quant_w2_sz=256,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )

    @parameterized.product(w_dtype=[jnp.int8, jnp.float8_e5m2], )
    def test_per_channel_quantization(self, w_dtype):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 512
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            w_dtype=w_dtype,
            subc_quant_w1_sz=hidden_size,
            subc_quant_w2_sz=intermediate_size,
            bt=32,
            bf=1024,
            bd1=hidden_size,
            bd2=hidden_size,
            btc=32,
            bfc=512,
            bd1c=256,
            bd2c=256,
        )

    def test_bias(self):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            has_bias=True,
            bt=32,
            bf=512,
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )

    @parameterized.named_parameters(
        # attn_dp=4, model=2: matches a DP-attention config where the KV-head
        # count forces most of the mesh onto the attention-DP axis.
        ("attn_dp4_model2", 4, 2),
        # attn_dp=2, model=4: the reverse split, to make sure the ep_rank ->
        # per-axis-coordinate decomposition isn't accidentally order-specific.
        ("attn_dp2_model4", 2, 4),
    )
    def test_multi_axis_ep_group(self, attn_dp_size, model_size):
        """`ep_axis_name` may span more than one mesh axis under DP-attention
        (e.g. ("attn_dp", "model")). Tokens/gating arrive sharded only along
        `attn_dp` (as they would out of a DP-attention block, replicated over
        `model`), while weights are already sharded over the *combined*
        (attn_dp, model) group (as GMM_EP already stores them). The kernel
        must reshard tokens/gating to match and route correctly across the
        whole combined group, not just the single `attn_dp` or `model` axis.
        """
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32

        mesh = Mesh(
            np.array(self.mesh_devices).reshape(1, attn_dp_size, model_size),
            axis_names=("data", "attn_dp", "model"),
        )
        ep_axis_name = ("attn_dp", "model")

        a, w1, w2, b1, b2, gating_output = gen_moe_inputs(
            dtype,
            top_k,
            num_experts,
            hidden_size,
            intermediate_size,
            num_tokens,
            seed=1234,
        )

        # Tokens/gating: sharded only along attn_dp, replicated over model --
        # i.e. NOT already laid out for the combined ep group.
        token_sharding = NamedSharding(mesh, P("attn_dp", None))
        a_dp = jax.device_put(a, token_sharding)
        gating_dp = jax.device_put(gating_output, token_sharding)
        # Weights: sharded over the full combined ep group, as GMM_EP already
        # stores them (see `_get_moe_weight_shardings`).
        ep_sharding = NamedSharding(mesh, P(ep_axis_name))
        w1_ep = jax.device_put(w1, ep_sharding)
        w2_ep = jax.device_put(w2, ep_sharding)

        actual = fused_ep_moe(
            mesh=mesh,
            tokens=a_dp,
            w1=w1_ep,
            w2=w2_ep,
            gating_output=gating_dp,
            top_k=top_k,
            renormalize_topk_logits=False,
            act_fn="silu",
            scoring_fn="softmax",
            ep_axis_name=ep_axis_name,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )
        expected = ref_moe(
            a,
            w1,
            w2,
            gating_output,
            top_k,
            renormalize_topk_logits=False,
            act_fn="silu",
            scoring_fn="softmax",
        )
        self.assertAllClose(actual, expected, atol=2e-1, rtol=2e-1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
