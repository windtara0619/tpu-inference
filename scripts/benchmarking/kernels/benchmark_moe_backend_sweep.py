# Copyright 2026 Google LLC
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
"""Sweep num_tokens comparing the GMM_EP and FUSED_MOE MoE backends, to pick
a MOE_FUSED_KERNEL_MAX_NUM_TOKENS threshold (see tpu_inference/envs.py and
tpu_inference/layers/common/moe.py::moe_apply).

Both backends are invoked through the real `moe_apply` entry point --
GMM_EP-formatted weights are stored once (as the real serving path would),
and the FUSED_MOE leg goes through the same on-the-fly
`convert_gmm_ep_weights_to_fused_moe` relayout that the dynamic switch uses in
production, so the measured time already includes that conversion's cost.

Defaults match Qwen3-30B-A3B (128 experts, hidden=2048, intermediate=768,
top_k=8) under the DP8_EP sharding strategy (attn_dp=4, model=2 for this
model's 4 KV heads + fp8 KV cache -- see ShardingConfigManager). Override
--attn-dp-size/--model-size for a different model/sharding; their product
must equal your device count.

Run on a TPU host:
    python scripts/benchmarking/kernels/benchmark_moe_backend_sweep.py \
        --num-tokens 16,32,64,128,256,512,1024,2048
"""

import argparse
import os

# ShardingAxisName's scheme is cached on first access, so this must be set
# before any tpu_inference.layers.common.sharding import below.
os.environ.setdefault("NEW_MODEL_DESIGN", "1")

import glob
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


def device_time_us(fn, *, reps: int, warmup: int) -> float:
    """Average on-device execution time per call, in microseconds, from an
    XLA profiler trace -- excludes Python dispatch/kernel-launch overhead.

    Some legs (e.g. the dynamic gmm_ep->FUSED_MOE switch) mix several eager
    weight-relayout ops with the jitted kernel call, so a single logical
    `fn()` call can emit more than one "XLA Modules" trace event; there's no
    clean call boundary to attribute events to individually. Instead of
    matching module count to call count (fragile, and what a naive port of
    collective_bench_lib.device_time's barrier-subtraction would need), this
    sums *all* device_duration_ps across the whole trace window -- which
    spans exactly `reps` back-to-back calls -- and divides by `reps`. That
    is correct regardless of how many modules a single call produces.
    """
    for _ in range(warmup):
        jax.block_until_ready(fn())
    trace_dir = tempfile.mkdtemp()
    try:
        with jax.profiler.trace(trace_dir):
            out = None
            for _ in range(reps):
                out = fn()
            jax.block_until_ready(out)
        pbs = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"),
                        recursive=True)
        if not pbs:
            raise RuntimeError("profiler wrote no trace")
        newest = max(pbs, key=os.path.getmtime)
        plane = jax.profiler.ProfileData.from_file(
            newest).find_plane_with_name("/device:TPU:0")
        if plane is None:
            raise RuntimeError("no /device:TPU:0 plane in the trace")
        total_ps = 0.0
        found = False
        for line in plane.lines:
            if line.name != "XLA Modules":
                continue
            for event in line.events:
                for stat_name, value in event.stats:
                    if stat_name == "device_duration_ps":
                        total_ps += value
                        found = True
        if not found:
            raise RuntimeError("no XLA-module device time in the profile")
        return total_ps / reps / 1e6  # ps -> us
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)

MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model",
                   "dcp", "pcp")


def build_mesh(attn_dp_size: int, model_size: int) -> Mesh:
    """The real NEW_MODEL_DESIGN mesh shape, with only attn_dp/model active --
    matches DP8_EP as built by tpu_runner.py's `_create_new_model_mesh`."""
    num_devices = jax.device_count()
    if attn_dp_size * model_size != num_devices:
        raise ValueError(
            f"attn_dp_size ({attn_dp_size}) * model_size ({model_size}) must "
            f"equal the device count ({num_devices})")
    # Shape order matches MESH_AXIS_NAMES: (data, attn_dp, attn_dp_expert,
    # expert, model, dcp, pcp); only attn_dp and model are non-1.
    devices = np.asarray(jax.devices()).reshape(1, attn_dp_size, 1, 1,
                                                 model_size, 1, 1)
    return Mesh(devices, MESH_AXIS_NAMES)


def make_layer(top_k: int, use_ep: bool) -> MagicMock:
    layer = MagicMock()
    layer._get_name.return_value = "moe"
    layer.activation = "silu"
    layer.swiglu_limit = None
    layer.top_k = top_k
    layer.renormalize = True
    layer.scoring_func = "softmax"
    layer.use_ep = use_ep
    return layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens",
                        default="16,32,64,128,256,512,1024,2048")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--attn-dp-size",
        type=int,
        default=4,
        help="mesh 'attn_dp' axis size (default matches Qwen3-30B-A3B DP8_EP)"
    )
    parser.add_argument(
        "--model-size",
        type=int,
        default=2,
        help="mesh 'model' axis size (default matches Qwen3-30B-A3B DP8_EP)")
    parser.add_argument("--reps",
                        type=int,
                        default=20,
                        help="traced calls per cell (median)")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    mesh = build_mesh(args.attn_dp_size, args.model_size)
    print(f"devices={jax.device_count()} ({jax.devices()[0].device_kind}), "
         f"mesh={dict(mesh.shape)}, EXPERT axis -> {ShardingAxisName.EXPERT}")

    num_experts, hidden_size, intermediate_size = (args.num_experts,
                                                    args.hidden_size,
                                                    args.intermediate_size)
    key_w1, key_w2 = jax.random.split(jax.random.key(0))
    w13_raw = (jax.random.normal(
        key_w1, (num_experts, 2 * intermediate_size, hidden_size),
        dtype=jnp.bfloat16) / 10)
    w2_raw = (jax.random.normal(
        key_w2, (num_experts, hidden_size, intermediate_size),
        dtype=jnp.bfloat16) / 10)
    raw = FusedMoEWeights(w13_weight=w13_raw,
                          w13_weight_scale=None,
                          w13_bias=None,
                          w2_weight=w2_raw,
                          w2_weight_scale=None,
                          w2_bias=None)
    # GMM_EP-formatted weights are what's actually stored on the layer in
    # production; FUSED_MOE is derived from these on the fly per call.
    gmm_ep_processed = process_moe_weights(raw,
                                           moe_backend=MoEBackend.GMM_EP,
                                           w13_reorder_size=1)
    gmm_ep_weights = shard_moe_weights(gmm_ep_processed, MoEBackend.GMM_EP,
                                       mesh)
    # For comparison only: FUSED_MOE weights as they'd be stored if this
    # layer had *statically* selected FUSED_MOE (paid once at load time, not
    # per call) -- isolates the kernel's own cost from the dynamic switch's
    # per-call relayout cost.
    fused_native_processed = process_moe_weights(raw,
                                                  moe_backend=MoEBackend.FUSED_MOE)
    fused_native_weights = shard_moe_weights(fused_native_processed,
                                             MoEBackend.FUSED_MOE, mesh)
    # GMM_TP: a genuinely different physical layout/sharding (tensor-parallel,
    # not expert-sharded) -- included for a full picture, not something the
    # dynamic switch (GMM_EP <-> FUSED_MOE) touches.
    w13_reorder_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
    gmm_tp_processed = process_moe_weights(raw,
                                           moe_backend=MoEBackend.GMM_TP,
                                           w13_reorder_size=w13_reorder_size)
    gmm_tp_weights = shard_moe_weights(gmm_tp_processed, MoEBackend.GMM_TP,
                                       mesh)

    layer_ep = make_layer(args.top_k, use_ep=True)
    layer_tp = make_layer(args.top_k, use_ep=False)

    names = [
        "gmm_ep", "gmm_tp", "fused_moe (dynamic)", "fused_moe (native)"
    ]
    print(" | ".join(f"{c:>20}" for c in ["num_tokens"] +
                     [f"{n} us" for n in names] +
                     ["gmm_tp x", "dynamic x", "native x"]))

    for num_tokens in [int(v) for v in args.num_tokens.split(",")]:
        key_x, key_g = jax.random.split(jax.random.key(1000 + num_tokens))
        token_sharding = NamedSharding(mesh, P("attn_dp", None))
        x = jax.device_put(
            (jax.random.normal(
                key_x, (num_tokens, hidden_size), dtype=jnp.bfloat16) / 10),
            token_sharding)
        gating = jax.device_put(
            jax.random.normal(key_g, (num_tokens, num_experts),
                              dtype=jnp.bfloat16), token_sharding)

        def call_gmm_ep():
            with patch("tpu_inference.envs.MOE_FUSED_KERNEL_MAX_NUM_TOKENS",
                      0):
                return moe_apply(layer=layer_ep,
                                 x=x,
                                 gating_output=gating,
                                 weights=gmm_ep_weights,
                                 moe_backend=MoEBackend.GMM_EP,
                                 mesh=mesh,
                                 extra_backend_kwargs={})

        def call_gmm_tp():
            return moe_apply(layer=layer_tp,
                             x=x,
                             gating_output=gating,
                             weights=gmm_tp_weights,
                             moe_backend=MoEBackend.GMM_TP,
                             mesh=mesh,
                             extra_backend_kwargs={})

        def call_fused_dynamic():
            # A huge threshold always triggers the dynamic switch for this
            # num_tokens -- this is the real cost the switch pays per call,
            # relayout included.
            with patch("tpu_inference.envs.MOE_FUSED_KERNEL_MAX_NUM_TOKENS",
                      10**9):
                return moe_apply(layer=layer_ep,
                                 x=x,
                                 gating_output=gating,
                                 weights=gmm_ep_weights,
                                 moe_backend=MoEBackend.GMM_EP,
                                 mesh=mesh,
                                 extra_backend_kwargs={})

        def call_fused_native():
            # Weights pre-converted once, outside the timed call -- what
            # FUSED_MOE would cost if *statically* selected for this layer.
            return moe_apply(layer=layer_ep,
                             x=x,
                             gating_output=gating,
                             weights=fused_native_weights,
                             moe_backend=MoEBackend.FUSED_MOE,
                             mesh=mesh,
                             extra_backend_kwargs={
                                 "ep_axis_name": ShardingAxisName.EXPERT
                             })

        calls = {
            "gmm_ep": call_gmm_ep,
            "gmm_tp": call_gmm_tp,
            "fused_moe (dynamic)": call_fused_dynamic,
            "fused_moe (native)": call_fused_native,
        }
        times, errs = {}, {}
        for name, call in calls.items():
            try:
                times[name] = device_time_us(call,
                                             reps=args.reps,
                                             warmup=args.warmup)
            except Exception as e:  # noqa: BLE001
                times[name], errs[name] = None, " ".join(str(e).split())[:70]

        cells = [f"{num_tokens:>20}"]
        cells += [
            f"{times[n]:>20.1f}" if times[n] else f"{'FAIL':>20}"
            for n in names
        ]
        base = times.get("gmm_ep")
        for n in ["gmm_tp", "fused_moe (dynamic)", "fused_moe (native)"]:
            speedup = base / times[n] if base and times.get(n) else None
            cells.append(f"{speedup:>20.2f}" if speedup else f"{'--':>20}")
        line = " | ".join(cells)
        if errs:
            line += "  " + "; ".join(f"{n}: {errs[n]}" for n in errs)
        print(line)

    print("\nAny '* x' column > 1.0 means that leg is faster than gmm_ep at "
         "that num_tokens. gmm_tp uses a different physical weight layout "
         "(tensor-parallel, not expert-sharded) -- it's shown for a full "
         "picture, not something MOE_FUSED_KERNEL_MAX_NUM_TOKENS touches. "
         "'dynamic' is what MOE_FUSED_KERNEL_MAX_NUM_TOKENS actually pays "
         "per call (relayout included); 'native' is the kernel-only cost if "
         "FUSED_MOE were statically selected instead, isolating the "
         "relayout's overhead. Pick MOE_FUSED_KERNEL_MAX_NUM_TOKENS as the "
         "largest num_tokens where the 'dynamic' column still beats gmm_ep.")


if __name__ == "__main__":
    main()
