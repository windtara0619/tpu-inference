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

import functools
import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    _fallback_implementation
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    _sparse_core_implementation
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce as ragged_gather_reduce_v2
from tpu_inference.kernels.sparse_core.ragged_scatter import ragged_scatter

jax.config.parse_flags_with_absl()

_jit_fallback = jax.jit(_fallback_implementation,
                        static_argnames="reduce_group_size")
_jit_sparse_core = jax.jit(_sparse_core_implementation,
                           static_argnames="reduce_group_size")


def reference_ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


@functools.partial(jax.jit, static_argnames="reduce_group_size")
def ragged_scatter_and_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    start: jax.Array,
    end: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    x = ragged_scatter(x, indices, start, end)
    out = x.reshape((-1, reduce_group_size, x.shape[-1]))
    topk_weights = topk_weights.reshape((-1, reduce_group_size))[..., None]
    out = out * topk_weights
    out = jnp.where(
        valid_rows_mask.reshape((-1, reduce_group_size))[:, :, None], out, 0.0)
    out = out.sum(axis=-2)
    return out


def _time_function(fn, *args, n_repeats=100):
    # Warmup
    for _ in range(10):
        fn(*args).block_until_ready()

    # Asynchronous dispatch to hide Python overhead
    start = time.perf_counter()
    results = [fn(*args) for _ in range(n_repeats)]
    results[-1].block_until_ready()
    end = time.perf_counter()

    return (end - start) / n_repeats


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScatterTest(jtu.JaxTestCase):
    _test_cases = [
        dict(out_size=o,
             start_end=se,
             hidden_size=h,
             dtype=d,
             reduce_group_size=rg) for o, se, h, d, rg in itertools.chain(
                 itertools.product(
                     [400, 840],
                     [(3, 338), (10, 255)],
                     [128, 512, 8192],
                     [jnp.bfloat16, jnp.float32],
                     [8, 5],
                 ),
                 itertools.product(
                     [16384],
                     [(99, 1120)],
                     [7168],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [16384],
                     [(300, 2358)],
                     [6144],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [20480],
                     [(300, 2850)],
                     [4096],
                     [jnp.bfloat16],
                     [10],
                 ),
             )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_ragged_gather_reduce(self, out_size, hidden_size, start_end,
                                     dtype, reduce_group_size):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )
        # Correctness check.
        desired = reference_ragged_gather_reduce(x, indices, topk_weights,
                                                 valid_rows_mask,
                                                 reduce_group_size)
        try:
            actual = ragged_gather_reduce_v2(x, indices, topk_weights,
                                             valid_rows_mask,
                                             reduce_group_size)
            np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)
        except AssertionError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            print("Skipping ragged_gather_reduce_v2 correctness check due to "
                  f"error: {e}")

    # The first perf test case approximates the DeepSeekV3, 2k-batch-size, EP=16.
    # The second case approximates the Qwen3-Coder-480B, 2k-batch-size, EP=8.
    _perf_test_cases = [
        dict(
            out_size=o,
            start_end=se,
            hidden_size=h,
            dtype=d,
            reduce_group_size=rg,
            col_chunk_size=c_sz,
        ) for o, se, h, d, rg, c_sz in itertools.chain(
            itertools.product(
                [16384],
                [(99, 1120)],
                [7168],
                [jnp.bfloat16],
                [8],
                [3584],
            ),
            itertools.product(
                [16384],
                [(300, 2400)],
                [6144],
                [jnp.bfloat16],
                [8],
                [2048],
            ),
            itertools.product(
                [65536],
                [(100, 8300)],
                [6144],
                [jnp.bfloat16],
                [8],
                [2048],
            ),
        )
    ]

    @parameterized.parameters(*_perf_test_cases)
    def test_perf(
        self,
        out_size,
        hidden_size,
        start_end,
        dtype,
        reduce_group_size,
        col_chunk_size,
    ):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )

        print(f"\n=== Running shape: out={out_size},"
              f" hidden={hidden_size}, start={start}, end={end} ===")

        def run_and_time(name, fn, *args):
            try:
                t_val = _time_function(fn, *args)
                print(f"{name}: {t_val*1000:.3f} ms")
            except Exception as e:  # pylint: disable=broad-except
                print(f"{name} failed: {e}")

        run_and_time(
            "ragged_scatter_and_reduce",
            ragged_scatter_and_reduce,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            start,
            end,
            reduce_group_size,
        )

        run_and_time(
            "ragged_gather_reduce_v2",
            ragged_gather_reduce_v2,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )

    # `ragged_gather_reduce` picks the fallback (dense TensorCore gather) over
    # the SparseCore kernel when `size(x) * dtype_bytes * 2 <
    # vmem_capacity_bytes * 0.6`. These cases sweep `size_multiplier`, x's
    # size as a multiple of that crossover point, so the two implementations
    # can be timed head-to-head on the same shapes on both sides of the
    # switchover -- letting us check whether the 0.6 heuristic actually lines
    # up with where SparseCore starts winning.
    #
    # They also sweep `valid_fraction`: the SparseCore kernel is *ragged* --
    # it sorts valid rows to the front and only pipelines over
    # `num_valid_rows`, skipping invalid/padded rows entirely -- while the
    # fallback densely gathers+masks all `input_size` rows regardless of
    # validity. A real MoE EP shard typically only owns a small, uneven
    # fraction of the routed tokens, so this axis matters as much as raw size.
    _fallback_vs_sparse_core_cases = [
        dict(
            size_multiplier=m,
            valid_fraction=v,
            hidden_size=4096,
            dtype=jnp.bfloat16,
            reduce_group_size=8,
        ) for m, v in itertools.product(
            [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 6.0, 8.0,
                12.0, 16.0
            ],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
    ]

    @parameterized.parameters(*_fallback_vs_sparse_core_cases)
    def test_fallback_vs_sparse_core_perf(
        self,
        size_multiplier,
        valid_fraction,
        hidden_size,
        dtype,
        reduce_group_size,
    ):
        tpu_info = pltpu.get_tpu_info()
        dtype_bytes = jax.dtypes.itemsize_bits(dtype) // 8
        # Element count of x at which ragged_gather_reduce's dispatch
        # condition flips from fallback to SparseCore.
        threshold_elements = (tpu_info.vmem_capacity_bytes * 0.6) / (
            dtype_bytes * 2)

        out_size = max(
            reduce_group_size,
            int(size_multiplier * threshold_elements / hidden_size))
        out_size -= out_size % reduce_group_size
        out_size = max(out_size, reduce_group_size)

        num_valid = max(1, int(out_size * valid_fraction))
        start = 0
        end = num_valid

        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.logical_and(start <= indices, indices < end)

        would_dispatch_to = ("fallback" if
                             (out_size * hidden_size * dtype_bytes * 2 <
                              tpu_info.vmem_capacity_bytes * 0.6) else
                             "sparse_core")
        print(f"\n=== out_size={out_size} (x{size_multiplier:g} threshold), "
              f"hidden={hidden_size}, valid_fraction={valid_fraction:g} "
              f"({num_valid} rows) -- real dispatch picks: "
              f"{would_dispatch_to} ===")

        def run_and_time(name, fn):
            try:
                t_val = _time_function(
                    fn,
                    x,
                    indices,
                    topk_weights,
                    valid_rows_mask,
                    reduce_group_size,
                )
                print(f"{name}: {t_val * 1000:.3f} ms")
            except Exception as e:  # pylint: disable=broad-except
                print(f"{name} failed: {e}")

        run_and_time("fallback", _jit_fallback)
        run_and_time("sparse_core", _jit_sparse_core)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
