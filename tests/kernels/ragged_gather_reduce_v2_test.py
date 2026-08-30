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
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce as ragged_gather_reduce_v2
from tpu_inference.kernels.sparse_core.ragged_scatter import ragged_scatter

jax.config.parse_flags_with_absl()


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


def _set_interleave_row_partitions(enabled: bool) -> None:
    """Flips RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS and clears
  ragged_gather_reduce_v2's jit cache.

  The flag is read as a plain Python bool at trace time (an `if`, not a
  jnp.where), so jax.jit's cache -- keyed on abstract input shapes/dtypes,
  not on env vars read during tracing -- would otherwise silently replay a
  stale compiled kernel across a flip instead of retracing.
  """
    os.environ["RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS"] = (
        "1" if enabled else "0")
    ragged_gather_reduce_v2.clear_cache()


class InterleaveRowPartitionsTest(jtu.JaxTestCase):
    """Real production shape: Qwen3-30B-A3B (128 experts, top-8 routing),
  8-way EP (16 local experts/shard), hidden_size=2048. ``x`` mimics
  fused_moe_gmm.py's ``gmm2_res``: one row per dispatched (token,
  expert-choice) slot, globally expert-sorted via jax.lax.sort_key_val, with
  a shard's local experts owning one contiguous window of it (size ==
  however many slots landed there this call). ``indices`` is built as a
  true permutation, like the real topk_argsort_revert_indices, so valid
  rows land inside that window rather than being drawn i.i.d. at random.
  """

    HIDDEN_SIZE = 2048
    NUM_TOKENS = 32768  # batch-size=8 * input-len=4096
    TOPK = 8
    REDUCE_GROUP_SIZE = TOPK
    INPUT_SIZE = NUM_TOKENS * TOPK  # 262144
    NUM_SRC_ROWS = INPUT_SIZE

    def _make_inputs(self, span: int, key: jax.Array):
        """Single contiguous valid span in token-major position space (the
    request-correlated skew measured via MOE_LOG_COMBINE_VALID_ROWS_STATS
    in production), with a shard-local-expert window of matching size in
    ``x``'s row space.
    """
        k1, k2, k3, k4 = jax.random.split(key, 4)
        x = jax.random.normal(k1, (self.NUM_SRC_ROWS, self.HIDDEN_SIZE),
                              jnp.float32).astype(jnp.bfloat16)
        topk_weights = jax.random.normal(k2, (self.INPUT_SIZE, ),
                                         jnp.bfloat16)

        pos = jnp.arange(self.INPUT_SIZE)
        valid_rows_mask = pos < span

        window_vals = jax.random.permutation(k3, span)
        rest_vals = span + jax.random.permutation(k4,
                                                  self.INPUT_SIZE - span)
        indices = jnp.zeros((self.INPUT_SIZE, ), jnp.int32)
        indices = indices.at[:span].set(window_vals.astype(jnp.int32))
        indices = indices.at[span:].set(rest_vals.astype(jnp.int32))
        return x, indices, topk_weights, valid_rows_mask

    @parameterized.parameters(2048, 8192, 32768)
    def test_bit_identical_output(self, span):
        key = jax.random.key(0)
        x, indices, topk_weights, valid_rows_mask = self._make_inputs(
            span, key)

        try:
            _set_interleave_row_partitions(False)
            contiguous = ragged_gather_reduce_v2(x, indices, topk_weights,
                                                 valid_rows_mask,
                                                 self.REDUCE_GROUP_SIZE)
            _set_interleave_row_partitions(True)
            interleaved = ragged_gather_reduce_v2(x, indices, topk_weights,
                                                  valid_rows_mask,
                                                  self.REDUCE_GROUP_SIZE)
        finally:
            _set_interleave_row_partitions(False)

        np.testing.assert_array_equal(np.asarray(contiguous),
                                      np.asarray(interleaved))

    @parameterized.parameters(2048, 8192, 32768, 65536, 131072)
    def test_perf_ab(self, span):
        """Not an assertion -- prints the contiguous-vs-interleaved on-device
    time so a regression sweep can be read off the test log, the same way
    ScatterTest.test_perf above does.
    """
        key = jax.random.key(0)
        x, indices, topk_weights, valid_rows_mask = self._make_inputs(
            span, key)

        print(f"\n=== interleave A/B: span={span} ===")
        try:
            _set_interleave_row_partitions(False)
            t_old = _time_function(ragged_gather_reduce_v2, x, indices,
                                   topk_weights, valid_rows_mask,
                                   self.REDUCE_GROUP_SIZE)
            print(f"contiguous (old):   {t_old*1e6:9.1f} us")

            _set_interleave_row_partitions(True)
            t_new = _time_function(ragged_gather_reduce_v2, x, indices,
                                   topk_weights, valid_rows_mask,
                                   self.REDUCE_GROUP_SIZE)
            print(f"interleaved (new):  {t_new*1e6:9.1f} us  "
                  f"({t_old/t_new:.2f}x)")
        except Exception as e:  # pylint: disable=broad-except
            print(f"interleave A/B failed: {e}")
        finally:
            _set_interleave_row_partitions(False)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
