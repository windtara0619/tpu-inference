import time

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from jax._src import dtypes
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention, ref_ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_old import (
    ragged_paged_attention as ragged_paged_attention_old)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing, merge_sequences_into_tiles)

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionKernelTest(jtu.JaxTestCase):

    def _test_ragged_paged_attention(
        self,
        seq_lens,  # List[(q_len, kv_len)]
        num_heads,  # [num_q_heads, num_kv_heads]
        head_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        *,
        num_kv_pages_per_block=8,
        num_queries_per_block=64,
        vmem_limit_bytes=100 * 1024 * 1024,
        max_num_batched_tokens=512,
        max_num_seq=8,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        rng = np.random.default_rng(1234)

        def gen_random(shape, dtype):
            return jnp.array(rng.random(size=shape,
                                        dtype=np.float32)).astype(dtype)

        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        cu_q_lens = [0]
        kv_lens = []
        for q_len, kv_len in seq_lens:
            assert q_len <= kv_len
            cu_q_lens.append(cu_q_lens[-1] + q_len)
            kv_lens.append(kv_len)

        max_num_batched_tokens = max(align_to(cu_q_lens[-1], 128),
                                     max_num_batched_tokens)
        max_num_seq = max(align_to(len(seq_lens), 8), max_num_seq)
        max_kv_len = max(kv_lens)
        pages_per_seq = cdiv(max_kv_len, page_size)
        num_q_heads, num_kv_heads = num_heads

        q = gen_random((max_num_batched_tokens, num_q_heads, head_dim),
                       q_dtype)
        k = gen_random((max_num_batched_tokens, num_kv_heads, head_dim),
                       kv_dtype)
        v = gen_random((max_num_batched_tokens, num_kv_heads, head_dim),
                       kv_dtype)
        page_cnt = 0
        page_indices_list = []
        kv_pages_list = []
        kv_packing = get_dtype_packing(kv_dtype)
        padded_head_dim = align_to(head_dim, 128)
        num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
        for kv_len in kv_lens:
            kv = gen_random((
                kv_len,
                num_kv_heads_x2 // kv_packing,
                kv_packing,
                padded_head_dim,
            ), kv_dtype)
            kv = jnp.pad(
                kv,
                (
                    (
                        0,
                        cdiv(kv_len, page_size) * page_size - kv_len,
                    ),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=jnp.nan,
            ).reshape(
                -1,
                page_size,
                num_kv_heads_x2 // kv_packing,
                kv_packing,
                padded_head_dim,
            )
            indices = page_cnt + jnp.arange(kv.shape[0], dtype=jnp.int32)
            indices = jnp.pad(
                indices,
                ((0, pages_per_seq - indices.shape[0]), ),
                constant_values=jnp.nan,
            )
            page_indices_list.append(indices)
            page_cnt += kv.shape[0]
            kv_pages_list.append(kv)

        kv_cache = jnp.concatenate(kv_pages_list, axis=0)
        kv_cache = jnp.pad(
            kv_cache,
            ((0, num_pages - kv_cache.shape[0]), (0, 0), (0, 0), (0, 0),
             (0, 0)),
            constant_values=jnp.nan,
        )
        page_indices = jnp.stack(page_indices_list, axis=0)
        page_indices = jnp.pad(
            page_indices,
            ((0, max_num_seq - page_indices.shape[0]), (0, 0)),
            constant_values=jnp.nan,
        )
        page_indices = page_indices.reshape(-1)

        cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, max_num_seq + 1 - cu_q_lens.shape[0]))
        kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
        kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
        distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

        args = (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

        kwargs = {
            "sliding_window": sliding_window,
            "soft_cap": soft_cap,
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }

        expected, expected_kv_cache = ref_ragged_paged_attention(
            *args,
            **kwargs,
        )

        start_time = time.perf_counter()
        output, updated_kv_cache = ragged_paged_attention(
            *args,
            **kwargs,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        output.block_until_ready()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logging.info("RPA v3 latency: %.3f ms", latency_ms)
        output = output[:cu_q_lens[distribution[-1]]]

        kv_cache_old = jnp.concatenate(kv_pages_list, axis=0)
        kv_cache_old = jnp.pad(
            kv_cache_old,
            ((0, num_pages - kv_cache_old.shape[0]), (0, 0), (0, 0), (0, 0),
             (0, 0)),
            constant_values=jnp.nan,
        )
        page_indices_old = jnp.stack(page_indices_list, axis=0)
        page_indices_old = jnp.pad(
            page_indices_old,
            ((0, max_num_seq - page_indices_old.shape[0]), (0, 0)),
            constant_values=jnp.nan,
        )
        page_indices_old = page_indices_old.reshape(-1)
        cu_q_lens_old = jnp.array(cu_q_lens, dtype=jnp.int32)
        cu_q_lens_old = jnp.pad(cu_q_lens_old,
                                (0, max_num_seq + 1 - cu_q_lens_old.shape[0]))
        kv_lens_old = jnp.array(kv_lens, dtype=jnp.int32)
        kv_lens_old = jnp.pad(kv_lens_old, (0, max_num_seq - kv_lens_old.shape[0]))
        distribution_old = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)
        args_old = (
            q,
            k,
            v,
            kv_cache_old,
            kv_lens_old,
            page_indices_old,
            cu_q_lens_old,
            distribution_old,
        )

        start_time = time.perf_counter()
        output_old, _ = ragged_paged_attention_old(
            *args_old,
            **kwargs,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        output_old.block_until_ready()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logging.info("RPA v3 (old) latency: %.3f ms", latency_ms)
        del output_old

        dtype_bits = dtypes.bit_width(jnp.dtype(kv_dtype))
        tols = {
            32: 0.15,
            16: 0.2,
            8: 0.2,
            4: 0.2,
        }
        tol = tols[dtype_bits]
        self.assertAllClose(output, expected, atol=tol, rtol=tol)
        mask = ~jnp.isnan(expected_kv_cache)
        self.assertArraysEqual(updated_kv_cache[mask], expected_kv_cache[mask])
        self.assertEqual(output.shape[-1], head_dim)

    def test_merge_sequences_into_tiles_basic(self):
        kv_lens = jnp.array([20, 1, 1, 130], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 10, 11, 12, 20], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 4], dtype=jnp.int32)

        (starts_seq, ends_seq, cu_q_lens_per_tile, cu_kv_lens_per_tile,
         tile_distribution) = merge_sequences_into_tiles(
             kv_lens,
             cu_q_lens,
             distribution,
             bq_sz=128,
             bkv_sz=128,
         )

        self.assertArraysEqual(starts_seq[:2], jnp.array([0, 3], jnp.int32))
        self.assertArraysEqual(ends_seq[:2], jnp.array([3, 4], jnp.int32))
        self.assertArraysEqual(tile_distribution,
                               jnp.array([0, 0, 2], jnp.int32))
        self.assertArraysEqual(cu_q_lens_per_tile[0, :4],
                               jnp.array([0, 10, 11, 12], jnp.int32))
        self.assertArraysEqual(cu_kv_lens_per_tile[0, :4],
                               jnp.array([0, 20, 21, 22], jnp.int32))
        self.assertArraysEqual(cu_q_lens_per_tile[1, :2],
                               jnp.array([0, 8], jnp.int32))
        self.assertArraysEqual(cu_kv_lens_per_tile[1, :2],
                               jnp.array([0, 130], jnp.int32))

    def test_merge_sequences_into_tiles_respects_distribution(self):
        kv_lens = jnp.array([10, 10, 10], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        distribution = jnp.array([1, 2, 3], dtype=jnp.int32)

        (starts_seq, ends_seq, _, _, tile_distribution) = (
            merge_sequences_into_tiles(
                kv_lens,
                cu_q_lens,
                distribution,
                bq_sz=128,
                bkv_sz=128,
            ))

        self.assertArraysEqual(starts_seq[:3],
                               jnp.array([0, 1, 2], jnp.int32))
        self.assertArraysEqual(ends_seq[:3],
                               jnp.array([1, 2, 3], jnp.int32))
        self.assertArraysEqual(tile_distribution,
                               jnp.array([1, 2, 3], jnp.int32))

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], )
    def test_ragged_paged_attention_basic(self, dtype):
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    # TODO: support integer (int8, int4) and fp4 kv cache
    @parameterized.product(
        q_dtype=[jnp.bfloat16],
        kv_dtype=[jnp.float8_e5m2, jnp.float8_e4m3fn],
        kv_scales=[(0.5, 0.5), (1.0, 1.0)],
    )
    def test_ragged_paged_attention_quantized_kv_cache(self, q_dtype, kv_dtype,
                                                       kv_scales):
        if not jtu.is_device_tpu_at_least(version=5):
            self.skipTest("Expect TPUv5+")
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000
        k_scale, v_scale = kv_scales

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            q_dtype,
            kv_dtype,
            num_pages,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    @parameterized.product(
        q_dtype=[jnp.bfloat16],
        kv_dtype=[jnp.float8_e5m2, jnp.float8_e4m3fn],
        q_scale=[0.5, 1.0],
        kv_scales=[(0.5, 0.5), (1.0, 1.0)],
    )
    def test_ragged_paged_attention_quantized_attention(
            self, q_dtype, kv_dtype, q_scale, kv_scales):
        if not jtu.is_device_tpu_at_least(version=5):
            self.skipTest("Expect TPUv5+")
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000
        k_scale, v_scale = kv_scales

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            q_dtype,
            kv_dtype,
            num_pages,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], )
    def test_ragged_paged_attention_decode_only(self, dtype):
        seq_lens = [
            (1, 18),
            (1, 129),
            (1, 597),
            (1, 122),
            (1, 64),
            (1, 322),
            (1, 463),
            (1, 181),
            (1, 1107),
            (1, 123),
            (1, 31),
            (1, 18),
            (1, 1229),
            (1, 229),
            (1, 87),
            (1, 1328),
        ]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], )
    def test_ragged_paged_attention_prefill_only(self, dtype):
        seq_lens = [
            (5, 18),
            (15, 129),
            (120, 597),
            (100, 122),
            (21, 64),
            (32, 322),
            (251, 463),
            (40, 181),
            (64, 1107),
            (99, 123),
            (10, 31),
            (5, 18),
            (3, 1229),
            (120, 229),
            (9, 87),
            (2, 1328),
        ]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], )
    def test_ragged_paged_attention_mixed(self, dtype):
        seq_lens = [
            (5, 18),
            (1, 129),
            (120, 597),
            (1, 122),
            (1, 64),
            (32, 322),
            (251, 463),
            (1, 181),
            (1, 1107),
            (99, 123),
            (1, 31),
            (5, 18),
            (3, 1229),
            (117, 229),
            (1, 87),
            (1, 1328),
        ]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(
        num_seqs=[1, 17],
        num_heads=[(32, 8), (12, 2), (5, 1), (3, 3)],
        head_dim=[80, 240],
        dtype=[jnp.float32, jnp.bfloat16],
        # num_kv_pages_per_block=[8, 16],
        # num_queries_per_block=[16, 32],
    )
    def test_ragged_paged_attention_complex(
        self,
        num_seqs,
        num_heads,
        head_dim,
        dtype,
        # num_kv_pages_per_block,
        # num_queries_per_block,
    ):
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            # num_kv_pages_per_block=num_kv_pages_per_block,
            # num_queries_per_block=num_queries_per_block,
        )

    @parameterized.product(sliding_window=[None, 5, 128], )
    def test_ragged_paged_attention_sliding_window(
        self,
        sliding_window: int | None,
    ):
        num_seqs = 5
        num_heads = (4, 4)
        dtype = jnp.float32
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            sliding_window=sliding_window,
        )

    @parameterized.product(soft_cap=[None, 50.0], )
    def test_ragged_paged_attention_logit_soft_capping(
        self,
        soft_cap: float | None,
    ):
        num_heads = (16, 2)
        num_seqs = 2
        dtype = jnp.float32
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            soft_cap=soft_cap,
        )

    def test_ragged_paged_attention_sliding_window_should_be_positive(self):
        dtype = jnp.float32
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._test_ragged_paged_attention(
                seq_lens,
                num_heads,
                head_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                sliding_window=0,
            )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._test_ragged_paged_attention(
                seq_lens,
                num_heads,
                head_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                sliding_window=-1,
            )

    def test_ragged_paged_attention_soft_cap_cannot_be_zero(self):
        dtype = jnp.float32
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        with self.assertRaisesRegex(ValueError, "must not be 0.0"):
            self._test_ragged_paged_attention(
                seq_lens,
                num_heads,
                head_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                soft_cap=0.0,
            )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
