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
"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing, get_tpu_version, next_power_of_2)


class RpaCase(Enum):
    """Represents the different cases for Ragged Paged Attention.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """
    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution):
        assert distribution.shape == (3, )
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        elif self == RpaCase.PREFILL:
            return distribution[0], distribution[1]
        elif self == RpaCase.MIXED:
            return distribution[1], distribution[2]
        else:
            raise ValueError(f"Unsupported RPA case: {self}")


def ref_ragged_paged_attention(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    if out_dtype is None:
        out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        # We do not set to -inf directly because (-inf) - (-inf) is nan.
        mask_value = jnp.finfo(out_dtype).min
    dynamic_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    merged_kv = merge_kv(keys, values)
    assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
        kv_cache.shape)
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0
    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert head_dim % 128 == 0
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    outputs = []

    for i in range(distribution[-1]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        indices_start = i * pages_per_seq
        indices_end = indices_start + cdiv(kv_len, page_size)
        indices = page_indices[indices_start:indices_end]
        q = queries[q_start:q_end, :, :actual_head_dim]

        # Update the kv cache.
        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        gathered_kv = gathered_kv.at[kv_len - q_len:kv_len].set(
            merged_kv[q_start:q_end])
        kv_cache = kv_cache.at[indices].set(
            gathered_kv.reshape(gathered_shape))

        kv = gathered_kv.reshape(
            -1, num_kv_heads_x2,
            head_dim)[:, :actual_num_kv_heads * 2, :].reshape(
                -1, actual_num_kv_heads, head_dim * 2)
        k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        attn = jnp.einsum("qhd,khd->hqk",
                          q,
                          k,
                          preferred_element_type=jnp.float32).astype(out_dtype)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if use_causal_mask:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span >= kv_span
            if sliding_window is not None:
                mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
            attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    return result, kv_cache


def get_smem_estimate_bytes(max_num_seqs, pages_per_seq):
    total_bits = (
        # kv_lens_ref: i32[max_num_seqs]
        align_to(max_num_seqs, 128) * 32 +
        # page_indices_ref: i32[max_num_seqs * pages_per_seq]
        align_to(max_num_seqs * pages_per_seq, 128) * 32 +
        # cu_q_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32 +
        # distribution_ref: i32[3]
        128 * 32 +
        # sem_ids_ref: i32[3]
        128 * 32 +
        # bo_ids_ref: i32[4]
        128 * 32 +
        # bkv_update_ids_ref: i32[6]
        128 * 32)
    return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    bkv_stride = cdiv(actual_num_kv_heads * 2, kv_packing)
    if has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * bkv_stride * kv_packing * head_dim) *
        (32 // kv_packing) +
        # bq_x2_ref + bo_x2_ref
        2 * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head *
             head_dim) * (32 // q_packing) +
        # l_ref + m_ref
        2 *
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32 +
        # acc_ref
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) *
        32)
    return cdiv(total_bits, 8)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


def _ragged_paged_attention_kernel(*args, **kwargs):
    distribution_ref = args[3]
    kv_lens_ref = args[0]
    cu_q_lens_ref = args[2]
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(distribution_ref)
    bq_csz = kwargs["bq_csz"]
    bkv_csz = kwargs["bkv_csz"]
    # kv_cache_hbm_ref is at args[9] (7 scalar-prefetch + q + kv before it).
    page_size = args[9].shape[1]  # kv_cache_hbm_ref.shape[1]

    def outer_cond(carry):
        (group_start,) = carry
        return group_start < end_seq_idx

    def outer_body(carry):
        (group_start,) = carry
        first_q_len = cu_q_lens_ref[group_start + 1] - cu_q_lens_ref[group_start]
        first_kv_len = kv_lens_ref[group_start]
        first_kv_padded = align_to(first_kv_len, page_size)

        def inner_cond(inner_carry):
            group_end, total_q, total_kv_padded = inner_carry
            in_range = group_end < end_seq_idx
            next_q = cu_q_lens_ref[group_end + 1] - cu_q_lens_ref[group_end]
            next_kv = kv_lens_ref[group_end]
            next_kv_padded = align_to(next_kv, page_size)
            return (in_range & ((total_q + next_q) <= bq_csz) &
                    ((total_kv_padded + next_kv_padded) <= bkv_csz))

        def inner_body(inner_carry):
            group_end, total_q, total_kv_padded = inner_carry
            next_q = cu_q_lens_ref[group_end + 1] - cu_q_lens_ref[group_end]
            next_kv = kv_lens_ref[group_end]
            next_kv_padded = align_to(next_kv, page_size)
            return group_end + 1, total_q + next_q, total_kv_padded + next_kv_padded

        group_end, _, _ = lax.while_loop(
            inner_cond,
            inner_body,
            (group_start + 1, first_q_len, first_kv_padded),
        )

        _ragged_paged_attention_kernel_loop(
            group_start,
            group_end,
            *args,
            **kwargs,
        )
        return (group_end,)

    lax.while_loop(outer_cond, outer_body, (start_seq_idx,))


def _ragged_paged_attention_kernel_loop(
    start_group_seq_id,
    end_group_seq_id,
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    # TODO(jevinjiang): merge these into one so we can save SMEM.
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    # Input
    q_hbm_ref,  # [max_num_tokens, actual_num_kv_heads, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    # Output
    o_hbm_ref,  # [max_num_tokens, actual_num_kv_heads, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    updated_kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    # Scratch
    ## Add one extra to handle bank conflicts for strided load if needed.
    bkv_x2_ref,  # [2, bkv_sz, num_kv_heads_x2 // kv_packing (+ 1), kv_packing, head_dim]
    bq_x2_ref,  # [2, bq_sz, actual_num_kv_heads, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, bq_sz, actual_num_kv_heads, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    sems,  # [4, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim],
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    static_q_len: int | None = None,
    bq_sz,  # bq fetch size
    bkv_sz,  # bkv prefetch size
    bq_csz,  # bq compute size
    bkv_csz,  # bkv compute size
    case: RpaCase = RpaCase.MIXED,
    debug_mode: bool = False,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]

    if case == RpaCase.DECODE:
        use_causal_mask = False

    out_dtype = acc_ref.dtype
    (
        max_num_tokens,
        actual_num_kv_heads,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        total_num_pages,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    bkv_stride = bkv_x2_ref.shape[2]
    assert bkv_stride in (
        num_kv_heads_x2_per_kv_packing,
        num_kv_heads_x2_per_kv_packing + 1,
    )
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    # num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    assert bkv_sz % page_size == 0
    bkv_p = bkv_sz // page_size
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)
    num_seqs = end_seq_idx - start_seq_idx

    # seq_idx alias for single-sequence backward compat (first seq in group).
    seq_idx = start_group_seq_id

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]
    kv_q_gap = kv_len - q_len
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0

    # ---------- Group precomputation (multi-seq path) ----------
    # Maximum number of sequences we can pack into one group.  Set conservatively;
    # sequences beyond this count fall into separate groups.
    _MAX_GROUP_SIZE = 32

    # We compute cumulative Q and KV lengths by plain accumulation so that all
    # intermediate values remain JAX scalars.  Creating a jnp.stack() VREG array
    # and indexing it inside Pallas TPU kernels triggers an unsupported
    # dynamic_slice primitive, so we avoid any array construction here.
    _tmp_cu_q = jnp.int32(0)
    _tmp_cu_kv = jnp.int32(0)
    for _gi in range(_MAX_GROUP_SIZE):
        _s_tmp = start_group_seq_id + _gi
        _active_tmp = _s_tmp < end_group_seq_id
        _q_tmp = jnp.where(
            _active_tmp, cu_q_lens_ref[_s_tmp + 1] - cu_q_lens_ref[_s_tmp], 0)
        _kv_tmp = jnp.where(_active_tmp, kv_lens_ref[_s_tmp], 0)
        _tmp_cu_q = _tmp_cu_q + _q_tmp
        _tmp_cu_kv = _tmp_cu_kv + align_to(_kv_tmp, page_size)

    group_total_q_len = _tmp_cu_q
    group_total_kv_padded = _tmp_cu_kv
    group_q_start = cu_q_lens_ref[start_group_seq_id]
    is_group = end_group_seq_id > start_group_seq_id + 1
    # -----------------------------------------------------------

    if sliding_window is not None:
        # TODO(jevinjiang): can skip by page_size instead of bkv_sz.
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - sliding_window,
                                            0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_q_start = cu_q_lens_ref[next_seq_idx]
        next_q_end = cu_q_lens_ref[next_seq_idx + 1]
        next_q_len = next_q_end - next_q_start
        next_kv_len = kv_lens_ref[next_seq_idx]
        next_kv_q_gap = next_kv_len - next_q_len
        next_seq_start_bkv_idx = (
            jnp.maximum(next_kv_q_gap - sliding_window, 0) // bkv_sz)

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    debug_print("[RPA debug] ======= In loop seq_idx={}", seq_idx)
    debug_print("[RPA debug] start_seq_idx={}", start_seq_idx)
    debug_print("[RPA debug] end_seq_idx={}", end_seq_idx)
    debug_print("[RPA debug] num_seqs={}", num_seqs)
    debug_print("[RPA debug] bkv_p={}", bkv_p)
    debug_print("[RPA debug] page_size={}", page_size)
    debug_print("[RPA debug] pages_per_seq={}", pages_per_seq)
    debug_print("[RPA debug] bkv_sz={}", bkv_sz)
    debug_print("[RPA debug] bq_sz={}", bq_sz)
    debug_print(f"[RPA debug] static_q_len={static_q_len}")
    debug_print("[RPA debug] q_start={}", q_start)
    debug_print("[RPA debug] q_end={}", q_end)
    debug_print("[RPA debug] q_len={}", q_len)
    debug_print("[RPA debug] kv_len={}", kv_len)
    debug_print("[RPA debug] kv_q_gap={}", kv_q_gap)
    debug_print(f"[RPA debug] sliding_window={sliding_window}")
    debug_print("[RPA debug] cur_seq_start_bkv_idx={}", cur_seq_start_bkv_idx)
    debug_print("[RPA debug] next_seq_start_bkv_idx={}",
                next_seq_start_bkv_idx)

    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
        k,  # [bkv_csz, head_dim]
        v,  # [bkv_csz, head_dim]
        l_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        m_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        *,
        processed_q_len,
        processed_kv_len,
        effective_kv_len,
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == head_dim
        actual_bq_csz = q.shape[0] // num_q_heads_per_kv_head
        assert k.shape == (bkv_csz, head_dim)
        assert v.shape == (bkv_csz, head_dim)
        assert l_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert m_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert k.dtype == v.dtype

        # Follow FlashAttention-2 forward pass.
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        s = jnp.matmul(q, k.T,
                       preferred_element_type=jnp.float32).astype(out_dtype)
        s *= sm_scale
        if k_scale is not None:
            s *= k_scale
        if q_scale is not None:
            s *= q_scale
        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        int_ty = jnp.int32
        if get_dtype_packing(q_dtype) != 1 and get_tpu_version() >= 6:
            int_ty = jnp.int16
        processed_q_len_int = processed_q_len.astype(int_ty)
        processed_kv_len_int = processed_kv_len.astype(int_ty)
        effective_kv_len_int = effective_kv_len.astype(int_ty)
        q_span = processed_q_len_int + (lax.broadcasted_iota(
            jnp.int32, s.shape, 0) // num_q_heads_per_kv_head).astype(int_ty)
        k_span = processed_kv_len_int + lax.broadcasted_iota(
            int_ty, s.shape, 1)
        v_span = processed_kv_len_int + lax.broadcasted_iota(
            int_ty, v.shape, 0)

        mask = None
        if use_causal_mask:
            assert not skip_kv_mask
            mask = mask_and(mask, q_span >= k_span)

        if not skip_kv_mask:
            mask = mask_and(mask, k_span < effective_kv_len_int)
            v = jnp.where(v_span < effective_kv_len_int, v, 0.0)

        if sliding_window is not None:
            mask = mask_and(mask, q_span < k_span + sliding_window)

        if mask is not None:
            s = jnp.where(mask, s, mask_value)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = m_ref[...]
        m_curr = jnp.maximum(m_prev, s_rowmax)
        m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=out_dtype)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_ref[...]
        l_ref[...] = exp_m_diff * l_prev + p_rowsum

        return p, v, exp_m_diff

    def flash_attention_step2_pv(
            p,  # [actual_bq_csz * num_q_heads_per_kv_head, bkv_csz]
            v,  # [bkv_csz, head_dim]
            exp_m_diff,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
            o_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
    ):
        assert len(p.shape) == 2
        assert p.shape[0] % num_q_heads_per_kv_head == 0
        assert p.shape[1] == bkv_csz
        actual_bq_csz = p.shape[0] // num_q_heads_per_kv_head
        assert v.shape == (bkv_csz, head_dim)
        assert exp_m_diff.shape == (actual_bq_csz * num_q_heads_per_kv_head,
                                    128)
        assert o_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head,
                               head_dim)
        pv = jnp.matmul(p, v,
                        preferred_element_type=jnp.float32).astype(out_dtype)
        if v_scale is not None:
            pv *= v_scale
        o_prev = o_ref[...]
        o_ref[...] = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            # Skip DMA if debug mode is enabled.
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[
            bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:])
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache,
                                     kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bkv-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_idx={}", bkv_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] kv_len_start={}", kv_len_start)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_left={}", kv_left)
        debug_print("[RPA debug] kv_left_frm_cache={}", kv_left_frm_cache)
        debug_print("[RPA debug] kv_left_frm_new={}", kv_left_frm_new)
        debug_print("[RPA debug] bkv_sz_frm_cache={}", bkv_sz_frm_cache)
        debug_print("[RPA debug] bkv_sz_frm_new={}", bkv_sz_frm_new)
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop.
            for i in range(bkv_p):
                # Ensure only effective kvs are copied.
                sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
                # If the page index is out of bound, we set page_idx to the last page.
                # And there will be no copy since sz will be 0.
                page_idx = jnp.minimum(page_indices_offset + i,
                                       num_page_indices - 1)
                _async_copy(
                    cache_hbm_ref.at[pl.ds(
                        page_indices_ref[page_idx] * page_size, sz)],
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )
                debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)

            new_kv_len_start = q_end - kv_left_frm_new
            debug_print("[RPA debug] new_kv_len_start={}", new_kv_len_start)
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, bkv_sz_frm_cache + bkv_sz_frm_new)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _update_kv_cache(seq_idx,
                         bkv_sem_idx,
                         offset,
                         update_sz,
                         *,
                         wait=False,
                         vmem_base_offset=0):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[
            bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:])

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_update_kv_cache-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] offset={}", offset)
        debug_print("[RPA debug] update_sz={}", update_sz)
        debug_print("[RPA debug] bkv_id={}", bkv_id)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_p_end={}", kv_p_end)
        debug_print("[RPA debug] ignore={}", ignore)
        debug_print("[RPA debug] p_ignore={}", p_ignore)
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)

            # vmem_base_offset shifts the read position for group seqs whose
            # KV data does not start at offset 0 of the VMEM buffer.
            vmem_pos = vmem_base_offset + (p_ignore + i) * page_size + ignore
            _async_copy(
                vmem_ref.at[pl.ds(vmem_pos, sz)],
                cache_hbm_ref.at[pl.ds(
                    page_indices_ref[page_indices_offset + i] * page_size +
                    ignore,
                    sz,
                )],
                sem,
                wait,
            )
            debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)
            return update_sz - sz, 0

        if not wait:
            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),  # total transfer size
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bq-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bq_idx={}", bq_idx)
        debug_print("[RPA debug] bq_sem_idx={}", bq_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

        _async_copy(
            q_hbm_ref.at[pl.ds(q_len_start, sz)],
            vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_send_bo-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bo_idx={}", bo_idx)
        debug_print("[RPA debug] bo_sem_idx={}", bo_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(
            jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx
                            <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx,
                             bkv_sem_idx,
                             offset,
                             update_sz,
                             wait=True)

    def strided_load(ref, start, sz, step, *, dtype=None):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        vec = jnp.concat(
            [ref[pl.ds(start + i, sz // step, step)] for i in range(folds)],
            axis=1)
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def load_bq(bq_sem_idx, kv_head_idx, start, sz):
        q_block = pltpu.bitcast(
                bq_x2_ref.at[bq_sem_idx, pl.ds(start, sz)][...],
                q_dtype)
        q_block = q_block[:, kv_head_idx]  # [sz, num_q_heads_per_kv_head_per_packing, q_packing, head_dim]
        return q_block.reshape(sz * num_q_heads_per_kv_head, head_dim)


    def load_bkv(bkv_sem_idx, kv_head_idx, start, sz):
        start *= bkv_stride
        sz *= bkv_stride
        step = bkv_stride
        kv_ref = (bkv_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(
            bkv_sz * step, head_dim))

        if kv_packing == 1:
            start += kv_head_idx * 2
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            k = pltpu.bitcast(k, kv_dtype)
            v = pltpu.bitcast(v, kv_dtype)
            return k, v

        num_kv_per_load = kv_packing // 2
        offset = kv_head_idx // num_kv_per_load
        kv_idx_in_load = kv_head_idx % num_kv_per_load
        kv = strided_load(kv_ref, start + offset, sz, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k = kv >> (kv_idx_in_load * 2 * bitwidth)
        v = k >> bitwidth
        k = pltpu.bitcast(k.astype(repack_ty), kv_dtype)
        v = pltpu.bitcast(v.astype(repack_ty), kv_dtype)
        return k, v

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        # no-op concatenation.
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])],
            axis=-1)[..., :shape[-1]]

    def mask_and(mask, new_mask):
        if mask is None:
            return new_mask
        return jnp.logical_and(mask, new_mask)

    # ------------------------------------------------------------------ #
    #  Group-level helpers (used when is_group == True)                  #
    # ------------------------------------------------------------------ #

    def _fetch_bkv_group(bkv_sem_idx, *, wait=False):
        """Load KV for every seq in the group into bkv_x2_ref[bkv_sem_idx].

        Each seq's KV occupies [cu_kv, cu_kv + align_to(kv_len, page_size))
        in the VMEM buffer.  The offsets are page-aligned so KV-cache scatter
        writes work correctly.  We compute the offsets by plain scalar
        accumulation (no jnp.stack / array indexing) to avoid unsupported
        dynamic_slice in the Pallas TPU lowering.
        """
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[
            bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:])

        if not wait:
            wait_update_kv_cache(bkv_sem_idx)
            _cu_kv_fetch = jnp.int32(0)
            for _gi in range(_MAX_GROUP_SIZE):
                _s = start_group_seq_id + _gi
                _active = _s < end_group_seq_id
                _kv_s = jnp.where(_active, kv_lens_ref[_s], 0)
                _q_s = jnp.where(
                    _active,
                    cu_q_lens_ref[_s + 1] - cu_q_lens_ref[_s], 0)
                _kv_s_padded = align_to(_kv_s, page_size)
                # Capture current cumulative offset and per-seq data as
                # scalars so the @pl.when closure sees the right values.
                _vmem_base = _cu_kv_fetch
                _kv_frm_cache = jnp.maximum(_kv_s - _q_s, 0)
                _kv_frm_new = _kv_s - _kv_frm_cache
                _q_end_s = cu_q_lens_ref[_s + 1]
                _page_off = _s * pages_per_seq

                @pl.when(_s < end_group_seq_id)
                def _fetch_one_seq(
                        _vmem_base=_vmem_base,
                        _kv_frm_cache=_kv_frm_cache,
                        _kv_frm_new=_kv_frm_new,
                        _q_end_s=_q_end_s,
                        _page_off=_page_off):
                    # Fetch from paged KV cache.
                    for _p in range(bkv_p):
                        _sz = jnp.clip(
                            _kv_frm_cache - _p * page_size, 0, page_size)
                        _pg = jnp.minimum(
                            _page_off + _p, num_page_indices - 1)
                        _async_copy(
                            cache_hbm_ref.at[pl.ds(
                                page_indices_ref[_pg] * page_size, _sz)],
                            vmem_ref.at[pl.ds(
                                _vmem_base + _p * page_size, _sz)],
                            sem,
                            wait=False,
                        )
                    # Fetch new KV tokens (from the current prefill step).
                    _new_kv_src = _q_end_s - _kv_frm_new
                    _async_copy(
                        kv_hbm_ref.at[pl.ds(_new_kv_src, _kv_frm_new)],
                        vmem_ref.at[pl.ds(
                            _vmem_base + _kv_frm_cache, _kv_frm_new)],
                        sem,
                        wait=False,
                    )

                _cu_kv_fetch = _cu_kv_fetch + _kv_s_padded
        else:
            # Dummy self-copy: flushes the semaphore (waits for all starts).
            dst = vmem_ref.at[pl.ds(0, group_total_kv_padded)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def _fetch_bq_group(bq_sem_idx, *, wait=False):
        """Load Q for all seqs in the group (contiguous in HBM)."""
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        if not wait:
            _async_copy(
                q_hbm_ref.at[pl.ds(group_q_start, group_total_q_len)],
                vmem_ref.at[pl.ds(0, group_total_q_len)],
                sem,
                wait=False,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, group_total_q_len)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def _send_bo_group(bo_sem_idx, *, wait=False):
        """DMA output for the entire group back to HBM."""
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        if not wait:
            _async_copy(
                vmem_ref.at[pl.ds(0, group_total_q_len)],
                o_hbm_ref.at[pl.ds(group_q_start, group_total_q_len)],
                sem,
                wait=False,
            )
        else:
            dst = o_hbm_ref.at[pl.ds(group_q_start, group_total_q_len)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def _update_kv_cache_for_group_seq(
            seq, kv_frm_new, kv_frm_cache, vmem_base, bkv_sem_idx):
        """Write new prefill tokens for one seq in the group to the KV cache.

        All arguments are scalars (not array elements) to avoid dynamic_slice.
        """
        @pl.when(kv_frm_new > 0)
        def _():
            _update_kv_cache(
                seq, bkv_sem_idx, kv_frm_cache, kv_frm_new,
                vmem_base_offset=vmem_base, wait=False)
            _update_kv_cache(
                seq, bkv_sem_idx, kv_frm_cache, kv_frm_new,
                vmem_base_offset=vmem_base, wait=True)

    def process_group():
        """Process all sequences in the group together with one MXU call.

        This is the multi-seq fast path: when the group's combined Q length
        fits in bq_csz and its combined KV length fits in bkv_csz we can
        concatenate the data, apply a cross-sequence causal mask, and issue a
        single matmul instead of one matmul per sequence.
        """
        l_ref[...] = jnp.full_like(l_ref, 0.0)
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        acc_ref[...] = jnp.full_like(acc_ref, 0.0)

        bq_sem_idx = sem_ids_ref[0]
        bkv_sem_idx = sem_ids_ref[1]
        bo_sem_idx = sem_ids_ref[2]

        # Start async fetches.
        _fetch_bq_group(bq_sem_idx, wait=False)
        _fetch_bkv_group(bkv_sem_idx, wait=False)

        # Wait for fetches to complete.
        _fetch_bq_group(bq_sem_idx, wait=True)
        _fetch_bkv_group(bkv_sem_idx, wait=True)

        # ---- Flash attention (one step, no inner loops needed) ----
        # The grouping condition guarantees group_total_q_len <= bq_csz and
        # group_total_kv_padded <= bkv_csz, so a single bq_csz × bkv_csz tile
        # covers the entire group.  We set bq_start=0 (no outer Q-block loop).
        #
        # Masking strategy: instead of building large bool arrays (which can
        # trigger "unsupported shape cast" in Mosaic), we compute s and bv
        # incrementally using scalar-boundary accumulators.
        # • s_masked starts all-mask_value; per-seq valid positions are
        #   revealed by jnp.where with the seq mask.
        # • bv_masked starts all-zero; valid KV rows are revealed similarly.
        # Both loops share the same scalar boundary accumulators.
        # Always use int32 inside process_group.  Mosaic does not support i16
        # arith.addi, so int16 iotas would break the mask accumulation loop.
        _mask_shape = (bq_csz * num_q_heads_per_kv_head, bkv_csz)
        _q_tok_iota = (
            lax.broadcasted_iota(jnp.int32, _mask_shape, 0) //
            num_q_heads_per_kv_head)                  # [bq_csz*H, bkv_csz] i32
        _kv_iota = lax.broadcasted_iota(
            jnp.int32, _mask_shape, 1)                # [bq_csz*H, bkv_csz] i32
        # Use a 2D iota for v masking so we never need [:, None] reshapes —
        # Mosaic does not support shape-casting bool arrays.
        _v_iota_2d = lax.broadcasted_iota(
            jnp.int32, (bkv_csz, head_dim), 0)        # [bkv_csz, head_dim] i32

        prev_lm_slice = None
        prev_p = None
        prev_v = None
        prev_exp_m_diff = None

        bq_start = 0
        lm_slice_size = bq_csz * num_q_heads_per_kv_head
        for kv_head_idx in range(actual_num_kv_heads):
                bk_c, bv_c = load_bkv(bkv_sem_idx, kv_head_idx, 0, bkv_csz)
                bq_c = load_bq(bq_sem_idx, kv_head_idx, bq_start, bq_csz)

                lm_slice = (kv_head_idx, pl.ds(0, lm_slice_size))

                # Compute Q×K^T with the same scaling as flash_attention_step1.
                if q_scale is not None:
                    bq_eff = bq_c / q_scale
                    if jnp.issubdtype(bk_c.dtype, jnp.floating):
                        dtype_info = jnp.finfo(bk_c.dtype)
                        bq_eff = jnp.clip(bq_eff, float(dtype_info.min),
                                          float(dtype_info.max))
                    bq_eff = bq_eff.astype(bk_c.dtype)
                else:
                    bq_eff = bq_c
                s_raw = jnp.matmul(
                    bq_eff, bk_c.T,
                    preferred_element_type=jnp.float32).astype(out_dtype)
                s_raw = s_raw * sm_scale
                if k_scale is not None:
                    s_raw = s_raw * k_scale
                if q_scale is not None:
                    s_raw = s_raw * q_scale
                if soft_cap is not None:
                    s_raw = soft_cap * jnp.tanh(s_raw / soft_cap)

                # Build s_masked and bv_masked by accumulating per-seq masks.
                # Using incremental jnp.where avoids allocating large bool
                # arrays that cause Mosaic shape-cast errors.
                s_masked = jnp.full(_mask_shape, mask_value, dtype=out_dtype)
                bv_masked = jnp.zeros((bkv_csz, head_dim), dtype=bv_c.dtype)

                _cu_q_m2 = jnp.int32(0)
                _cu_kv_m2 = jnp.int32(0)
                for _gi2 in range(_MAX_GROUP_SIZE):
                    _s_m2 = start_group_seq_id + _gi2
                    _active_m2 = _s_m2 < end_group_seq_id
                    _q_s_m2 = jnp.where(
                        _active_m2,
                        cu_q_lens_ref[_s_m2 + 1] - cu_q_lens_ref[_s_m2], 0)
                    _kv_s_m2 = jnp.where(_active_m2, kv_lens_ref[_s_m2], 0)
                    _kv_pad_m2 = align_to(_kv_s_m2, page_size)

                    # All arithmetic in int32 — Mosaic does not support i16 addi.
                    _cu_q_lo2 = _cu_q_m2            # int32
                    _cu_q_hi2 = _cu_q_m2 + _q_s_m2  # int32
                    _cu_kv_lo2 = _cu_kv_m2           # int32
                    _cu_kv_hi2 = _cu_kv_m2 + _kv_pad_m2  # int32
                    _kv_q_gap2 = _kv_s_m2 - _q_s_m2  # int32

                    _q_in2 = ((_cu_q_lo2 <= _q_tok_iota) &
                               (_q_tok_iota < _cu_q_hi2))
                    _kv_in2 = ((_cu_kv_lo2 <= _kv_iota) &
                                (_kv_iota < _cu_kv_hi2))
                    _same2 = _q_in2 & _kv_in2
                    _local_q2 = _q_tok_iota - _cu_q_lo2   # int32
                    _local_kv2 = _kv_iota - _cu_kv_lo2    # int32

                    _smask2 = _same2
                    if use_causal_mask:
                        _smask2 = _smask2 & (
                            _kv_q_gap2 + _local_q2 >= _local_kv2)
                    if not skip_kv_mask:
                        _smask2 = _smask2 & (_local_kv2 < _kv_s_m2)
                    if sliding_window is not None:
                        _smask2 = _smask2 & (
                            _kv_q_gap2 + _local_q2 <
                            _local_kv2 + jnp.int32(sliding_window))

                    s_masked = jnp.where(_smask2, s_raw, s_masked)

                    # v masking: zero out padding rows within this seq's slab.
                    # Use _v_iota_2d (shape bkv_csz × head_dim) so no reshape
                    # is needed — Mosaic can't cast bool arrays via shape cast.
                    _v_in2 = ((_cu_kv_lo2 <= _v_iota_2d) &
                               (_v_iota_2d < _cu_kv_lo2 + _kv_s_m2))
                    bv_masked = jnp.where(_v_in2, bv_c, bv_masked)

                    _cu_q_m2 = _cu_q_m2 + _q_s_m2
                    _cu_kv_m2 = _cu_kv_m2 + _kv_pad_m2

                # Flash-attention softmax on the masked scores.
                s_rowmax = jnp.max(s_masked, axis=1, keepdims=True)
                m_prev = m_ref.at[*lm_slice][...]
                m_curr = jnp.maximum(m_prev, s_rowmax)
                m_ref.at[*lm_slice][...] = m_curr
                p_cur = jnp.exp(
                    s_masked - broadcast_minor(m_curr, s_masked.shape))
                p_rowsum = jnp.sum(p_cur, axis=1, keepdims=True,
                                   dtype=out_dtype)
                exp_m_diff = jnp.exp(m_prev - m_curr)
                l_prev = l_ref.at[*lm_slice][...]
                l_ref.at[*lm_slice][...] = exp_m_diff * l_prev + p_rowsum

                if prev_lm_slice is not None:
                    pv = jnp.matmul(
                        prev_p, prev_v,
                        preferred_element_type=jnp.float32).astype(out_dtype)
                    if v_scale is not None:
                        pv = pv * v_scale
                    o_prev = acc_ref.at[*prev_lm_slice][...]
                    acc_ref.at[*prev_lm_slice][...] = (
                        broadcast_minor(prev_exp_m_diff, o_prev.shape) *
                        o_prev + pv)

                prev_lm_slice = lm_slice
                prev_p = p_cur
                prev_v = bv_masked   # already zero-padded in mask loop above
                prev_exp_m_diff = exp_m_diff

        # Execute pv of the last iteration.
        if prev_lm_slice is not None:
            pv = jnp.matmul(
                prev_p, prev_v,
                preferred_element_type=jnp.float32).astype(out_dtype)
            if v_scale is not None:
                pv = pv * v_scale
            o_prev = acc_ref.at[*prev_lm_slice][...]
            acc_ref.at[*prev_lm_slice][...] = (
                broadcast_minor(prev_exp_m_diff, o_prev.shape) * o_prev + pv)

        # ---- Finalise output ----
        acc = acc_ref[...]
        l_vals = broadcast_minor(l_ref[...], acc.shape)
        out = (acc * pl.reciprocal(l_vals, approx=True)
               if (l_vals.dtype == jnp.float32 and out_dtype != jnp.float32)
               else lax.div(acc, l_vals)).astype(out_dtype)

        # Pack into bo_x2_ref layout and send to HBM.
        bo_new_sem = lax.select(bo_sem_idx == 0, 1, 0)
        sem_ids_ref[2] = bo_new_sem
        wait_send_bo(bo_sem_idx)

        out_ref = (bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
            bq_sz,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_packing,
            head_dim,
        ))
        out_packed = pltpu.bitcast(out, out_ref.dtype).reshape(
            actual_num_kv_heads,
            bq_sz,
            num_q_heads_per_kv_head_per_packing,
            head_dim,
        ).swapaxes(0, 1)
        out_ref[...] = out_packed

        # Send output for all group Q tokens.
        bo_ids_ref[bo_sem_idx] = start_group_seq_id
        bo_ids_ref[bo_sem_idx + 2] = jnp.int32(0)
        _send_bo_group(bo_sem_idx, wait=False)

        # Update KV cache: write new prefill tokens for each seq in the group.
        # Use scalar accumulators to avoid array indexing (dynamic_slice).
        _cu_kv_upd = jnp.int32(0)
        for _gi in range(_MAX_GROUP_SIZE):
            _s_u = start_group_seq_id + _gi
            _active_u = _s_u < end_group_seq_id
            _kv_u = jnp.where(_active_u, kv_lens_ref[_s_u], 0)
            _q_u = jnp.where(
                _active_u,
                cu_q_lens_ref[_s_u + 1] - cu_q_lens_ref[_s_u], 0)
            _kv_pad_u = align_to(_kv_u, page_size)
            _vmem_base_u = _cu_kv_upd          # capture current offset
            _kv_frm_cache_u = jnp.maximum(_kv_u - _q_u, 0)
            _kv_frm_new_u = _kv_u - _kv_frm_cache_u

            @pl.when(_s_u < end_group_seq_id)
            def _kv_upd(
                    _s_u=_s_u,
                    _kv_frm_new_u=_kv_frm_new_u,
                    _kv_frm_cache_u=_kv_frm_cache_u,
                    _vmem_base_u=_vmem_base_u):
                _update_kv_cache_for_group_seq(
                    _s_u, _kv_frm_new_u, _kv_frm_cache_u,
                    _vmem_base_u, bkv_sem_idx)

            _cu_kv_upd = _cu_kv_upd + _kv_pad_u

    # ------------------------------------------------------------------ #
    #  (end of group helpers)                                             #
    # ------------------------------------------------------------------ #

    def process(static_q_len=None):
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        actual_bq_csz = min(bq_csz, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *,
                             num_bkv):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            next_bq_start_bkv_idx = 0
            if sliding_window is not None:
                next_bq_start_bkv_idx = (jnp.maximum(
                    kv_q_gap +
                    (bq_idx + 1) * actual_bq_sz - sliding_window, 0) // bkv_sz)
            next_bkv_idx = lax.select(is_last_bkv, next_bq_start_bkv_idx,
                                      next_bkv_idx)
            next_bkv_idx = lax.select(is_last_bq, next_seq_start_bkv_idx,
                                      next_bkv_idx)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            # Re-initialize l, m, acc to 0 before bkv loop.
            l_ref[...] = jnp.full_like(l_ref, 0.0)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx)

            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = 0
            if sliding_window is not None:
                # Recalculate the start_bkv_idx based on the processed_q_len.
                start_bkv_idx = (
                    jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz)
            if use_causal_mask:
                effective_kv_len = jnp.minimum(kv_len,
                                               processed_q_len + actual_bq_sz)
            else:
                effective_kv_len = kv_len
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                assert bkv_sz % kv_packing == 0

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx)
                processed_kv_len = bkv_idx * bkv_sz

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx,
                                                   bkv_sem_idx)

                # Start updating bkv to kv cache if applicable.
                # Only needed in last bq loop.
                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == num_bq - 1))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset,
                                          update_sz)

                debug_print(
                    "[RPA debug] -----------flash attention-----------")
                debug_print("[RPA debug] seq_idx={}", seq_idx)
                debug_print("[RPA debug] bq_idx={}", bq_idx)
                debug_print("[RPA debug] bkv_idx={}", bkv_idx)
                if debug_mode:
                    # Skip flash attention if debug mode is enabled.
                    return

                # Flash attention with cur bkv and bq
                effective_bkv_sz = jnp.minimum(
                    effective_kv_len - bkv_idx * bkv_sz, bkv_sz)
                effective_bkv_sz = jnp.maximum(effective_bkv_sz, 0)

                num_loops = cdiv(effective_bkv_sz, bkv_csz)

                @pl.loop(0, num_loops, unroll=False)
                def attention_loop(idx):
                    prev_lm_slice = None
                    prev_p = None
                    prev_v = None
                    prev_exp_m_diff = None
                    bkv_start = idx * bkv_csz

                    for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                        for kv_head_idx in range(actual_num_kv_heads):
                            bk_c, bv_c = load_bkv(
                                bkv_sem_idx,
                                kv_head_idx,
                                bkv_start,
                                bkv_csz,
                            )
                            bq_c = load_bq(bq_sem_idx, kv_head_idx, bq_start,
                                           actual_bq_csz)

                            lm_slice_start = bq_start * num_q_heads_per_kv_head
                            lm_slice_size = actual_bq_csz * num_q_heads_per_kv_head
                            lm_slice = (kv_head_idx,
                                        pl.ds(lm_slice_start, lm_slice_size))

                            # FlashAttn is divided into `flash_attention_step1_qk_softmax`
                            # and `flash_attention_step2_pv` to pipeline the computation.
                            # `step2_pv` for the previous KV head, which depends on the
                            # softmax output, is overlapped with `step1_qk_softmax` for the
                            # current KV head, reducing overall wait times.
                            cur_p, cur_v, cur_exp_m_diff = flash_attention_step1_qk_softmax(
                                bq_c,
                                bk_c,
                                bv_c,
                                l_ref.at[*lm_slice],
                                m_ref.at[*lm_slice],
                                processed_q_len=processed_q_len + bq_start,
                                processed_kv_len=processed_kv_len + bkv_start,
                                effective_kv_len=effective_kv_len,
                            )
                            if prev_lm_slice is not None:
                                flash_attention_step2_pv(
                                    prev_p,
                                    prev_v,
                                    prev_exp_m_diff,
                                    acc_ref.at[*prev_lm_slice],
                                )
                            prev_lm_slice = lm_slice
                            prev_p = cur_p
                            prev_v = cur_v
                            prev_exp_m_diff = cur_exp_m_diff

                    # Execute pv of last iteration.
                    assert prev_lm_slice is not None
                    flash_attention_step2_pv(
                        prev_p,
                        prev_v,
                        prev_exp_m_diff,
                        acc_ref.at[*prev_lm_slice],
                    )

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)  # noqa
            out = (acc * pl.reciprocal(l, approx=True) if
                   (l.dtype == jnp.float32 and out_dtype != jnp.float32) else
                   lax.div(acc, l)).astype(out_dtype)

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            out_ref = (bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                bq_sz,
                actual_num_kv_heads,
                num_q_heads_per_kv_head_per_packing,
                head_dim,
            ))
            out = pltpu.bitcast(out, out_ref.dtype).reshape(
                actual_num_kv_heads,
                bq_sz,
                num_q_heads_per_kv_head_per_packing,
                head_dim,
            ).swapaxes(0, 1)
            out_ref[...] = out

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(start_group_seq_id == start_seq_idx)
    def prologue():
        @pl.when(is_group)
        def _group_prologue():
            _fetch_bq_group(bq_sem_idx=0, wait=False)
            _fetch_bkv_group(bkv_sem_idx=0, wait=False)

        @pl.when(jnp.logical_not(is_group))
        def _single_prologue():
            start_fetch_bq(seq_idx=start_seq_idx, bq_idx=0, bq_sem_idx=0)
            start_fetch_bkv(seq_idx=start_seq_idx,
                            bkv_idx=cur_seq_start_bkv_idx,
                            bkv_sem_idx=0)

    @pl.when(
        jnp.logical_and(start_seq_idx <= start_group_seq_id,
                         start_group_seq_id < end_seq_idx))
    def pipeline():
        @pl.when(is_group)
        def _():
            process_group()

        @pl.when(jnp.logical_not(is_group))
        def _():
            process(static_q_len=static_q_len)

    @pl.when(end_group_seq_id == end_seq_idx)
    def epilogue():
        for i in range(2):
            wait_send_bo(bo_sem_idx=i)
            wait_update_kv_cache(bkv_sem_idx=i)

    ### ------- Kernel end ------- ###


def has_bank_conflicts(stride, distance=24, num_banks=32):
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


def merge_kv(
        k: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
        v: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concat([k, v],
                   axis=-1).reshape(max_num_tokens, actual_num_kv_heads_x2,
                                    actual_head_dim),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


def prepare_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
        k: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
        v: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        ))
    # TODO(kyuyeunk, chengjiyao): Add kv quantization here.
    kv = merge_kv(k, v)
    return q, kv


def prepare_outputs(
    out,  # [max_num_tokens, actual_num_kv_heads, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
):
    (
        max_num_tokens,
        actual_num_kv_heads,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    return (out.reshape(
        max_num_tokens,
        actual_num_kv_heads,
        num_q_heads_per_kv_head_per_q_packing * q_packing,
        head_dim,
    )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim].reshape(
        max_num_tokens, actual_num_q_heads, actual_head_dim))


# Expect to run this validation during runtime.
def dynamic_validate_inputs(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
):
    q, k, v = queries, keys, values
    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    max_num_tokens = q.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    i, j, k = distribution
    if not (i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")

    if k > max_num_seqs:
        raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")

    if cu_q_lens[k] > max_num_tokens:
        raise ValueError(
            f"Total q tokens {cu_q_lens[k]} must be <= {max_num_tokens=}.")
    for i in range(k):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        kv_len = kv_lens[i]
        if not (0 < q_len <= kv_len):
            raise ValueError(
                f"Require 0 < {q_len=} <= {kv_len=} at sequence {i}.")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Require {page_cnt=} <= {pages_per_seq=} at sequence {i} where"
                f" {kv_len=} and {page_size=}.")
        for p in range(page_cnt):
            page_idx = page_indices[i * pages_per_seq + p]
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence"
                    f" {i} where {kv_len=} and {page_size=}.")


# Expect to run this validation during compile time.
def static_validate_inputs(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate inputs to the RPA kernel statically."""
    if use_causal_mask:
        if skip_kv_mask:
            raise ValueError("Can not skip kv mask when using causal mask.")

    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(
            f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(
            f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}"
        )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(
            f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}"
        )

    actual_head_dim = q.shape[2]
    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(f"Expected {actual_num_q_heads=} to be divisible by"
                         f" {actual_num_kv_heads=}.")

    expected_kv_cache_shape = get_kv_cache_shape(
        kv_cache.shape[0],
        kv_cache.shape[1],
        actual_num_kv_heads,
        actual_head_dim,
        kv_cache.dtype,
    )

    if kv_cache.shape != expected_kv_cache_shape:
        raise ValueError(
            f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
        )

    (
        _,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        head_dim,
    ) = kv_cache.shape

    if head_dim != align_to(actual_head_dim, 128):
        raise ValueError(
            f"Expected {head_dim=} is equal to {align_to(actual_head_dim, 128)=}"
        )
    # Note: we expect the kv quantization happens outside of the RPA kernel.
    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"Expected {kv_cache.dtype=} to be equal to {k.dtype=} and {v.dtype=}."
        )
    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
    if kv_packing != get_dtype_packing(kv_cache.dtype):
        raise ValueError(
            f"{kv_packing=} does not match with {kv_cache.dtype=}")

    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    if num_kv_heads_x2 % 2 != 0:
        raise ValueError(
            f"Combined KV heads must be divisible by 2, but got {num_kv_heads_x2}"
        )
    if (num_kv_heads_x2 % kv_packing != 0
            or num_kv_heads_x2 // 2 < actual_num_kv_heads):
        raise ValueError(
            f"Invalid {num_kv_heads_x2=}, {actual_num_kv_heads=}, {kv_packing=}"
        )

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype
            == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}")

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(
            cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}")

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3, ):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")

    def _validate_block_sizes(block_sizes, prefix):
        if block_sizes is None:
            return
        bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes
        if not (0 < bq_csz and bq_sz % bq_csz == 0):
            raise ValueError(
                f"{prefix} {bq_csz=} and {bq_sz=} must satisfy (0 < bq_csz and bq_sz"
                " % bq_csz == 0).")
        if not (0 < bkv_csz and bkv_sz % bkv_csz == 0):
            raise ValueError(
                f"{prefix} {bkv_csz=} and {bkv_sz=} must satisfy (0 < bkv_csz and"
                " bkv_sz % bkv_csz == 0).")
        if bkv_sz % page_size != 0:
            raise ValueError(
                f"{prefix} {bkv_sz=} must be divisible by {page_size=}.")
        if bkv_csz % page_size != 0:
            raise ValueError(
                f"{prefix} {bkv_csz=} must be divisible by {page_size=}.")

    _validate_block_sizes(d_block_sizes, "decode")
    _validate_block_sizes(p_block_sizes, "prefill")
    _validate_block_sizes(m_block_sizes, "mixed")

    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del out_dtype
    del q_scale
    del k_scale
    del v_scale


def get_default_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    max_num_seqs,
    pages_per_seq,
    *,
    case: RpaCase = RpaCase.MIXED,
):
    """Get (bq, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas.

    Note the default block sizes are not necessarily optimal.
    """
    tpu_version = get_tpu_version()

    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = next_power_of_2(
        align_to(actual_num_kv_heads * 2, kv_packing))
    head_dim = align_to(head_dim, 128)
    num_q_heads_per_kv_head = next_power_of_2(actual_num_q_heads //
                                              actual_num_kv_heads)

    max_q = next_power_of_2(max_num_tokens)
    max_kv = pages_per_seq * page_size

    min_bkv_sz_to_peak = (16 * 1024 * 1024 * kv_packing // 4 // head_dim //
                          num_kv_heads_x2)

    match tpu_version:
        case 5 | 6:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = min(min_bkv_sz_to_peak, max_kv)
                bq_csz = 1
                bkv_csz = min(min_bkv_sz_to_peak, max_kv)
            else:
                bq_sz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(1024, max_kv)
                bq_csz = min(512 // num_q_heads_per_kv_head, max_q)
                bkv_csz = min(512, align_to(max_kv // 2, page_size))
        case 7:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = min(min_bkv_sz_to_peak, max_kv)
                bq_csz = 1
                bkv_csz = min(min_bkv_sz_to_peak, max_kv)
            else:
                bq_sz = min(2048 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(2048, max_kv)
                bq_csz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_csz = min(512, align_to(max_kv // 2, page_size))
        case _:
            raise NotImplementedError(f"Unsupported {tpu_version=}.")

    return {
        "bq_sz": max(1, bq_sz),
        "bkv_sz": align_to(bkv_sz, page_size),
        "bq_csz": max(1, bq_csz),
        "bkv_csz": align_to(bkv_csz, page_size),
    }


@jax.jit(
    static_argnames=(
        "use_causal_mask",
        "skip_kv_mask",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "out_dtype",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "d_block_sizes",
        "p_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "disable_bounds_checks",
        "disable_semaphore_checks",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # Each case takes a tuple of (bq_sz, bkv_sz, bq_csz, bkv_csz).
    # - bq_sz: the block size for the query fetching.
    # - bkv_sz: the block size for the kv fetching.
    # - bq_csz: the compute size of the block query.
    # - bkv_csz: the compute size of the block kv.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
):
    """Ragged paged attention that supports mixed prefill and decode.

  Args:
    queries: concatenated all sequences' queries.
    keys: concatenated all sequences' keys (quantized).
    values: concatenated all sequences' values (quantized).
    kv_cache: paged KV cache with TPU-friendly shape.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    use_causal_mask: if true, use causal mask.
    skip_kv_mask: only set to true if use_causal_mask=False and each dynamic
      kv_len % bkv_csz == 0. Set to true can improve performance.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    out_dtype: the dtype of the output and the accumulator for matmul. Set
      lower for better performance, set higher for better accuracy. If None, it
      uses q.dtype.
    mask_value: mask value for causal mask.
    q_scale: the scale for the query.
    k_scale: the scale for the key.
    v_scale: the scale for the value.
    chunk_prefill_size: the chunk prefill size for the attention.
    d_block_sizes: the block sizes for the decode case.
    p_block_sizes: the block sizes for the prefill case.
    m_block_sizes: the block sizes for the mixed case.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    debug_mode: if true, RPA does not issue any DMAs or run flash attention but
      print debug info. Need to compile with `--xla_tpu_enable_log_recorder`.
    disable_bounds_checks: if true, disable bounds checks.
    disable_semaphore_checks: if true, disable semaphore checks.

  Returns:
    The output of the attention.
  """
    q, k, v = queries, keys, values
    tpu_version = get_tpu_version()

    if out_dtype is None:
        out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        # We do not set to -inf directly because (-inf) - (-inf) is nan.
        mask_value = jnp.finfo(out_dtype).min

    if vmem_limit_bytes is None:
        # TODO(jevinjiang, jacobplatin): change this to use
        # `get_vmem_estimate_bytes` when VREG spilling is fixed.
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]

    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q, kv = prepare_inputs(q, k, v)
    (
        max_num_tokens,
        _,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache.shape[1]
    num_kv_heads_x2_per_kv_packing = kv_cache.shape[2]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    init_sem_ids = jnp.zeros((3, ), jnp.int32)
    # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    init_bo_ids = jnp.full((4, ), -1, jnp.int32)
    # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    init_bkv_update_ids = jnp.full((6, ), -1, jnp.int32)

    def run_rpa_kernel(
        q,
        kv_cache,
        *,
        bq_sz,
        bkv_sz,
        bq_csz,
        bkv_csz,
        static_q_len=None,
        case: RpaCase = RpaCase.MIXED,
    ):
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        bkv_stride = num_kv_heads_x2_per_kv_packing
        if has_bank_conflicts(bkv_stride):
            bkv_stride += 1

        bkv_double_buf = pltpu.VMEM(
            (2, bkv_sz, bkv_stride, *kv_cache.shape[3:]),
            kv_cache.dtype,
        )

        bq_double_buf = pltpu.VMEM(
            (2, bq_sz, actual_num_kv_heads, *q.shape[2:]),
            q.dtype,
        )

        bo_double_buf = bq_double_buf

        l_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
            out_dtype,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
            out_dtype,
        )

        scratch_shapes = [
            bkv_double_buf,  # (bkv_x2_ref) Double buffering for kv block.
            bq_double_buf,  # (bq_x2_ref) Double buffering for q block.
            bo_double_buf,  # (bo_x2_ref) Double buffering for output block.
            # Semaphores for double buffering of bkv, bq, bo and bkv_update.
            pltpu.SemaphoreType.DMA((4, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            # TODO(jevinjiang): can we use ragged page_indices to save some smem?
            page_indices,
            cu_q_lens,
            distribution,
            init_sem_ids,
            init_bo_ids,
            init_bkv_update_ids,
        )

        scope_name = f"RPA{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"
        kernel = pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                use_causal_mask=use_causal_mask,
                skip_kv_mask=skip_kv_mask,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bq_csz=bq_csz,
                bkv_csz=bkv_csz,
                case=case,
                debug_mode=debug_mode,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1, ),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                # TODO(jevinjiang): since each sequence depends on the previous
                # one, we need some extra work to support Megacore mode.
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                # Paged attention invokes multiple small DMAs for each pages
                # instead of a single large DMA. Therefore, the overhead of bounds
                # checking becomes too significant so we disable it.
                disable_bounds_checks=disable_bounds_checks,
                # Only set to true if you gurantee there is no race condition.
                disable_semaphore_checks=disable_semaphore_checks,
            ),
            out_shape=[
                pltpu.HBM(shape=q.shape, dtype=q.dtype),
                pltpu.HBM(shape=kv_cache.shape, dtype=kv_cache.dtype),
            ] if tpu_version >= 7 else [
                jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                jax.ShapeDtypeStruct(shape=kv_cache.shape,
                                     dtype=kv_cache.dtype),
            ],
            input_output_aliases={
                7: 0,
                9: 1
            },
            name=scope_name,
        )

        if tpu_version >= 7:
            # jit to color the memory since the q, kv are just preprocessed.
            @jax.jit
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(
                    *scalar_prefetches,
                    pltpu.with_memory_space_constraint(q, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
                )
        else:
            # TODO(b/494285697): v6 has issues with pinning aliased memory.
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(*scalar_prefetches, q, kv, kv_cache)

        return run(scalar_prefetches, q, kv, kv_cache)

    def _prepare_block_sizes(block_sizes, case):
        if block_sizes is None:
            return get_default_block_sizes(
                q.dtype,
                kv_cache.dtype,
                actual_num_q_heads,
                actual_num_kv_heads,
                head_dim,
                page_size,
                max_num_tokens,
                max_num_seqs,
                pages_per_seq,
                case=case,
            )
        return {
            "bq_sz": block_sizes[0],
            "bkv_sz": block_sizes[1],
            "bq_csz": block_sizes[2],
            "bkv_csz": block_sizes[3],
        }

    # Decode-only
    d_start, d_end = RpaCase.DECODE.get_range(distribution)
    q, kv_cache = lax.cond(
            d_start != d_end,
            lambda _: run_rpa_kernel(
                q,
                kv_cache,
                **_prepare_block_sizes(d_block_sizes, RpaCase.DECODE),
                static_q_len=1,
                case=RpaCase.DECODE),
            lambda _: (q, kv_cache),
            operand=None)

    if chunk_prefill_size is not None:
        # Prefill-only
        p_start, p_end = RpaCase.PREFILL.get_range(distribution)
        q, kv_cache = lax.cond(
                p_start != p_end,
                lambda _: run_rpa_kernel(
                    q,
                    kv_cache,
                    **_prepare_block_sizes(p_block_sizes, RpaCase.PREFILL),
                    static_q_len=chunk_prefill_size,
                    case=RpaCase.PREFILL),
                lambda _: (q, kv_cache),
                operand=None)
    
    # Mixed
    m_start, m_end = RpaCase.MIXED.get_range(distribution)
    q, kv_cache = lax.cond(
            m_start != m_end,
            lambda _: run_rpa_kernel(
                q,
                kv_cache,
                **_prepare_block_sizes(m_block_sizes, RpaCase.MIXED),
                static_q_len=None,
                case=RpaCase.MIXED),
            lambda _: (q, kv_cache),
            operand=None)

    return (
        prepare_outputs(q, actual_num_q_heads_per_kv_head, actual_head_dim),
        kv_cache,
    )
