"""Utility functions for ragged paged attention."""
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jnp.dtype(dtype).itemsize * 8


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

  Args:
    x: The input number (should be an integer).

  Returns:
    The smallest integer power of 2 that is >= x.
  """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if 'TPU' not in kind:
        return -1
    if kind.endswith(' lite'):
        kind = kind[:-len(' lite')]
    if kind.endswith('p') or kind.endswith('e'):
        kind = kind[:-1]
    if kind == 'TPU7x':
        return 7
    assert kind[:-1] == 'TPU v', kind
    return int(kind[-1])


def merge_sequences_into_tiles(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    bq_sz: int,
    bkv_sz: int,
    chunk_prefill_size: int | None = None,
    max_seqs_per_tile: int = 8,
):
    """Merge sequences into tiles for ragged paged attention.

    Returns:
      starts_seq: int32[max_num_seqs]
      ends_seq: int32[max_num_seqs]
      cu_q_lens_per_tile: int32[max_num_seqs, max_seqs_per_tile + 1]
      cu_kv_lens_per_tile: int32[max_num_seqs, max_seqs_per_tile + 1]
      tile_distribution: int32[3]
    """
    max_num_seqs = kv_lens.shape[0]
    num_seqs = distribution[2]
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]

    seq_indices = jnp.arange(max_num_seqs, dtype=jnp.int32)
    decode_end = distribution[0]
    prefill_end = distribution[1]
    seq_mode = jnp.where(seq_indices < decode_end, 0,
                         jnp.where(seq_indices < prefill_end, 1, 2))
    effective_q_lens = q_lens
    if chunk_prefill_size is not None:
        prefill_mask = jnp.logical_and(seq_indices >= decode_end,
                                       seq_indices < prefill_end)
        effective_q_lens = jnp.where(
            prefill_mask,
            jnp.minimum(q_lens, chunk_prefill_size),
            q_lens,
        )

    starts_seq = jnp.full((max_num_seqs, ), -1, dtype=jnp.int32)
    ends_seq = jnp.full((max_num_seqs, ), -1, dtype=jnp.int32)
    cu_q_lens_per_tile = jnp.zeros((max_num_seqs, max_seqs_per_tile + 1),
                                   dtype=jnp.int32)
    cu_kv_lens_per_tile = jnp.zeros((max_num_seqs, max_seqs_per_tile + 1),
                                    dtype=jnp.int32)

    def init_state():
        return (
            jnp.int32(0),  # tile_idx
            jnp.int32(0),  # tile_seq_count
            jnp.int32(0),  # used_q_size
            jnp.int32(0),  # used_kv_size
            jnp.int32(0),  # current_mode
            starts_seq,
            ends_seq,
            cu_q_lens_per_tile,
            cu_kv_lens_per_tile,
        )

    def add_sequence(state, seq_idx):
        (
            tile_idx,
            tile_seq_count,
            used_q_size,
            used_kv_size,
            current_mode,
            starts_seq,
            ends_seq,
            cu_q_lens_per_tile,
            cu_kv_lens_per_tile,
        ) = state
        q_len = effective_q_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        mode = seq_mode[seq_idx]

        can_merge = jnp.logical_and(
            tile_seq_count > 0,
            jnp.logical_and(
                tile_seq_count < max_seqs_per_tile,
                jnp.logical_and(
                    mode == current_mode,
                    jnp.logical_and(
                        (used_q_size % bq_sz + q_len) < bq_sz,
                        (used_kv_size % bkv_sz + kv_len) < bkv_sz,
                    ),
                ),
            ),
        )
        start_new_tile = jnp.logical_not(can_merge)

        def _close_tile(ends_seq):
            return ends_seq.at[tile_idx].set(seq_idx)

        ends_seq = lax.cond(
            jnp.logical_and(start_new_tile, tile_seq_count > 0),
            _close_tile,
            lambda x: x,
            ends_seq,
        )

        new_tile_idx = tile_idx + jnp.where(
            jnp.logical_and(start_new_tile, tile_seq_count > 0),
            1,
            0,
        )
        tile_idx = new_tile_idx
        tile_seq_count = jnp.where(start_new_tile, 0, tile_seq_count)
        used_q_size = jnp.where(start_new_tile, 0, used_q_size)
        used_kv_size = jnp.where(start_new_tile, 0, used_kv_size)
        current_mode = jnp.where(start_new_tile, mode, current_mode)

        def _start_tile(starts_seq):
            return starts_seq.at[tile_idx].set(seq_idx)

        starts_seq = lax.cond(start_new_tile, _start_tile, lambda x: x,
                              starts_seq)

        tile_seq_count = tile_seq_count + 1
        used_q_size = used_q_size + q_len
        used_kv_size = used_kv_size + kv_len

        cu_q_lens_per_tile = cu_q_lens_per_tile.at[
            tile_idx, tile_seq_count].set(used_q_size)
        cu_kv_lens_per_tile = cu_kv_lens_per_tile.at[
            tile_idx, tile_seq_count].set(used_kv_size)

        return (
            tile_idx,
            tile_seq_count,
            used_q_size,
            used_kv_size,
            current_mode,
            starts_seq,
            ends_seq,
            cu_q_lens_per_tile,
            cu_kv_lens_per_tile,
        )

    def loop_body(seq_idx, state):
        return lax.cond(seq_idx < num_seqs, add_sequence, lambda s, _: s, state,
                        seq_idx)

    state = init_state()
    state = lax.fori_loop(0, max_num_seqs, loop_body, state, unroll=False)
    (
        tile_idx,
        tile_seq_count,
        _,
        _,
        _,
        starts_seq,
        ends_seq,
        cu_q_lens_per_tile,
        cu_kv_lens_per_tile,
    ) = state
    num_tiles = tile_idx + jnp.where(tile_seq_count > 0, 1, 0)

    ends_seq = lax.cond(
        tile_seq_count > 0,
        lambda e: e.at[tile_idx].set(num_seqs),
        lambda e: e,
        ends_seq,
    )

    tile_indices = jnp.arange(max_num_seqs, dtype=jnp.int32)
    valid_tile = tile_indices < num_tiles
    decode_tile_end = jnp.sum(
        jnp.logical_and(valid_tile, ends_seq <= decode_end))
    prefill_tile_end = jnp.sum(
        jnp.logical_and(valid_tile, ends_seq <= prefill_end))
    tile_distribution = jnp.array(
        [decode_tile_end, prefill_tile_end, num_tiles], dtype=jnp.int32)

    return (
        starts_seq,
        ends_seq,
        cu_q_lens_per_tile,
        cu_kv_lens_per_tile,
        tile_distribution,
    )
