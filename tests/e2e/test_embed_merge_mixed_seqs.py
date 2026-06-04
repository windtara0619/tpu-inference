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

"""E2E correctness tests for llm.embed() with MERGE_MIXED_SEQS on and off.

vLLM already manages its own EngineCore subprocess internally.  These tests
run LLM directly in the pytest process (same pattern as test_step_pooling.py)
to avoid a second subprocess layer that would compete for the TPU.

Each test sets os.environ["MERGE_MIXED_SEQS"] before creating the LLM so
that the EngineCore subprocess inherits the correct env at spawn time.
The LLM is cleaned up with an explicit shutdown + del/gc between tests so
the TPU lock is released before the next test's EngineCore acquires it.

To test both modes in one run:
    python -m pytest tests/e2e/test_embed_merge_mixed_seqs.py -v

To test a single mode:
    MERGE_MIXED_SEQS=0 python -m pytest ... -k normal
    MERGE_MIXED_SEQS=1 python -m pytest ... -k merged
"""

import multiprocessing as mp

try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import gc
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

_LLM_KWARGS = dict(
    runner="pooling",
    max_num_seqs=16,
    max_model_len=512,
    max_num_batched_tokens=512,
    dtype="bfloat16",
    trust_remote_code=True,
    tensor_parallel_size=1,
    # Disable prefix caching: TPU v6e uses fp8 KV cache by default, so a
    # second embed() call would dequantize from fp8 and produce small
    # numerical differences compared to the first full-prefill call.
    # Disabled here so the determinism tests get bit-identical runs.
    enable_prefix_caching=False,
)

# Mix of lengths to exercise different grouping behaviours in the merged path.
_TEST_PROMPTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Embedding model test sentence.",
    "Short.",
    "A slightly longer sentence to vary the batch composition.",
    "Another test.",
    "One more embedding request.",
    "Final test prompt.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(merge_enabled: bool):
    """Set MERGE_MIXED_SEQS, create LLM.  Skip if the TPU is unavailable."""
    os.environ["MERGE_MIXED_SEQS"] = "1" if merge_enabled else "0"
    from vllm import LLM
    try:
        return LLM(model=MODEL_NAME, **_LLM_KWARGS)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("tpu", "backend", "libtpu", "jax")):
            pytest.skip(f"TPU not available: {e}")
        raise


def _shutdown(llm) -> None:
    """Explicitly terminate the EngineCore subprocess before deleting the LLM.

    vLLM's EngineCore holds the exclusive TPU lock.  Without an explicit
    shutdown the lock may not be released before the next test's LLM tries
    to acquire it.
    """
    try:
        # v1 engine path
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()


def _embed(llm, prompts=_TEST_PROMPTS) -> list[list[float]]:
    return [list(r.outputs.embedding)
            for r in llm.embed(prompts, use_tqdm=False)]


def _assert_valid(embeddings: list[list[float]], label: str) -> None:
    assert len(embeddings) == len(_TEST_PROMPTS), (
        f"{label}: expected {len(_TEST_PROMPTS)} embeddings, "
        f"got {len(embeddings)}")
    dim = len(embeddings[0])
    assert dim > 0, f"{label}: embedding dimension is 0"
    for i, emb in enumerate(embeddings):
        assert len(emb) == dim, (
            f"{label}[{i}]: dimension {len(emb)} != expected {dim}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_normal_path():
    """MERGE_MIXED_SEQS=0: llm.embed() returns valid, deterministic embeddings."""
    llm = _make_llm(merge_enabled=False)
    try:
        embs1 = _embed(llm)
        embs2 = _embed(llm)

        _assert_valid(embs1, "normal[run1]")
        _assert_valid(embs2, "normal[run2]")

        # Same model + same input → bit-identical output on TPU.
        assert np.array_equal(np.array(embs1), np.array(embs2), equal_nan=True), (
            "Normal path (MERGE_MIXED_SEQS=0) is not deterministic: "
            "run1 and run2 differ.")
    finally:
        _shutdown(llm)


def test_embed_merged_path():
    """MERGE_MIXED_SEQS=1: llm.embed() returns valid, deterministic embeddings."""
    llm = _make_llm(merge_enabled=True)
    try:
        embs1 = _embed(llm)
        embs2 = _embed(llm)

        _assert_valid(embs1, "merged[run1]")
        _assert_valid(embs2, "merged[run2]")

        assert np.array_equal(np.array(embs1), np.array(embs2), equal_nan=True), (
            "Merged path (MERGE_MIXED_SEQS=1) is not deterministic: "
            "run1 and run2 differ.")
    finally:
        _shutdown(llm)


def test_embed_merged_matches_normal():
    """Merged kernel produces numerically close embeddings to the normal kernel.

    Both LLMs load the same model weights, so the two kernel paths (RPAm vs
    RPAmerged) should produce the same embeddings up to bfloat16 rounding.
    """
    # --- normal path ---
    llm = _make_llm(merge_enabled=False)
    try:
        normal_embs = _embed(llm)
    finally:
        _shutdown(llm)

    # --- merged path (fresh LLM, TPU re-acquired) ---
    llm = _make_llm(merge_enabled=True)
    try:
        merged_embs = _embed(llm)
    finally:
        _shutdown(llm)

    _assert_valid(normal_embs, "normal")
    _assert_valid(merged_embs, "merged")

    n = np.array(normal_embs, dtype=np.float32)
    m = np.array(merged_embs, dtype=np.float32)
    assert n.shape == m.shape, (
        f"Shape mismatch: normal {n.shape} vs merged {m.shape}")

    # The two kernel paths (RPAm vs RPAmerged) produce identical results up to
    # bfloat16 accumulated rounding (~14 ULPs observed, max abs diff ~0.0023).
    # atol=5e-3 is comfortably above that while still catching regressions
    # like the q-limit bug (max diff ~0.27).
    assert np.allclose(n, m, rtol=0, atol=5e-3), (
        "Merged path embeddings differ from normal path beyond bfloat16 "
        f"tolerance (atol=5e-3).\n"
        f"Max abs diff : {np.abs(n - m).max():.6f}\n"
        f"Mean abs diff: {np.abs(n - m).mean():.6f}"
    )
