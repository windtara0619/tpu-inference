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
"""Reproduces the SparseCore stalls inside ragged_gather_v2's *inner* pipeline.

Profiling a real serving config (TP=8, expert-parallel + `attn_dp_size=8`,
`--batch-size 8 --input-len 4096 --output-len 2`, see
`examples/tpu_profiling.py`; trace at
`/home/tarading/profile/moe_4096_2_trace`) showed `ragged_gather_v2` stalling
inside `fused_moe_gmm`.

This test does NOT reproduce that via a synthetic out_size-vs-window timing
proxy. It captures a *real* `jax.profiler.trace` of `ragged_gather_v2` and
inspects the actual per-subcore pipeline-stage events the nested
`pltpu.emit_pipeline` call at ragged_gather_v2.py:203 (`inner_pipeline`,
grid=`(num_row_subchunks, num_cols)`) emits on the SparseCore "TEC N" trace
lines: `ep_initialize_0` -> `ep_copy_in` -> `ep_wait_in` -> `ep_run_kernel` ->
`ep_copy_out` (repeating), plus a trailing `ep_wait_out`/`ep_finalize`.
`ep_wait_in` is the literal DMA-wait bubble: the subcore blocked waiting for
the next indirectly-gathered row-subchunk's input DMA before it can run the
inner pipeline's `col_loop`, which is exactly the bubble described in the
TODO above `outer_pipeline` in ragged_gather_v2.py ("creates a pipeline
bubble (DMA wait) when the inner pipeline empties and restarts with new
indices").

The reproduction target below is calibrated directly against the real
trace, not guessed: aggregating self-time (stack-reconstructed so nested
`sc_ragged_gather_v2.*`/`ep_run_kernel` spans are never double counted, and
overlap-safe so concurrently in-flight calls on the same subcore track don't
inflate the denominator) across all 2,208 `sc_ragged_gather_v2.*` subcore
instances in `/home/tarading/profile/moe_4096_2_trace` gives:
  - `ep_wait_in` / total on-device time            ~= 12.6%
  - (total - `ep_run_kernel`) / total (all overhead) ~= 26.4%
(`sc_ragged_gather_v2.80` itself is `bf16[262144,2048]` gathered from
`bf16[32768,2048]`, i.e. `hidden_size=2048`, `in_size=32768`,
`out_size=262144` -- the shapes used below.)

This test reproduces that same stall mechanism live: it runs
`ragged_gather_v2` at those real shapes under `jax.profiler.trace`, parses
the resulting XSpace with the same self-time-correct, overlap-safe
reconstruction used to derive the numbers above, and asserts that a
sizeable, stable fraction of SparseCore time is `ep_wait_in` / non-compute
overhead rather than `ep_run_kernel` -- reproducing the inner-pipeline
stall directly from a real trace instead of inferring it indirectly.
"""

import collections
import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

import tpu_inference.envs as envs
from tpu_inference.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

jax.config.parse_flags_with_absl()

# Exact shapes of `sc_ragged_gather_v2.80` in the real trace at
# /home/tarading/profile/moe_4096_2_trace (long_name:
# `bf16[262144,2048]{...} custom-call(..., bf16[32768,2048]{...} %args_3_,
# bf16[262144,2048]{...} %args_4_)`), i.e. the permute step's SparseCore
# gather inside fused_moe_gmm's `_process_tokens_locally`.
_REAL_IN_SIZE = 32768
_REAL_HIDDEN_SIZE = 2048
_REAL_OUT_SIZE = 262144


def _load_xplane_pb2():
    """Locate XSpace proto bindings across available profiler packages.

    Mirrors `tests/kernels/mla_tuned_vs_baseline_test.py`'s helper of the
    same name.
    """
    for modname in (
            "tensorflow.tsl.profiler.protobuf.xplane_pb2",
            "tensorflow.core.profiler.protobuf.xplane_pb2",
            "xprof.protobuf.xplane_pb2",
            "tensorboard_plugin_profile.protobuf.xplane_pb2",
            "tensorboard.plugins.profile.protobuf.xplane_pb2",
    ):
        try:
            return __import__(modname, fromlist=["XSpace"])
        except ImportError:
            continue
    return None


def _self_time_under_gather(events):
    """Stack-based, overlap-safe self-time reconstruction over one track's
    full `(offset_ps, end_ps, name)` event list (sorted by offset).

    Chrome-trace / XSpace events on one track only carry `(offset, end,
    name)` -- nesting (parent/child) is implicit via interval containment,
    the same way Perfetto/xprof reconstruct it. Naively summing durations
    per name double counts (an outer `ep_run_kernel` can wrap a nested
    `ep_run_kernel`, and on the real trace a *different*
    `sc_ragged_gather_v2.NN` instance can even be nested inside another's
    span via SparseCore's async/pipelined dispatch); self-time -- duration
    minus the total duration of a node's direct children -- attributes
    every picosecond to exactly one node, so summing self-time per leaf
    name is exact regardless of how calls interleave.

    Returns `(self_time_by_leaf_name, union_ps)`: self-time is only counted
    for events whose nearest enclosing ancestor (or the event itself) is a
    `sc_ragged_gather_v2.*` span, and `union_ps` is the (overlap-safe) total
    time covered by the union of all top-level `sc_ragged_gather_v2.*`
    spans on this track -- the correct denominator, since two such spans
    can overlap without either containing the other.
    """
    self_time = collections.Counter()
    stack = []  # dicts: end, name, child_dur, under_gather
    gather_intervals = []

    def close(node):
        self_t = node["end"] - node["start"] - node["child_dur"]
        if node["under_gather"]:
            self_time[node["name"]] += self_t
        if stack:
            stack[-1]["child_dur"] += node["end"] - node["start"]

    for s, en, nm in events:
        while stack and stack[-1]["end"] <= s:
            close(stack.pop())
        is_gather = nm.startswith("sc_ragged_gather_v2.")
        if is_gather:
            gather_intervals.append((s, en))
        parent_under = stack[-1]["under_gather"] if stack else False
        stack.append({
            "start": s,
            "end": en,
            "name": nm,
            "child_dur": 0,
            "under_gather": parent_under or is_gather,
        })
    while stack:
        close(stack.pop())

    gather_intervals.sort()
    union_ps = 0
    cur_s = cur_e = None
    for s, en in gather_intervals:
        if cur_s is None:
            cur_s, cur_e = s, en
        elif s <= cur_e:
            cur_e = max(cur_e, en)
        else:
            union_ps += cur_e - cur_s
            cur_s, cur_e = s, en
    if cur_s is not None:
        union_ps += cur_e - cur_s

    return self_time, union_ps


def _capture_inner_pipeline_self_time(run_fn, args, n_calls=3):
    """Runs `run_fn(*args)` `n_calls` times under `jax.profiler.trace`,
    finds every `sc_ragged_gather_v2*` custom-call span on every SparseCore
    "TEC N" track, and returns `(self_time_by_event_name, total_union_ps)`
    aggregated (overlap-safe) across all of them -- the same methodology
    used to derive the ~12.6% / ~26.4% figures in the module docstring from
    the real trace at /home/tarading/profile/moe_4096_2_trace.
    """
    xplane_pb2 = _load_xplane_pb2()
    if xplane_pb2 is None:
        return None, None

    jax.block_until_ready(run_fn(*args))  # warmup / compile

    tmp_dir = tempfile.mkdtemp(prefix="ragged_gather_v2_stall_test_")
    try:
        with jax.profiler.trace(tmp_dir, create_perfetto_link=False):
            for _ in range(n_calls):
                out = run_fn(*args)
            jax.block_until_ready(out)

        pb_files = []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith(".xplane.pb"):
                    pb_files.append(os.path.join(root, f))
        if not pb_files:
            return None, None

        xspace = xplane_pb2.XSpace()
        with open(pb_files[0], "rb") as f:
            xspace.ParseFromString(f.read())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_self_time = collections.Counter()
    total_union_ps = 0
    for plane in xspace.planes:
        if "SparseCore" not in plane.name:
            continue
        event_meta = {m.id: m for m in plane.event_metadata.values()}
        for line in plane.lines:
            if not line.name.startswith("TEC"):
                continue
            events = sorted(
                (e.offset_ps, e.offset_ps + e.duration_ps,
                 event_meta[e.metadata_id].name) for e in line.events)
            self_time, union_ps = _self_time_under_gather(events)
            for name, dur in self_time.items():
                total_self_time[name] += dur
            total_union_ps += union_ps
    return total_self_time, total_union_ps


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedGatherV2InnerPipelineStallTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if pltpu.get_tpu_info().sparse_core is None:
            self.skipTest("Requires SparseCore hardware.")
        xplane_pb2 = _load_xplane_pb2()
        if xplane_pb2 is None:
            self.skipTest(
                "Requires XSpace proto bindings (tensorflow, xprof, or "
                "tensorboard-plugin-profile) to inspect the real trace's "
                "per-subcore pipeline-stage events.")

    @parameterized.named_parameters(
        dict(testcase_name="tiny_window", window=8),
        dict(testcase_name="small_window", window=512),
        dict(testcase_name="full_window", window=_REAL_OUT_SIZE),
    )
    def test_inner_pipeline_dma_wait_stalls_at_real_shapes(self, window):
        """At the exact shapes of the real `sc_ragged_gather_v2.80` call
        (out_size=262144, hidden_size=2048, in_size=32768, bf16), a real
        captured trace shows `ep_wait_in` -- the inner nested
        `pltpu.emit_pipeline`'s (ragged_gather_v2.py:203) DMA-wait bubble --
        and the broader non-`ep_run_kernel` overhead consuming a stable
        share of SparseCore time, matching (within margin) the ~12.6%
        `ep_wait_in` / ~26.4% total-overhead figures measured by
        aggregating over all 2,208 real `sc_ragged_gather_v2.*` instances in
        /home/tarading/profile/moe_4096_2_trace (see module docstring).
        """
        key = jax.random.key(0)
        k_hidden, k_indices = jax.random.split(key)
        x = jax.random.normal(k_hidden, (_REAL_IN_SIZE, _REAL_HIDDEN_SIZE),
                              jnp.float32).astype(jnp.bfloat16)
        indices = jax.random.randint(k_indices, (_REAL_OUT_SIZE, ), 0,
                                     _REAL_IN_SIZE, jnp.int32)
        start = jnp.array(0, jnp.int32)
        end = jnp.array(window, jnp.int32)

        @jax.jit
        def run(x, indices, start, end):
            return ragged_gather_v2(x, indices, start, end)

        self_time, total_union_ps = _capture_inner_pipeline_self_time(
            run, (x, indices, start, end))
        if total_union_ps is None or total_union_ps == 0:
            self.skipTest("No sc_ragged_gather_v2 SparseCore trace events "
                          "captured -- cannot measure the inner-pipeline "
                          "stall from this profile.")

        wait_in_frac = self_time.get("ep_wait_in", 0) / total_union_ps
        run_kernel_frac = self_time.get("ep_run_kernel", 0) / total_union_ps
        overhead_frac = 1 - run_kernel_frac
        print(f"\n[window={window}] total_union={total_union_ps/1e6:.2f}us "
              f"ep_run_kernel={run_kernel_frac:.2%} "
              f"ep_wait_in={wait_in_frac:.2%} "
              f"total_overhead={overhead_frac:.2%}")

        # Reproduces the regression: a real, sizeable share of on-device
        # time inside ragged_gather_v2 is the inner pipeline's DMA-wait
        # bubble (or other non-compute pipeline-stage overhead), not actual
        # gather compute. Thresholds are set with margin below both the
        # real-trace aggregate (12.6% / 26.4%) and this same measurement
        # repeated on local SparseCore hardware (16-18% / 26-34%). If these
        # regress toward 0, the bubble described in the TODO above
        # `outer_pipeline` (ragged_gather_v2.py) may have been fixed and
        # the thresholds can be tightened or the test retired.
        self.assertGreater(
            wait_in_frac, 0.08,
            "Expected ep_wait_in (the inner nested emit_pipeline's DMA-wait "
            "bubble at ragged_gather_v2.py:203) to consume a real share of "
            "SparseCore time for ragged_gather_v2 at real production "
            "shapes.")
        self.assertGreater(
            overhead_frac, 0.15,
            "Expected non-ep_run_kernel overhead (ep_wait_in + ep_wait_out "
            "+ ep_finalize + ep_initialize_0 + ep_copy_in/out) to consume a "
            "real share of SparseCore time for ragged_gather_v2 at real "
            "production shapes.")

    def test_stall_is_a_fixed_rate_not_amortized_by_more_real_work(self):
        """The inner-pipeline bubble is paid per grid step of the nested
        `pltpu.emit_pipeline` (ragged_gather_v2.py:203), which restarts
        every outer block -- so, unlike a one-off fixed dispatch cost, this
        stall's *fraction* of total time does not shrink as the window
        (real work) grows. That distinguishes it from ordinary launch
        overhead and is what makes it show up throughout a full profiling
        trace rather than just at small-batch call sites.
        """
        key = jax.random.key(1)
        k_hidden, k_indices = jax.random.split(key)
        x = jax.random.normal(k_hidden, (_REAL_IN_SIZE, _REAL_HIDDEN_SIZE),
                              jnp.float32).astype(jnp.bfloat16)
        indices = jax.random.randint(k_indices, (_REAL_OUT_SIZE, ), 0,
                                     _REAL_IN_SIZE, jnp.int32)

        @jax.jit
        def run(x, indices, start, end):
            return ragged_gather_v2(x, indices, start, end)

        fracs = {}
        for window in (32, _REAL_OUT_SIZE):
            start = jnp.array(0, jnp.int32)
            end = jnp.array(window, jnp.int32)
            self_time, total_union_ps = _capture_inner_pipeline_self_time(
                run, (x, indices, start, end))
            if total_union_ps is None or total_union_ps == 0:
                self.skipTest(
                    "No sc_ragged_gather_v2 SparseCore trace events "
                    "captured.")
            fracs[window] = self_time.get("ep_wait_in", 0) / total_union_ps

        print(f"\nep_wait_in fraction at window=32: {fracs[32]:.2%}, "
              f"at window={_REAL_OUT_SIZE} (full out_size): "
              f"{fracs[_REAL_OUT_SIZE]:.2%}")

        # A 8192x larger window (32 -> 262144 real rows) should not buy a
        # proportional reduction in the *fraction* of time spent stalled --
        # both should remain well above a token amount (with margin below
        # the real trace's ~12.6% aggregate and this repo's own ~16-18%
        # measurement on local SparseCore hardware).
        self.assertGreater(fracs[32], 0.08)
        self.assertGreater(fracs[_REAL_OUT_SIZE], 0.08)

    def test_num_row_subchunks_cap_crossover_sweep(self):
        """A/B sweep of `RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS` (4, the
        pre-investigation baseline, vs 8) across window sizes, to find
        where raising the cap is a net win vs. a net loss.

        Raising the cap doubles `block_size`, which means fewer restarts of
        the inner nested `pltpu.emit_pipeline` (ragged_gather_v2.py:203)
        for a *fixed* window -- so it helps once `window` is large enough
        that `num_blocks = cdiv(window, block_size)` actually drops. But
        the inner pipeline's grid is `(num_row_subchunks, num_cols)`, a
        *static* shape independent of how much of `[start, end)` is real
        data -- so for a small window (where `num_blocks` is already 1
        under both caps), a bigger cap only adds more inner-pipeline steps
        over padding, and net *hurts* absolute latency even though the
        `ep_wait_in` *fraction* looks better (the same fixed ramp-up cost
        is diluted over more, mostly-wasted, `ep_run_kernel` time).

        This directly measures per-call device time (not just the stall
        fraction) for both configs across a log-spaced window sweep at the
        real `sc_ragged_gather_v2.80` shapes, to locate the crossover
        instead of eyeballing a couple of window sizes.
        """
        orig_cap = os.environ.get("RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS")
        windows = [8, 32, 128, 512, 1024, 2048, 4096, 8192, 32768, 131072,
                  _REAL_OUT_SIZE]
        caps = [4, 8]

        key = jax.random.key(3)
        k_hidden, k_indices = jax.random.split(key)
        x = jax.random.normal(k_hidden, (_REAL_IN_SIZE, _REAL_HIDDEN_SIZE),
                              jnp.float32).astype(jnp.bfloat16)
        indices = jax.random.randint(k_indices, (_REAL_OUT_SIZE, ), 0,
                                     _REAL_IN_SIZE, jnp.int32)

        # per_call_us[cap][window] = median on-device time (us) for one
        # ragged_gather_v2 call at that (cap, window).
        per_call_us = {cap: {} for cap in caps}
        wait_frac = {cap: {} for cap in caps}
        try:
            for cap in caps:
                os.environ["RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS"] = str(
                    cap)
                # envs.* is read lazily from os.environ (unless
                # enable_envs_cache() was called), but the *compiled*
                # program for ragged_gather_v2 (itself @jax.jit) is cached
                # by JAX independent of env vars -- clear it so this cap is
                # actually retraced instead of silently reusing cap=4's
                # compiled program at the same shapes.
                jax.clear_caches()
                self.assertEqual(
                    envs.RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS, cap)

                @jax.jit
                def run(x, indices, start, end):
                    return ragged_gather_v2(x, indices, start, end)

                for window in windows:
                    start = jnp.array(0, jnp.int32)
                    end = jnp.array(window, jnp.int32)
                    n_calls = 3
                    self_time, total_union_ps = (
                        _capture_inner_pipeline_self_time(
                            run, (x, indices, start, end),
                            n_calls=n_calls))
                    if total_union_ps is None or total_union_ps == 0:
                        self.skipTest(
                            "No sc_ragged_gather_v2 SparseCore trace "
                            "events captured.")
                    per_call_us[cap][window] = (
                        total_union_ps / n_calls / 1e6)
                    wait_frac[cap][window] = (
                        self_time.get("ep_wait_in", 0) / total_union_ps)
        finally:
            if orig_cap is None:
                os.environ.pop("RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS",
                               None)
            else:
                os.environ["RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS"] = (
                    orig_cap)
            jax.clear_caches()

        print(f"\n{'window':>8s} {'cap=4 us':>10s} {'cap=8 us':>10s} "
              f"{'speedup':>9s} {'cap=4 wait%':>12s} {'cap=8 wait%':>12s} "
              f"{'winner':>8s}")
        crossover_window = None
        for window in windows:
            t4 = per_call_us[4][window]
            t8 = per_call_us[8][window]
            speedup = t4 / t8  # >1 means cap=8 is faster
            winner = "cap=8" if speedup > 1 else "cap=4"
            if speedup > 1 and crossover_window is None:
                crossover_window = window
            print(f"{window:8d} {t4:10.4f} {t8:10.4f} {speedup:9.3f} "
                  f"{wait_frac[4][window]:12.2%} {wait_frac[8][window]:12.2%} "
                  f"{winner:>8s}")

        if crossover_window is not None:
            print(f"\ncrossover: cap=8 starts winning on absolute time at "
                  f"window>={crossover_window} "
                  f"(out of {_REAL_OUT_SIZE} = out_size, "
                  f"window_fraction>={crossover_window/_REAL_OUT_SIZE:.4f})")
        else:
            print("\nno crossover found: cap=4 wins (or ties) at every "
                  "swept window")

        # Sanity/regression anchors from the manual A/B in conversation:
        # cap=8 should win decisively at the full window (matches the
        # ~11% absolute speedup measured earlier), and should *not* win
        # unconditionally -- there should exist at least one small window
        # where cap=4 (fewer wasted inner-pipeline steps over padding) is
        # faster in absolute time, confirming this is a real trade-off and
        # not a free win.
        self.assertLess(per_call_us[8][_REAL_OUT_SIZE],
                        per_call_us[4][_REAL_OUT_SIZE],
                        "Expected cap=8 to win on absolute device time at "
                        "the full window (fewer inner-pipeline restarts).")
        self.assertGreater(
            per_call_us[8][windows[0]], per_call_us[4][windows[0]],
            "Expected cap=4 to win on absolute device time at the "
            "smallest window (num_blocks==1 under both caps, so a bigger "
            "cap only adds wasted inner-pipeline steps over padding) -- "
            "if this no longer holds, the cap=8 trade-off may have "
            "changed and this test's crossover framing should be "
            "revisited.")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
