# AGENTS.md

## AutoResearch work

For any task involving the AutoResearch Kernel Agent, Scenario Director, Serving Analysis Agent, `StaticSliceSpec`, kernel research, or serving-plan generation:

1. Read `docs/autoresearch/README.md`.
2. Follow its required reading order.
3. Revalidate design anchors against the current checkout before editing.
4. Keep changes isolated on a task branch/worktree and preserve unrelated work.

## Execution rules

- Establish correctness before measuring or claiming performance.
- Do not mutate production traffic, weaken SLOs, or bypass promotion gates.
- Do not silently change mathematical intent in chip-specific code. Escalate a `KernelDesignSpec` conflict to Kernel Engineering.
- Keep artifacts deterministic, versioned, replayable, and tied to scenario, repository, model, topology, and toolchain fingerprints.
- Treat missing, truncated, or corrupted evidence as an explicit degraded/failed state.
- Do not commit secrets, model credentials, large profiler captures, generated binaries, or raw benchmark outputs unless the active task explicitly defines their storage path.
- Never fabricate unit-test, benchmark, compiler, or profiler results. Report commands run, environment, raw artifact locations, and anything not run.
- Prefer scoped tests first. TPU-dependent tests that cannot run locally must be listed precisely for remote execution.
- A performance candidate cannot replace the baseline without independent verification, full weighted replay, SLO compliance, no unintended recompilation, an observable selector, and a baseline fallback.
