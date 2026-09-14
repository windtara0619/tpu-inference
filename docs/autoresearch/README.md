# AutoResearch implementation context

This directory converts the AutoResearch design documents into repository-native, agent-readable context. The linked Google Docs remain the human-facing sources of truth. The Markdown files record the exact source revisions used for this package.

## Required reading order

1. Read [`/AGENTS.md`](../../AGENTS.md) for execution and evidence rules.
2. Read [`high-level-design.md`](high-level-design.md) for system boundaries, ownership, data flow, and promotion policy.
3. Before implementing Phase 0, read [`scenario-director-serving-analysis.md`](scenario-director-serving-analysis.md) in full.
4. Read [`context-manifest.yaml`](context-manifest.yaml) for machine-readable sources, invariants, contracts, and exit criteria.
5. Inspect the current code and tests at the checked-out commit. Document paths in the designs are anchors, not permission to assume APIs are unchanged.

## Source priority

When evidence conflicts, use this order:

1. User instructions for the active task.
2. Safety, correctness, and promotion invariants in this package.
3. The detailed Phase 0 design for Scenario Director and Serving Analysis behavior.
4. The high-level design for system-wide boundaries.
5. Current repository code and tests for implementation mechanics.
6. Examples, prior experiments, and benchmark values, which are evidence rather than permanent requirements.

If a Google Doc revision differs from the revision in `context-manifest.yaml`, stop and resync the affected Markdown before relying on it for a design decision.

## System boundary

```text
ServingScenario
  -> Serving Analysis Agent
  -> StaticSliceSpec + BottleneckCandidate + evidence
  -> Kernel Engineering Subagent
  -> AnalyticalKernelModel + KernelDesignSpec
  -> chip-specific Kernel Coding Subagent
  -> candidate implementation + implementation evidence
  -> independent Verification Gate
  -> Serving Plan selector + baseline fallback
```

The Scenario Director freezes the optimization contract. The Serving Analysis Agent turns dynamic production behavior into immutable, replayable research inputs. Kernel Engineering owns mathematical and schedule design. Kernel Coding implements that design for a target chip and must escalate conflicts rather than silently changing the algorithm.

## Phase 0 scope

Phase 0 implements only:

- Scenario validation, normalization, fingerprinting, and lifecycle.
- Evidence capture and integrity checks.
- Graph-hit workload reconstruction and weighted context selection.
- Immutable `StaticSliceSpec` generation.
- Compiler/profiler evidence correlation.
- Bottleneck ranking and replay-bundle emission.
- Deterministic storage, provenance, reruns, observability, and tests.

Phase 0 does not redesign kernels, generate target code, update a serving selector, or promote a result.

## Non-negotiable invariants

- Correctness is established before performance claims.
- Production requests are never mutated by research code.
- Every artifact is versioned and traceable to scenario, repository, model, topology, and toolchain fingerprints.
- A `StaticSliceSpec` is immutable and replayable.
- Missing or truncated evidence yields an explicit degraded or failed result; it is never silently treated as complete.
- Research runs are branch/worktree isolated and reversible.
- The Research Orchestrator is the sole writer of shared run state.
- Promotion requires full replay, SLO compliance, no unintended recompilation, an observable selector, and a known-good baseline fallback.
- Never invent benchmark or test results. Mark unavailable evidence as pending and state the exact command or artifact needed.

## Terminology

`StaticSliceSpec` is an internal implementation record, not a general industry term. It names one fixed serving context with exact shapes, topology, memory/layout state, workload weight, and replay inputs.

Normative words have their usual requirements meaning:

- **MUST**: required for correctness or acceptance.
- **SHOULD**: expected unless a documented reason justifies deviation.
- **MAY**: optional.

## Canonical sources

- [AutoResearch Kernel Agent for New Chips — Design Doc](https://docs.google.com/document/d/1eDtoOujHeuqTQauXiozcyFd2gp1RyDCcIwws42XBJU0/edit)
- [Scenario Director and Serving Analysis Agent Detailed Design](https://docs.google.com/document/d/1ZwuZ5udWU3Xq_dM-Hbjbv8mRk0KGG7RooVXtRj1Dwn4/edit)
