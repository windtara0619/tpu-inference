---
document_id: autoresearch-high-level-design
title: "AutoResearch Kernel Agent for New Chips — High-Level Design"
format: agent-readable-markdown
status: design
canonical_source:
  type: google_doc
  title: "AutoResearch Kernel Agent for New Chips — Design Doc"
  url: "https://docs.google.com/document/d/1eDtoOujHeuqTQauXiozcyFd2gp1RyDCcIwws42XBJU0/edit"
  revision: "ANLCKQmA0fJAXW7jSdn3kJaoLm1vBEtXuPh_hIpkLoCu4_-_P_eOf8_JdzxMMBtO9pMslQOzO8mHoNkyyuzllri30VjsjOpLqLJQq_XobqE"
repository_target:
  repository: windtara0619/tpu-inference
  baseline_commit: a4cfff0f10cee7525295236ce7cd13eba230633e
normative_scope: "system boundaries, ownership, contracts, verification, and delivery plan"
sync_policy: "When this file conflicts with the canonical source at the recorded revision, treat the Google Doc as authoritative. If its revision changed, review and resync before implementation."
---

> Agent context note: This is a text-first export for implementation agents. Embedded diagrams were replaced by their captions. Revalidate repository paths and APIs against the checked-out commit before changing code.
# AutoResearch Kernel Agent for New Chips

## Chip-portable workload-aware co-design across serving, compiler, kernel, and memory—starting with TPU

**Status:** Draft v0.1 for internal review
**Date:** Sep 7, 2026
**Repository:** vllm-project/tpu-inference
**Primary audience: accelerator inference engineers, compiler teams, kernel researchers, and technical leadership**
**CORE PROPOSAL  Build a chip-portable autonomous research system that starts from a measured production workload, identifies representative precompiled serving configurations, and jointly optimizes scheduling, graph compilation, operator selection, kernels, and memory. For each configuration, combined compiler and profiler analysis identifies the exposed bottleneck; one agent builds a mathematical performance model and kernel design, and a chip-specific coding agent implements and benchmarks candidate kernels. The system packages verified choices into a bounded set of precompiled variants with deterministic dispatch and safe fallbacks, and promotes them only when full serving A/B tests improve the target objective without violating correctness, latency SLOs, compilation-time, or memory budgets.**

# Executive summary

Traditional kernel optimization asks one implementation to cover many model architectures and dynamic traffic patterns. Production serving violates that assumption: a tensor-parallel MoE path may win at low token counts, while an expert-parallel path can win when higher counts amortize communication. AutoResearch reverses the abstraction. It first reduces measured traffic to fixed serving contexts whose execution conditions are fully specified, then expands optimization across service policy, compiled graph, operator choice, mathematics, kernel design, and memory. The architecture targets any new accelerator chip; TPU is the first implementation and the examples use tpu-inference. Porting to another chip changes only the machine-topology context and the chip-specific Kernel Coding Subagent.
**Inner loop — Research Orchestrator. Given an optimization goal—for example, maximize throughput for ISL/OSL 8192/1024 on Qwen3.5-397B-A17B-FP8 on TPU v7-8—the orchestrator runs a shared chip-neutral workflow; only the machine-topology/toolchain context and the chip-specific Kernel Coding Subagent change across chips:**

1. Run the Serving Analysis Agent on the running service, request/scheduler logs, compiler dumps, and host/device profiles. It builds a measured graph-hit heatmap over co-occurring precompiled token and request buckets, split by phase, and preserves dominant modes and SLO-relevant tails.
2. Have the Serving Analysis Agent freeze each selected workload region as a fully specified serving context: graph key, operator shapes, topology, memory/layout state, replay inputs, and the fraction of benchmark or production forwards it represents.
3. Have the Serving Analysis Agent combine compiler and profiler evidence to map each context to optimized graph operations, memory/liveness effects, and source; then emit the ranked exposed critical-path bottlenecks.
4. Dispatch the Kernel Engineering Subagent to build the mathematical model of each optimizing component and the full kernel, then decide the design contract: algorithm, graph and fusion boundaries, pipeline, tiling, buffering, placement, and memory layout.
5. Use the mathematical-model-versus-baseline gap to select the service, graph, operator, mathematical, and memory opportunities that should become kernel candidates.
6. Dispatch the chip-specific Kernel Coding Subagent to generate and implement every kernel variant specified by the Kernel Engineering Subagent, using the target chip’s language, intrinsics, compiler, and benchmark skills.
7. Run independent correctness, compilation, memory, mechanism, microbenchmark, and serving experiments; compare device results with the mathematical model, classify residuals, and feed the lessons to the outer loop.
8. Emit a versioned Serving Plan with evidence, selector rules, fallbacks, and an immediate re-profile so the next bottleneck starts another cycle.

**Outer loop — Research Policy Learner. It converts experiment history—including prediction error, residual diagnoses, crossover results, and dead ends—into better serving analysis, compiler and profiler interpretation, mathematical models, coding skills, priors, transformations, and experiment sequencing. It may improve how research is performed, but it cannot weaken correctness or promotion gates.**
> Diagram omitted from this text-first context package. Use the adjacent caption or the canonical Google Doc when visual detail is required.
*Figure 1\. The chip-portable inner loop freezes static slices, follows a model-to-code handoff, and emits a serving plan; the outer loop improves how the agent researches.*

# 1. Problem and design thesis

## 1.1 Fragmented local optimization

A production inference step is a coupled system: request arrival and scheduling determine padded token shape; shape determines the precompiled executable; executable structure determines fusion, collective, and materialization choices; those choices determine kernel shapes, memory pressure, and which chip engines can overlap. Optimizing each layer independently leaves cross-layer opportunities unclaimed. The coupling is architectural; TPU supplies the first concrete engine and topology vocabulary.

| Layer | Typical local assumption | Missed coupling |
| :---- | :---- | :---- |
| Serving / scheduler | Kernel performance is fixed | Batch composition changes tile efficiency, routing density, collective amortization, and the winning kernel variant. |
| Graph / compiler | Operator cost is summarized by a custom call | HBM round-trips, zero-fill, concatenation, donation, prefetch, and kernel-internal VMEM lifetimes may dominate. |
| Operator | One implementation or threshold is globally adequate | Dense vs. SparseCore, merged vs. per-sequence, and algorithmic paths cross over by shape and utilization. |
| Kernel | One benchmark shape represents production | Token buckets, request counts, KV length, routing skew, and phase produce different bottlenecks. |

## 1.2 Design thesis

**THESIS  The correct optimization target is not a generally tuned kernel; it is the measured distribution reduced to fully specified, static serving slices. The agent should co-design each slice so execution and memory are predictable, then compile only the variants whose workload-weighted value exceeds their compilation, memory, and maintenance cost.**
This is intentionally stronger than today’s standalone performance engineering and stronger than a conventional autotuner. Standalone efforts optimize one layer while treating neighboring decisions as fixed; an autotuner searches parameters inside a preselected implementation. AutoResearch freezes a static slice, combines compiler and profiler evidence to expose the true bottleneck, asks the Kernel Engineering Subagent for a mathematical design contract, asks the chip-specific Kernel Coding Subagent to implement it, and validates the combined serving plan end to end.

## 1.3 Why TPU is the first implementation

* **Discrete compiled contexts.** JAX/XLA programs are specialized to static shapes; tpu-inference already constructs token and request padding ladders and precompiles the cross-product needed by the backbone.
* **Deterministic hardware behavior.** Stable device execution makes paired microbenchmarks, calibrated engine timelines, and crossover detection unusually valuable.
* **Heterogeneous engines.** MXU, VPU, DMA, SparseCore, and ICI can overlap, so explicit scheduling and memory placement create large wins that a scalar FLOP count cannot predict.
* **Finite decision surface. TPU serving already uses a bounded ladder of token/request buckets, creating natural cells in which the agent can learn and validate specialized plans. The core contracts remain chip-neutral: a new chip supplies its topology context and Kernel Coding Subagent while reusing the same slicing, modeling, verification, plan, and learning loop.**

# 2. Goals, non-goals, and principles

## 2.1 Goals

* **Serving analysis input — The Scenario Director freezes model/quantization, machine-topology/toolchain, traffic/SLO, budget, and permission metadata. The Serving Analysis Agent ingests the running service, request/scheduler logs, compiler dumps, and host/device profiles.**
* **Serving analysis output: workload model — Build the measured phase × padded-token × padded-request hit heatmap over the exact precompiled bucket ladders, then freeze the selected hit cells as StaticSliceSpecs; retain dominant modes, crossover neighborhoods, and SLO tails.**
* **Serving analysis output: bottlenecks — Combine compiler dumps and profiler traces to map each selected context from optimized operations to source, identify costs outside custom kernels, and emit a ranked bottleneck list with exposed critical-path evidence.**
* **Sequential model-to-code research — The Research Orchestrator sends each bottleneck grounded in compiler and profiler evidence first to the Kernel Engineering Subagent, which builds the context-specific mathematical model and KernelDesignSpec. Only then does the chip-specific Kernel Coding Subagent generate and implement the specified candidates; the Verification Gate independently evaluates them.**
* **Serving Plan Compiler — Emit a bounded set of precompiled variants, selector rules, fallbacks, evidence, and a rollback path for the measured workload.**
* **Research Policy Learner — Learn from crossovers, prediction residuals, failures, and wins to improve serving analysis, compiler and profiler interpretation, mathematical models, chip-coding skills, priors, and experiment selection across runs.**

## 2.2 Non-goals

* Replacing XLA or Mosaic in the first version. The agent initially operates through source transformations, compiler flags, custom kernels, and measured executable selection.
* Optimizing synthetic kernels without a path to a production serving objective.
* Unbounded runtime adaptation. The first version favors a small, auditable variant set and explicit selectors.
* Autonomous production rollout. The system may create a branch or PR and a canary plan; human approval remains required for deployment.
* Weakening numerical correctness, model quality, or reproducibility gates in pursuit of performance.

## 2.3 Operating principles

| Principle | Implication |
| :---- | :---- |
| Measure before explaining | Profile the shipping shape and reconstruct the actual critical path before generating hypotheses. |
| Compiled code is the truth | Inspect optimized HLO/Mosaic and the trace, not only Python/JAX source. |
| Memory is part of the graph | Track materialization, bytes moved, lifetime, placement, donation, zero-init, and spill risk. |
| Distribution beats average | Prioritize expected contribution and tail impact across contexts, not one mean kernel latency. |
| E2E is the promotion gate | Microbench wins are evidence; serving wins are outcomes. |
| Dead ends are assets | Record hard constraints, failed mechanisms, and invalidated causal stories for future search. |
| Specific and static | Reduce the measured workload into fully specified traffic slices; optimize each fixed shape and context, then compose the bounded Serving Plan. |
| Mathematical model before implementation | The Kernel Engineering Subagent derives the dependency-constrained mathematical model and KernelDesignSpec first—including algorithm, graph and fusion boundaries, pipeline, tiling, buffering, placement, layout, and lifetimes. The Kernel Coding Subagent implements that contract and targets zero avoidable bubbles or spills. |

# 3. Serving Plan abstraction and objective

> Diagram omitted from this text-first context package. Use the adjacent caption or the canonical Google Doc when visual detail is required.
*Figure 2\. The optimization unit spans every layer that changes the cost of a serving context.*

## 3.1 Serving Plan

A Serving Plan is the deployable output of one research run. The Scenario Director supplies an immutable, chip-neutral ServingScenario organized as model/quantization, machine topology/toolchain, and traffic/SLO metadata. The Serving Analysis Agent supplies the measured precompiled-bucket heatmap, StaticSliceSpec set, and ranked bottleneck list; the Kernel Engineering Subagent supplies mathematical models and KernelDesignSpecs; the chip-specific Kernel Coding Subagent supplies candidate implementations; the Verification Gate supplies accepted evidence and model-gap lessons; and the Serving Plan Compiler assembles the minimum variant set needed to serve the workload.

| Plan field | Contents |
| :---- | :---- |
| Scenario | Model/checkpoint/tokenizer and architecture; dtype/quantization; chip family, machine topology, engine and memory hierarchy, interconnect/mesh, and compiler/runtime revisions. TPU generation/slice and TP/EP/DP/CP mesh are the first adapter. |
| Workload envelope | Measured phase × padded-token × padded-request graph-hit heatmap over the exact precompiled ladders, with normalized per-cell hit share; actual shapes, KV lengths, prefix reuse, operator live rows, validity/skew, dominant modes, and SLO tails. |
| Service policy | max\_num\_batched\_tokens, max\_num\_seqs, batching/prefill policy, phase and admission rules. |
| Executable variants | Graph fingerprint, static shape, sharding/layout, operator and kernel selections, compile artifact. |
| Selector | Static bucket mapping plus tightly bounded dynamic features where justified; always includes a safe fallback. |
| Evidence | Accepted ExperimentRecord IDs, AnalyticalKernelModel, KernelDesignSpec, and OptionMatrix; predicted-vs-measured residuals and lessons; correctness and provenance; compiler/trace diffs; microbenchmark and E2E A/B statistics; memory and compile cost. |

## 3.2 Research context schema

The Scenario Director and Serving Analysis Agent organize every run into three explicit categories so every mathematical model, implementation, experiment, and selector is reproducible across chips:
**MODEL \+ QUANTIZATION METADATA  Model/checkpoint/tokenizer; layer count; hidden/intermediate dimensions; attention heads; expert count and top-k; weight/activation/cache dtypes; quantization format, scale granularity, accumulation type, and quality constraints.**
**MACHINE TOPOLOGY \+ TOOLCHAIN  Chip family and slice; device/interconnect topology; parallel mesh; engine inventory; memory hierarchy, capacities, bandwidths, registers, lanes, and alignment limits; compiler/runtime and kernel language; sharding, layout, cache-residency, and synchronization rules. The TPU adapter instantiates these fields with TPU generation, ICI, MXU/VPU/DMA/SparseCore, HBM/VMEM, and Pallas/XLA details.**
**TRAFFIC \+ SLO  Arrival process and concurrency; ISL/OSL distribution; prefill/decode mix; prefix reuse and KV length; the exact precompiled token and request ladders; actual and padded token/request counts; operator live rows, valid fraction, and routing skew; the measured phase × padded-token × padded-request graph-hit heatmap and per-cell hit share; throughput/TTFT/TPOT/E2E targets; compile, memory, and experiment budgets. For example, a server may precompile num\_tokens \= 8, 16, 32, 64, 128, 256, 512, 1,024, and 2,048; the benchmark or production trace determines which of those buckets receive traffic.**
**EXECUTION KEY  Within one fixed scenario, c \= (phase, padded\_tokens, actual\_tokens, padded\_requests, active\_requests, KV\_length\_bucket, top\_k, operator\_live\_rows, valid\_fraction, routing\_skew, cache/layout state).**
Model/quantization and machine topology/toolchain are static scenario metadata; compile-time fields in c may specialize executables. Dynamic fields may select among precompiled programs on the host, feed a measured cheap branch, or remain observation-only features. The implementation represents each selected fixed context in a record named StaticSliceSpec. The record contains the executable key, exact operator shapes, topology, dtype/quantization, layout/cache state, workload weight p(c), replay inputs, and allowable selector features. Here p(c) is the normalized fraction of compiled forwards in the target benchmark or production trace that use the complete execution context c. This pivotal reduction turns dynamic, general kernel optimization into a finite set of specific problems whose execution is fully predictable within the declared context. The plan must retain dominant joint modes—for example, a (2,048-token, 16-request) executable receiving 40% of forwards—even when marginal histograms would hide them.

## 3.3 Optimization objective

The primary score is workload-weighted serving cost, constrained by the user's SLO and resource budgets. One practical formulation is:
**J(P) \= Σc p(c) · \[w₁·step(c,P) \+ w₂·TTFT(c,P) \+ w₃·TPOT(c,P)\] \+ λcompile·C(P) \+ λmem·M(P) \+ λrisk·R(P)**
For a throughput target, the metric term can be dollars or accelerator-seconds per token under a fixed SLO. For latency-sensitive serving, P99 TTFT and inter-token latency become hard constraints rather than small weights. Here p(c) is the measured frequency of the complete execution context c—not a product of independent token, request, phase, or KV-length marginals. A joint distribution is required because these values co-occur in one forward and interact: a 512-token graph with one request can have a different bottleneck and winning kernel than a 512-token graph with 64 requests. The Scenario Director fixes the weights and constraints; the Research Orchestrator prioritizes with them, the Verification Gate evaluates them, and the Serving Plan Compiler charges every retained variant.

* Rank contexts by contribution p(c) × exposed\_time(c), not frequency alone.
* Use exposed critical-path time rather than summing overlapped MXU/VPU/DMA/ICI events.
* Charge each extra executable for compilation time, cache footprint, startup/warmup cost, and operational complexity.

## 3.4 High-level OptionMatrix example

Consider a serving workload compiled into several token and request buckets. The agent does not choose one globally best kernel. For each high-impact execution context, it creates a structured decision record across four layers:

* Service and graph — batching and admission policy, precompiled executable, sharding, graph boundaries around custom calls, and TP/EP flow.
* Operator and engine — legal algorithms and TensorCore, SparseCore, VPU, DMA, or collective placement.
* Kernel and memory — tile, chunk, and buffer settings; fusion, pipelining, layout, liveness, and spill constraints.
* Evidence and fallback — analytical prediction, measured result, selector, validity envelope, and safe fallback.

The OptionMatrix is the research surface, not a hand-written dispatch table. For each StaticSliceSpec, the Kernel Engineering Subagent converts opportunity lenses into a mathematical design space and KernelDesignSpec; the Kernel Coding Subagent generates the corresponding legal kernel candidates; and the Verification Gate determines which candidates earn a place in the plan. Small-context, crossover, high-throughput, and SLO-tail modes may converge on different plans.

# 4. System architecture

> Diagram omitted from this text-first context package. Use the adjacent caption or the canonical Google Doc when visual detail is required.
*Figure 3\. Inner/outer execution-state view; Figure 1 shows agent ownership.*

## 4.1 Major components

| Component | Responsibility | Primary artifacts |
| :---- | :---- | :---- |
| Scenario Director | Freezes model/quantization metadata, the portable machine-topology/toolchain context, traffic/SLO, repository revision, budget, and permissions. | Scenario manifest |
| Serving Analysis Agent | Ingests a running serving workload with request and scheduler logs, compiler dumps, and host/device profiles; builds the measured precompiled-bucket heatmap, freezes selected contexts as StaticSliceSpecs, maps compiler and profiler evidence to source and graph boundaries, and ranks exposed critical-path bottlenecks. | Precompiled-bucket heatmap \+ StaticSliceSpec set \+ ranked bottleneck list |
| Research Orchestrator | Owns shared run state and the plan-level OptionMatrix; sends component tasks grounded in compiler and profiler evidence to the Kernel Engineering Subagent, passes approved KernelDesignSpecs to the chip-specific Kernel Coding Subagent, schedules independent verification, and re-ranks after every trial. | Task graph \+ OptionMatrix |
| Kernel Engineering \+ chip-specific Kernel Coding Subagents | Runs a strict sequential handoff. The Kernel Engineering Subagent first builds the mathematical model and decides algorithm, graph and fusion boundaries, pipeline, tiling, buffering, placement, and memory layout. Only after it emits KernelDesignSpec does the chip-specific Kernel Coding Subagent generate and implement the specified candidates and improve its coding skills through a nested KernelSageBench loop. Neither subagent can self-promote. | AnalyticalKernelModel \+ KernelDesignSpec \+ KernelCodingResult |
| Verification Gate | Independently gates correctness, confidence, SLOs, resources, and causality; compares measured device behavior with the mathematical prediction and classifies material residuals; neither subagent can self-approve. | ExperimentRecord \+ ModelGapLesson |
| Serving Plan Compiler | Pareto-prunes variants and emits selector, fallback, and rollout evidence. | Serving Plan |
| Research Policy Learner | Updates cost models, residual taxonomy, routing priors, transformations, and tools from typed history; cannot change hard gates. | Versioned research policy \+ lessons |

## 4.2 Chip portability contract

INVARIANT CORE  The Scenario Director, Serving Analysis Agent, Research Orchestrator, Kernel Engineering Subagent, Verification Gate, Serving Plan Compiler, data contracts, and outer learner share the same logic and contracts across chips.
PER-CHIP CONTEXT  A new chip supplies the machine-topology/toolchain block: engine inventory and primitive costs, memory hierarchy and limits, interconnect, synchronization, compiler/runtime, kernel language, and layout rules. This is context consumed by the chip-neutral mathematical model—not a fork of the orchestration logic.
PER-CHIP CODING SUBAGENT  A new chip also supplies one Kernel Coding Subagent that translates KernelDesignSpec into target source, invokes the chip compiler, and runs the chip harness. The subagent may have chip-specific skills and its own inner/outer coding loop, but it cannot change the engineering model contract or promotion gates.
TPU-FIRST IMPLEMENTATION  The first adapter uses tpu-inference, TPU/ICI topology, MXU/VPU/DMA/SparseCore and HBM/VMEM facts, Pallas/XLA, [KernelSage-ai/autoresearch](https://github.com/KernelSage-ai/autoresearch), and KernelSageBench. A second-chip port succeeds only if these two adapter surfaces—machine context and Kernel Coding Subagent—are sufficient.

# 5. Inner research loop

## Step 0 — Scenario Director freezes the optimization contract

Before profiling, the Scenario Director freezes three groups: (1) model and quantization metadata, including architecture, checkpoint, tokenizer, dtypes, formats, scales, and quality constraints; (2) machine topology and toolchain, including chip/slice, device and interconnect topology, parallel mesh, engine inventory, memory hierarchy and limits, compiler/runtime, kernel language, server command, and repository commit; and (3) traffic and SLO, including the request generator, arrival/concurrency, ISL/OSL, phase mix, warmup, target metrics, budgets, and permissions. Reproducibility is part of the input, not cleanup. The TPU adapter fills group (2) with TPU, ICI, MXU/VPU/DMA/SparseCore, HBM/VMEM, and Pallas/XLA facts.

## Step 1 — Serving Analysis Agent captures baseline evidence

* Capture request/scheduler events, host preprocessing, compiled graph steps, custom calls, chip-engine activity, collectives/interconnect, and memory-transfer timelines with shared step identifiers. For every compiled forward, also log phase, executable fingerprint, padded and actual num\_tokens, padded and active num\_requests, and the identifiers needed to join the forward to latency and trace events. These records are the inputs to the real workload heatmap. On TPU, the timeline includes JAX/XLA, MXU/VPU/DMA/SparseCore, and ICI.
* Validate trace integrity: expected duration and steps, nonzero raw trace, event-count truncation checks, and reconciliation between leaf self-time, engine timelines, and wall time.
* Build an engine-aware critical path from the trace and optimized compiler program. Overlapped work is not exposed work: on TPU, a VPU norm hidden under an MXU GEMM is not equal to an exposed VPU norm, and a SparseCore gather on a noncritical lane should not outrank a serialized main-core gather.
* Calibrate primitive costs at exact tile shapes—such as transfer latency \= descriptor overhead \+ bytes/bandwidth—and retain confidence and residuals as inputs to the Kernel Engineering Subagent’s mathematical model in Step 4\.

## Step 2 — Serving Analysis Agent builds the precompiled-bucket heatmap

**PIVOTAL REDUCTION  This step makes the rest of AutoResearch tractable by converting the open-ended goal “optimize serving on this chip” into a finite set of fully specified, static research problems. Count every forward against its actual precompiled graph key: phase × padded token bucket × padded request bucket, with actual token/request counts attached. Render the measured hit-count and hit-share heatmap over the exact configured ladders. For example, if num\_tokens is precompiled at 8, 16, 32, 64, 128, 256, 512, 1,024, and 2,048, those values are the heatmap columns—not synthetic intermediate sizes. Maintain the additional joint views that change kernel behavior: graph key × KV length, MoE rows × valid fraction, expert shard × routing skew, and sequence length × merge occupancy. The distribution is joint because these values co-occur in one forward and their interaction can change the bottleneck and winning plan. Report hit count, hit share, cumulative share, expected exposed cost, and SLO risk. Within each StaticSliceSpec, the declared execution context is fixed so the kernel model and experiment are predictable.**
For every selected context, emit an immutable StaticSliceSpec that freezes the graph key, exact operator shapes, topology, dtype/quantization, layout/cache state, workload weight p(c), replay inputs, and allowed selection mechanism. The workload weight is the normalized graph-hit share of that complete context in the target trace. The Serving Analysis Agent, Kernel Engineering Subagent, Kernel Coding Subagent, and Verification Gate consume this same slice. This is the key transformation: dynamic, general kernel optimization becomes a reproducible static problem with known inputs, machine behavior, and acceptance criteria.
Only precompiled token buckets with nonzero hit share in the target benchmark or production trace enter the optimization queue; configured but unhit buckets remain compilation and correctness coverage, not tuning targets. Among hit contexts, select by workload-weighted exposed cost while retaining dominant modes and long-tail/SLO contexts. For example, a (2,048 tokens, 16 requests) graph hit by 40% of forwards is a first-class replay target even if another context has higher per-step latency. The output is a weighted replay suite plus crossover-neighborhood cases whose weights reconcile to the target workload.

## Step 3 — Serving Analysis Agent combines compiler and profiler evidence to diagnose bottlenecks

Within each selected context, the Serving Analysis Agent combines compiler dumps with profiler traces to rank the top five opportunities by exposed critical-path contribution, not profiler category totals alone. It maps scenario → full graph key → operator live shape → trace event → optimized compiled op/custom call → graph boundaries around the custom call → source → materialized bytes and lifetimes → engine → dependencies → observed cost. The result is a bottleneck grounded in compiler and profiler evidence diagnosis shared through the immutable StaticSliceSpec.

| Context | Ranked evidence | Question generated |
| :---- | :---- | :---- |
| Decode, 512 reqs | Main-core MoE gather dominates | Is the gathered array real data or a function of its index? |
| Short embedding seqs | Tiny independent attention tiles starve MXU | Can several sequences share one block-diagonal tile? |
| Large ragged gather | Repeated SparseCore pipeline restarts | Where is the chunk-size crossover and what constrains deeper prefetch? |
| Fused attention | Opaque custom call plus graph-side HBM materialization | Can work and memory lifetimes cross the XLA/Pallas boundary? |

## Step 4 — Kernel Engineering Subagent builds the mathematical model and KernelDesignSpec

Because the StaticSliceSpec fixes model, quantization, topology, shapes, memory state, and workload weight, the Kernel Engineering Subagent constructs a precise mathematical model of each hot component and the full kernel. Through the machine-topology interface it computes exact tensor extents, bytes, placements, lifetimes, primitive service times, dependencies, synchronization, launch cost, and host/compiler boundaries. It then produces an AnalyticalKernelModel and KernelDesignSpec covering algorithm, fusion, pipeline, tiling, buffering, placement, memory layout, predicted cost, legal envelope, and fallback. No kernel code is generated before this contract exists.
**MATHEMATICAL MODEL  T̂(o,c) \= scheduleₒ(MXU, VPU, DMA, SparseCore, ICI | exact shapes, bytes, dependencies, lifetimes, semaphores) \+ launch/host overhead. The lower bound is the longest dependency-constrained path, not a sum of overlapped engine times.**
For every key component, the Kernel Engineering Subagent builds the model-backed kernel portion of the OptionMatrix: algorithm and operator flow; engine placement; tile, chunk, and buffer counts; fusion/pipelining; sharding/layout; synchronization; and fallback. The Research Orchestrator adds service/graph choices and prioritization. Cheap filters remove only provably illegal options. Remaining design points are ranked by p(c) × predicted gain × Pr(success) ÷ experiment cost, with bounded exploration around crossover boundaries.

## Step 5 — Kernel Coding Subagent generates and implements the specified options

Every approved KernelDesignSpec becomes a KernelCodingTask for the chip-specific Kernel Coding Subagent. The six categories in Section 6 are opportunity lenses used by the engineering model, not independent top-level agents. The handoff has three explicit stages:

* Input — freeze the StaticSliceSpec, bottleneck grounded in compiler and profiler evidence, baseline, AnalyticalKernelModel, KernelDesignSpec, legal option envelope, edit scope, budget, tests, benchmarks, and ablations.
* Generate \+ implement — the Kernel Coding Subagent materializes every candidate required by the design contract in an isolated worktree, using the target chip’s coding skills, language, intrinsics, compiler, and benchmark harness. On TPU it integrates [KernelSage-ai/autoresearch](https://github.com/KernelSage-ai/autoresearch) and KernelSageBench.
* Return — package source patches, build artifacts, candidate identity, implemented mechanism, compile status, and local correctness/performance results as a branch-local KernelCodingResult. Only the Research Orchestrator updates shared run state; the Verification Gate owns acceptance.

Kernel Engineering Subagent — builds the mathematics before code. It converts the StaticSliceSpec and bottleneck grounded in compiler and profiler evidence into an AnalyticalKernelModel and KernelDesignSpec, deciding algebra, engine placement, graph and fusion boundaries, pipeline, tiling, buffering, synchronization, and memory layout. It is chip-portable because it consumes the common machine-topology context instead of emitting target-specific source.
Kernel Coding Subagent — is the only chip-specific subagent. It follows the KernelDesignSpec to generate and implement every requested candidate, compile it, and optimize it against the chip harness. Its nested inner loop loads coding skills, edits source, compiles, and benchmarks; its outer loop converts outcomes and dead ends into better coding skills. The TPU implementation uses [KernelSage-ai/autoresearch](https://github.com/KernelSage-ai/autoresearch) and KernelSageBench.
Section 6 defines the six opportunity lenses and shows how the Research Orchestrator, Kernel Engineering Subagent, and Kernel Coding Subagent use them in sequence.

## Step 6 — Verification Gate runs controlled experiments

1. Static feasibility: shape divisibility, lane limits, HBM/VMEM/register estimates, sharding constraints, and variant budget.
2. Numerical correctness: reference equivalence over random, adversarial, empty, boundary, ragged, skewed, and multi-device cases; model-quality checks when precision changes.
3. Compilation: all intended token/request buckets compile, cache keys are stable, and no unintended recompilation occurs at runtime.
4. Analytical fidelity \+ kernel microbenchmark: warmup excluded; device time from raw trace; paired, randomized repetitions; for every option, record predicted and measured totals and per-engine time, absolute/relative error, and confidence.
5. Integrated op/graph mechanism: verify HLO/Mosaic, bytes, layout/liveness, and critical path; classify material residuals as bubble, spill/implicit copy, recomputation, serialization, compiler layout, launch/host overhead, or model misspecification.
6. Full serving A/B: replay joint graph-hit weights, dominant modes, crossover neighborhoods, and SLO tails; alternate baseline/candidate order; retain raw results and confidence intervals.

## Step 7 — Serving Plan Compiler updates the plan and restarts the loop

After the Verification Gate accepts a candidate, the Serving Plan Compiler updates and Pareto-prunes the plan. Accepted, rejected, and inconclusive trials all emit a ModelGapLesson: prediction error, residual classification, evidence, and the serving-analysis, mathematical-model, coding-skill, or search-policy update it suggests. The Research Orchestrator immediately re-profiles and re-ranks because the next bottleneck may move across layers or engines. The loop stops when expected remaining value is below experiment cost, the plan hits the target, or the budget is exhausted.

# 6. Opportunity lenses and subagent responsibilities

| Lens | Questions | Representative actions and owner |
| :---- | :---- | :---- |
| Service / flow | Can scheduler choices create better compiled shapes or reduce queuing and padding? | Research Orchestrator: max batched tokens/seqs, phase split, prefill chunking, admission, host-device overlap |
| Graph / compiler | What materializes, zero-fills, concatenates, reshards, spills, or serializes around custom calls, and which cost is absent from the mathematical model? | Serving Analysis Agent \+ Kernel Engineering Subagent: byte/lifetime model grounded in compiler and profiler evidence, graph and fusion boundaries, weight-load transform, donation, prefetch, sharding, and collective placement |
| Operator dispatch | Which registered algorithm and hardware placement wins in each full context? | Kernel Engineering Subagent models TP vs EP GMM, SparseCore vs dense, one-hot vs sort/count, and merged vs per-sequence choices; the TPU Kernel Coding Subagent generates and implements every specified option |
| Kernel engineering | Where do the modeled schedule and device timeline diverge through engine bubbles, pipeline restarts, or resource spills? | Kernel Engineering Subagent: derive the mathematical timeline and choose tiling, buffering, prefetch order, software pipeline, fusion, and engine assignment; Kernel Coding Subagent: implement the contract |
| Mathematical | Is each materialized array real data or a pure function of its index? Is the operation algebraically necessary or replaceable by a stronger formulation? | Kernel Engineering Subagent proves the algebraic contract and chooses fake-gather deletion, index/range identities, segmented formulations, and recomputation-versus-storage; Kernel Coding Subagent implements the selected formulation |
| Memory / layout | What bytes move, where do values live, and for how long? | Kernel Engineering Subagent models exact bytes and lifetimes and selects the dataflow; Kernel Coding Subagent implements it to eliminate memory round-trips, spills, duplicate fetches, and ownership ambiguity |

## 6.1 Mathematical model, option enumeration, and device residuals

For each StaticSliceSpec, the Kernel Engineering Subagent derives the ideal workflow before the Kernel Coding Subagent writes code: minimize dependency-constrained completion time subject to exact bytes, placement, liveness, synchronization, and topology-declared memory/register/lane limits. The result is an AnalyticalKernelModel, a feasible KernelDesignSpec, and a model-backed OptionMatrix—not merely a parameter guess.

* Calibrate every primitive at the exact tile shapes declared by the machine context. In the TPU fused-attention study, this meant fitting DMA overhead and bandwidth, reconstructing MXU/VPU/DMA swimlanes, and matching the v0 model to silicon at 545 µs before changing code.
* Enumerate all registered legal component choices—TP/EP GMM, TensorCore/SparseCore routing, algorithms, tiles, chunks, buffers, fusion, pipeline, layout, and fallback—then prune only by proven shape, lane, memory, correctness, or compile constraints.
* Measure each prioritized option and compute model\_gap \= measured − predicted, plus relative error and per-engine residuals. In the same prior study, a predicted \~28 µs of one-time KV projection conflicted with silicon and exposed repeated K/V recomputation; after pipelining, 438 µs predicted versus 424 µs measured became a bounded calibration residual rather than ignored noise.
* Turn each residual into a typed lesson and sequential route: exposed idle window, extra transfer, repeated compute, serialization, or engine crossover → Kernel Engineering Subagent updates the mathematical model or KernelDesignSpec → Kernel Coding Subagent implements the revised candidate. Require lifetime proof for buffering, retain hard constraints and dead ends, and feed every lesson to the Research Policy Learner.

> Diagram omitted from this text-first context package. Use the adjacent caption or the canonical Google Doc when visual detail is required.
Figure 4\. Prior-work example: the calibrated v0 timeline matched silicon at 545 µs and exposed idle MXU windows; software pipelining reduced the measured kernel to 424 µs. The remaining 14 µs prediction gap is itself a research signal.

## 6.2 Memory-aware graph analysis

The Serving Analysis Agent adds a memory view to the optimized compiler program: logical shape, physical layout, bytes, placement, lifetime, alias/donation, and boundary crossings. It reports peak live memory, repeated materialization, zero-init, conversions, and traffic. The Kernel Engineering Subagent incorporates those facts into the model and KernelDesignSpec; the Kernel Coding Subagent implements the selected dataflow. TPU uses HLO/Mosaic and HBM/VMEM as the first concrete instance.

| Memory symptom | Likely opportunity |
| :---- | :---- |
| Q/K/V written to HBM then immediately read by attention | Fuse projection/norm/RoPE or move the boundary so intermediates remain in VMEM. |
| Static weight concatenation rebuilt per layer/step | Materialize once during weight load and preserve sharding metadata. |
| Donated output buffer zero-filled before full overwrite | Prove overwrite semantics and eliminate initialization. |
| Large operand copied to VMEM on empty/small calls | Async prefetch, persistent/cross-program placement, or shape-specific bypass. |
| Layout changes when a seemingly irrelevant op is removed | Ablate with compiled-layout diff; capture beneficial layout as an explicit constraint. |

## 6.3 Kernel Engineering Subagent: mathematical discovery

The Kernel Engineering Subagent is responsible for discovering mathematical transformations, not only choosing schedules, tiles, fusion boundaries, or layouts. Starting from the ranked bottleneck and fixed serving context, it questions whether each operation and materialized intermediate is mathematically necessary; searches for algebraic identities, streaming recurrences, segmented formulations, and recomputation-versus-storage trades; and derives the validity envelope before any target-specific code is written.
FlashAttention is the canonical pattern: an online softmax formulation maintains running maxima and normalization statistics while processing key/value tiles, so the full attention matrix does not need to be materialized. The mathematical reformulation enables the IO-aware fused kernel; the kernel code is the implementation of that prior result.
The MoE gather investigation provides the same pattern in serving. The engineering agent proved that token\_indices\[topk\_argsort\_indices\] equals topk\_argsort\_indices // topk because token\_indices is a pure function of position, replaced a gathered range mask with scalar bounds, and retained only the embedding gather that moves real token data. The opportunity was deletion by proof, not a faster implementation of unnecessary work.
Every mathematical discovery must emit a proof sketch or equivalence argument, assumptions and validity envelope, predicted byte and critical-path impact, counterexamples or boundary conditions, and required correctness tests and ablations. These become part of KernelDesignSpec. If equivalence is uncertain, the idea remains a hypothesis and cannot be handed to the coding subagent as an approved design.

## 6.4 Kernel Coding Subagent: instruction fidelity, correctness, and skill improvement

The chip-specific Kernel Coding Subagent is a contract-following implementer. KernelCodingTask freezes the StaticSliceSpec, algorithm and dataflow, graph/fusion boundary, legal parameter envelope, fallback, edit scope, and test obligations from KernelDesignSpec. The subagent must implement every requested candidate without silently changing the mathematics; any necessary deviation returns to the Kernel Engineering Subagent for a revised design contract.
KernelSageBench supplies a four-level curriculum that progresses from Level 1 atomic operators, to Level 2 fused operator patterns, Level 3 neural-network blocks and complete models, and Level 4 transformer workloads at serving shapes. The same levels are used for capability evaluation: instruction fidelity, numerical correctness, successful compilation, mechanism verification, and performance are measured separately so a fast but semantically wrong implementation never receives credit.
Its inner loop is executable and bounded: load the reference, search space, KernelDesignSpec, and accumulated discoveries; enumerate required candidates; implement in an isolated workspace; compile; run reference equivalence on random, adversarial, boundary, ragged, and multi-device cases; profile and benchmark only after correctness passes; classify the result; and revise within the declared budget.
Its outer loop turns typed WIN, LOSS, INVALID, and INCONCLUSIVE outcomes, code-review corrections, residuals, and dead ends into better chip-specific coding skills, synthesis guidance, search spaces, and candidate ordering. This is skill improvement through auditable research memory and held-out tasks, not permission to rewrite the mathematical contract or acceptance criteria.
Guardrails are part of the agent design: verify the harness and its hash before a run, protect reference and harness paths, preserve seeds and build artifacts, stop after repeated invalid attempts, keep candidates branch-local, and prohibit self-promotion. The independent Verification Gate remains the authority for correctness and serving acceptance.

# 7. Experiment and verification harness

## 7.1 Harness contract

The Verification Gate—not either subagent—owns acceptance. Its harness asks: Is it correct? Did every intended bucket compile? Did the mechanism occur on the target device? How close were the mathematical model and predicted option cost to reality, what explains the residual, and did the workload-weighted serving objective improve?

| Gate | Required evidence | Failure behavior |
| :---- | :---- | :---- |
| Correctness | Reference/property tests, adversarial shapes, multi-device equality/tolerance | Reject; classify and store counterexample |
| Compilation | All target buckets compile; no cache miss in steady state; resource reports | Reject or narrow envelope |
| Model \+ mechanism | Analytical bound and option prediction; measured per-engine timeline; HLO/Mosaic/trace evidence; residual classification | Do not claim causality or a reusable lesson; diagnose or record bounded uncertainty |
| Microperformance | Paired device-time distribution with noise and effect size | Reject, rerun, or revise crossover |
| Serving | Throughput/TTFT/TPOT/E2E and SLO tails under replay | Do not promote even if microbench wins |
| Operational | Variant count, compile/startup cost, HBM/cache footprint, fallback, maintainability | Pareto-prune or require review |

## 7.2 Statistical discipline

* The Research Orchestrator supplies both subagents the same immutable StaticSliceSpec and baseline, but enforces order: the Kernel Engineering Subagent first emits the AnalyticalKernelModel and KernelDesignSpec; only then does the Kernel Coding Subagent receive a KernelCodingTask. The Verification Gate uses paired, alternating baseline/candidate trials on the same hardware and excludes compilation and warmup explicitly.
* Report sample count, median/mean, robust dispersion, effect size, and bootstrap confidence interval. Tail metrics need enough requests; a three-step trace cannot support a P99 claim.
* Verify trace completeness before analysis. If JSON export is truncated or raw trace is missing, downgrade the claim and use a stronger source such as a new raw capture or full load test.
* Separate predicted, isolated, bundled, and serving effects. If measured device behavior differs materially from the model—or a bundle shows a tail win that isolated trials cannot explain—store the residual, lower causal confidence, and schedule targeted primitive, spill, recomputation, dependency, or interaction ablations.

## 7.3 Promotion rule

**PROMOTE ONLY IF  The independent Verification Gate confirms correctness and a confident workload-weighted win; protected SLOs and compile/cache/memory/complexity budgets pass; every material model gap is explained or explicitly bounded with follow-up; and a safe fallback remains. Only the Serving Plan Compiler may promote.**

# 8. Outer self-improvement loop

The Research Policy Learner consumes TaskSpec, AgentResult, ExperimentRecord, ModelGapLesson, OptionMatrix coverage, and Serving Plan outcomes. In the MVP it updates inspectable routing priors, calibrated cost models, residual classifiers, tool coverage, and experiment order—not the foundation model. Later versions may train proposal or ranking models; no learned policy may change hard gates.

| Learned asset | Update signal | How it is used |
| :---- | :---- | :---- |
| Opportunity priors | Success/failure by model, hardware, context, and mechanism | Rank which lens and transformation to try first. |
| Cost models | Analytical lower bound and predicted vs measured MXU/VPU/DMA/SparseCore/ICI time, bytes, and resource use; residual class | Calibrate future predictions, identify hidden work, and route residuals to the right subagent and opportunity lens. |
| Crossover surfaces | Variant winner over shape, validity, skew, and topology | Seed dispatch policies and choose new boundary experiments. |
| Failure classifier | Compiler error, OOM/register, lane/divisibility, correctness counterexample | Avoid repeated dead ends; propose nearest feasible variant. |
| Harness templates | Which tests and traces caught real bugs | Generate stronger, operator-specific validation suites. |
| Transformation library | Reviewed, generalizable patches | Reuse algebraic, layout, fusion, buffering, and scheduling patterns. |
| Confidence calibration | Predicted win probability vs promoted outcome | Control exploration and accelerator experiment spend. |

## 8.1 Self-improvement protocol

1. Snapshot the Research Policy Learner policy before every run and link every task and experiment to that version.
2. Convert observations into typed lessons: invariant, crossover, prediction residual, hidden-work diagnosis, hard constraint, failed causal explanation, or reusable transformation.
3. The Research Policy Learner generates a candidate next policy and replays it on held-out investigations, including dominant-mode selection, complete option coverage, model-gap diagnoses, dead ends, and self-corrections.
4. Promote only if it finds equal-or-better plans with lower experiment cost, calibrated prediction error, and no degradation in correctness or failure handling.
5. Retain immediate rollback to the previous policy and never allow the Research Policy Learner—or either subagent—to modify hard acceptance gates.

## 8.2 Research memory hierarchy

* **Run memory:** raw traces, patches, measurements, failures, and decisions for one scenario.
* **Operator memory: full OptionMatrix coverage, shape/variant crossover surfaces, invariants, constraints, residual diagnoses, and best-known implementations.**
* **Hardware memory: calibrated unit costs and residuals, bandwidth/overhead, tile behavior, lane/register/VMEM constraints, and compiler-version fingerprints.**
* **Research memory: which sequence efficiently turns an analytical or trace residual into a verified serving win and which explanations failed.**

# 9. TPU-first integration with tpu-inference

The TPU implementation begins in vllm-project/tpu-inference because it already contains token/request padding ladders, a compilation manager that precompiles shape combinations, MoE operator-level switches, SparseCore tuned-parameter mappings, and kernel tuner infrastructure. These become the first adapter for the chip-neutral Scenario Director, Serving Analysis Agent, Research Orchestrator, Kernel Engineering Subagent, Verification Gate, and Serving Plan Compiler. TPU-specific source generation lives behind the TPU Kernel Coding Subagent rather than leaking into core contracts.

## 9.1 Current code anchors

| Area | Current mechanism | AutoResearch extension |
| :---- | :---- | :---- |
| runner/utils.py | Exponential token buckets and bisect-to-next-bucket selection | Record the joint phase × actual/padded token × actual/padded request graph-hit histogram, dominant modes, and an evidence-backed bucket ladder. |
| runner/compilation\_manager.py | Precompiles backbone across token and attention-request padding combinations | Compile a bounded graph/OptionMatrix manifest per full context; record compile cost, fingerprints, and selector/fallback provenance. |
| layers/common/fused\_moe\_gmm.py | Thresholds, chunking, one-hot/dense/SparseCore paths and reduce-scatter options | Register TP/EP GMM, dense/SparseCore, algorithm, tiling, collective, and fallback options in a typed policy with bucket-aware selection. |
| ragged\_gather\_reduce\_tuned\_params.py | TuningKey→TunableParams lookup by input/hidden/reduce-group/dtype | Add hardware/compiler/version and validity/skew envelope; attach benchmark evidence. |
| tools/kernel/tuner/v1 | Kernel-specific search and timing harness | Standard OptionMatrix and ExperimentRecord; analytical predictions, workload-derived cases, model-gap capture, HLO/trace evidence, and E2E escalation. |

## 9.2 Proposed repository structure

**MODULES  Chip-neutral core (tpu\_inference/autoresearch/ initially): scenario, machine\_context, serving\_analysis, static\_slice, analytical\_model, kernel\_design\_spec, option\_registry, orchestrator, verification, residuals, plan\_compiler, policy\_learner, and records. Subagents: kernel\_engineering\_subagent consumes the common machine context and emits AnalyticalKernelModel \+ KernelDesignSpec; chips/tpu/kernel\_coding\_subagent integrates [KernelSage-ai/autoresearch](https://github.com/KernelSage-ai/autoresearch), KernelSageBench, Pallas/XLA, and TPU implementation skills. A future chips/\<new\_chip\>/ adapter adds only its topology/toolchain context and Kernel Coding Subagent. Fusion, Pipelining, SparseCore/TensorCore, Mathematical, and Memory/Layout remain lenses or chip capabilities, not top-level agents. Only the orchestrator writes shared run state; the subagents exchange typed, immutable artifacts; the Plan Compiler accepts only gated IDs.**

## 9.3 Variant binding strategy

Use two levels of specialization:

1. **Compile-time specialization.** Static fields such as padded token/request bucket, dtype, topology, layout, and kernel parameters select separate executables. The compiler constant-folds the chosen path.
2. **Bounded runtime selection.** Dynamic fields such as valid fraction or routing skew may choose between precompiled programs on the host, or between branches inside one program when branch overhead and code size are measured to be acceptable.

A variant budget prevents cartesian explosion. The Serving Plan Compiler starts with one baseline and at most one specialized variant for each high-value context, merges neighboring contexts when the same plan wins within tolerance, and evicts variants whose expected value does not repay compile/cache cost.

## 9.4 Serving instrumentation

* Attach stable scenario, full graph-key/context, OptionMatrix, TaskSpec, and experiment IDs to scheduler output, model execution, custom calls, analytical predictions, and trace events.
* Log every compiled forward-graph hit as (phase, padded tokens, padded requests), with actual counts, DP rank, KV length, prefix-cache behavior, and latency. Materialize hit-count and hit-share heatmaps over the exact configured token and request ladders, plus cumulative share; flag dominant modes and SLO tails. Only nonzero-hit token buckets become tuning targets.
* For MoE, log operator rows, valid fraction, group sizes, per-shard expert load, partition load, and selected path without copying large arrays to host in the hot path.
* Persist executable hash, graph/kernel variant IDs, selected TP/EP and TensorCore/SparseCore paths, tile/chunk/buffer settings, compiler flags, and compile-cache hit/miss.
* Sample expensive telemetry and support replay mode so production overhead remains bounded.

# 10. Data model and artifacts

## 10.1 ExperimentRecord

| Field group | Required fields |
| :---- | :---- |
| Identity | experiment ID, parent task ID, orchestrator policy version, Kernel Engineering Subagent/model version, Kernel Coding Subagent/skill version, chip adapter version, repo commit, patch hash |
| Scenario | model/checkpoint/tokenizer and architecture; quantization/dtypes; chip family and machine topology; engine/memory/interconnect resources; kernel language/compiler/runtime/repository; traffic/SLO contract |
| Context | StaticSliceSpec ID; full graph key plus exact operator live shape, topology, dtype/quantization, layout/cache state, replay inputs, allowed selector features, hit count/share, cumulative share, SLO risk, and expected-cost weight |
| Candidate | OptionMatrix cell/coverage, opportunity lens, bottleneck grounded in compiler and profiler evidence, AnalyticalKernelModel ID, KernelDesignSpec ID, KernelCodingTask/Result ID, edit scope, mechanism, source, engine path, tile/chunk/buffer/fusion/pipeline/layout, selector, fallback, mathematical prediction |
| Correctness | reference, cases, tolerance, model-quality checks, failure counterexamples |
| Performance | raw samples, warmup, predicted and measured total/per-engine time, absolute/relative model gap, residual class/confidence, effect/CI, trace/HLO/Mosaic artifacts, engine counters |
| Resources | compile time, executable bytes, HBM/VMEM/register estimate and measured peaks |
| Decision | Verification Gate result, accepted/rejected/inconclusive reason, causal confidence, ModelGapLesson, Serving Plan Compiler action, reusable lesson, follow-up |

## 10.2 OptionMatrix and ModelGapLesson

The OptionMatrix is the exhaustive registered search surface for one full context. Each row records the component, implementation, TP/EP placement, TensorCore/SparseCore choice, algorithm, tile/chunk/buffer settings, fusion/pipeline/layout, legality and reason, predicted cost, fallback, experiment status, and evidence ID.
Every measured row emits a ModelGapLesson: analytical lower bound, predicted and measured total/per-engine cost, absolute and relative residual, classification, evidence and confidence, and the calibration update, new hypothesis, hard constraint, or tool change to carry forward. The record exists for accepted, rejected, and inconclusive trials.

## 10.3 ServingPlan manifest

The manifest becomes both experiment artifact and runtime configuration. It contains scenario ID, ordered selectors, executable/operator variant IDs, TP/EP and engine choices, tiling/pipeline/layout parameters, context envelope, fallback, accepted ExperimentRecord and ModelGapLesson IDs, OptionMatrix coverage, and resource cost. The Serving Plan Compiler uses only gated results; runtime selection is deterministic and observable.

## 10.4 Evidence bundle

* Raw workload histogram and replay requests
* Raw and normalized traces with integrity report
* HLO/Mosaic/compiler reports and graph diff
* Orchestrator task graph, compiler mapping, StaticSliceSpec, AnalyticalKernelModel, KernelDesignSpec, KernelCodingTask and KernelCodingResult, source patch, both subagent logs and versions, build logs, correctness output, and reference outputs
* Analytical lower bound, complete registered OptionMatrix, predicted-vs-measured residuals and diagnoses, microbenchmark samples, ablations, E2E A/B, statistical summary, and plots
* Decision note including dead ends, limitations, and rollback instructions

# 11. Safety, correctness, and reproducibility

| Risk boundary | Required control |
| :---- | :---- |
| Source mutation | Per-coding-task worktree; KernelDesignSpec and KernelCodingTask edit allowlists; orchestrator-only merge; automatic ablation revert; Verification Gate plus patch/dependency review. The engineering-to-coding dependency is explicit and cannot be bypassed. |
| Hardware execution | Per-run time/cost limit; explicit target device; no destructive VM or data operations; watchdog and cleanup. |
| Numerical behavior | Reference/property suite and tolerance policy; precision/model-quality changes require additional approval. |
| Deployment | No direct production mutation; Plan Compiler emits PR evidence and canary recommendation; human approval, reversible selector, and baseline fallback remain required. |
| Self-improvement | Versioned Research Policy Learner, mathematical-model calibration, and per-chip Kernel Coding Subagent skill state; held-out replay, immutable gates, reviewable diffs, and rollback; no learner or subagent can self-promote. |
| Claims | Every percentage linked to raw samples and scenario; report mathematical bound, prediction, device result, model gap, mechanism, isolated effect, bundle effect, and E2E effect separately. |

# 12. Evaluation plan

## 12.1 Replay the four prior investigations

| Case | Capability under test | Expected research behavior |
| :---- | :---- | :---- |
| MoE fake gathers | Trace-to-source \+ mathematics | Serving Analysis Agent maps and ranks all three gathers; the Kernel Engineering Subagent proves token\_indices\[idx\] \= idx // top\_k, replaces a permuted range mask with scalar bounds, and emits the algebraic KernelDesignSpec; the TPU Kernel Coding Subagent implements it and confirms the remaining embedding gather is real data; Verification Gate proves equivalence and E2E. |
| SparseCore dispatch and tiling | Distribution \+ crossover \+ hard constraints | Cartographer recovers dominant live shapes and validity; the OptionMatrix sweeps TensorCore vs SparseCore and chunk/tile settings. Gate should rediscover the \<2,048-row dense fallback, validity-dependent reduce crossover, \~0.58×/1.13× chunk crossover, and hard lane/register/VMEM dead ends. |
| RPAmerged | Algorithm and layout redesign | Serving Analysis Agent detects MXU starvation; the Kernel Engineering Subagent models and selects block-diagonal packing; the TPU Kernel Coding Subagent implements it; and the orchestrator re-profiles the CPU bottleneck shift. |
| Fused projection-attention | Cross-boundary memory \+ multi-engine scheduling | Serving Analysis Agent calibrates unit ops and reconstructs MXU/VPU/DMA swimlanes; the Kernel Engineering Subagent builds a model that matches v0 at 545 µs and emits the 438 µs fusion/pipelining KernelDesignSpec; the TPU Kernel Coding Subagent implements it. The Gate measures 424 µs, explains or bounds the 14 µs residual, and records the lesson. |

## 12.2 MVP success criteria

* Reproduce the joint graph-hit/context distributions—including dominant modes and SLO tails—and identify the true top opportunity in at least three of the four replays.
* For at least two hot components, demonstrate the complete sequential handoff from one immutable StaticSliceSpec and bottleneck grounded in compiler and profiler evidence to an AnalyticalKernelModel and KernelDesignSpec, then to every specified TPU kernel candidate and independent verification. Report predicted-versus-measured gaps for every tested option and produce evidence-backed patches for at least two cases, including one cross-layer change.
* Correctly retain at least one negative result or hard constraint instead of fabricating an optimization.
* Emit a Serving Plan with two bucket-specific variants and demonstrate zero unintended recompilations under replay.
* Improve a chosen production-like serving objective with no protected-metric regression, then re-profile and identify the next bottleneck.
* Show that the updated Research Policy Learner reduces experiments-to-solution on one held-out replay without weakening gates.

## 12.3 Metrics

| Category | Metrics |
| :---- | :---- |
| Serving | req/s, input/output/total tok/s, mean/P50/P99 TTFT, TPOT, inter-token, E2E, SLO attainment |
| Device | step latency, exposed critical path, engine occupancy, kernel/custom-call time, collective time, bytes moved |
| Research | time/accelerator-hours to first accepted change, experiments per acceptance, KernelDesignSpec coverage, candidate implementation coverage, model absolute/relative error, explained-residual share, compiler-mapping precision, engineering-to-coding handoff success, failed-build rate, prediction calibration |
| Plan | variant count, compile time, startup time, executable cache bytes, coverage, fallback rate, regression rate |

# 13. Delivery roadmap

| Phase | Deliverable | Exit criterion |
| :---- | :---- | :---- |
| 0 — Observability | Scenario Director and Serving Analysis Agent: three-category scenario manifest, trace integrity, full graph keys, precompiled-bucket heatmap, StaticSliceSpec set, ranked bottleneck list, and replay | One run reproducibly maps requests → buckets → ops → source. |
| 1 — Single-op research | Orchestrator, Serving Analysis Agent with compiler-and-profiler evidence mapping, Kernel Engineering Subagent, AnalyticalKernelModel and KernelDesignSpec schemas, TPU Kernel Coding Subagent, option registry, Verification Gate, and TPU capability/skill adapters | Complete legal option coverage; calibrated predictions; competing tasks independently rediscover engine and chunk crossovers and explain material residuals. |
| 2 — Bucketed plans | Serving Plan Compiler: variant registry, compilation integration, selector, evidence links, and fallback | Two variants improve weighted objective without recompilation. |
| 3 — Cross-layer co-design | Kernel Engineering Subagent model extensions for fusion/pipelining and cross-boundary memory, plus TPU Kernel Coding Subagent implementation and joint graph/kernel experiments | Land one E2E win not expressible as kernel-parameter tuning. |
| 4 — Portability \+ outer loop | Research Policy Learner: typed research memory, model/coding-skill improvement, candidate ranking, and a second-chip adapter consisting only of machine-topology context plus a new Kernel Coding Subagent | Lower search cost on an unseen TPU scenario and prove the core loop ports to a second chip without changing chip-neutral orchestration, modeling, verification, or plan contracts. |

# 14. Risks and mitigations

| Risk | Mitigation |
| :---- | :---- |
| Combinatorial search | Complete registered OptionMatrix followed by proof-based feasibility pruning; Research Orchestrator task graph; hypothesis deduplication; top-context budgets; subagent concurrency caps; active learning at crossover boundaries; Pareto pruning. |
| Overfitting to a trace | Rolling and held-out windows; phase/tail preservation; replay plus live canary; envelope/fallback semantics. |
| Compile/cache explosion | Variant budget, neighborhood merging, explicit charge in objective, cache telemetry and eviction. |
| Noisy or misleading profiles | Raw trace integrity checks, self-time reconstruction, total reconciliation, paired trials, stronger-source fallback. |
| Incorrect causal story | Model-vs-device residuals, targeted ablation, compiled-program diff, mechanism counters, and separate prediction/isolated/bundled/E2E claims. |
| Opaque compiler behavior | Treat HLO/Mosaic and model gaps as evidence; fingerprint compiler version; retain layout-sensitive and spill/serialization counterexamples. |
| Unmaintainable patches | Scoped TaskSpec/AgentResult, isolated worktrees, Verification Gate, minimal variants, transformation library, code-owner review, and complexity cost. |
| Outer-loop regression | Immutable Verification Gate rules, shadow replay, versioned Research Policy Learner state, rollback, and human promotion. |

# 15. Open decisions

* Objective ownership: who sets the throughput/TTFT/TPOT weights and protected SLOs for each run?
* Variant and option budget: which implementations must be registered and exhaustively modeled, and how many full-graph executables or operator variants may one Serving Plan retain per hardware/model pair?
* Dynamic selection: which runtime features—especially request bucket, live rows, valid fraction, and routing skew—may be read cheaply and safely before model execution?
* Subagent contract and versioning: what schema version connects compiler evidence → AnalyticalKernelModel → KernelDesignSpec → KernelCodingTask/Result; which model and coding-skill changes require review; and how is the sequential engineering-to-coding dependency enforced?
* Workload source: production rolling trace, benchmark contract, customer-supplied replay, or a mixture?
* Promotion: what statistical confidence, minimum materiality, model-error tolerance, and unexplained-residual bound justify a PR, default flip, or automatic canary proposal?
* Storage and retention: which raw traces and compiled artifacts are retained, and how are sensitive request features anonymized?

**RECOMMENDED FIRST DECISION  Define one canonical Qwen3-30B-A3B / TPU v6e-8 replay scenario with explicit model/quantization, TPU machine-topology/toolchain, and traffic/SLO metadata. Implement the chip-neutral Serving Analysis Agent contracts—precompiled-bucket heatmap, StaticSliceSpec set, and compiler/profile evidence mapping—then the Kernel Engineering Subagent’s AnalyticalKernelModel \+ KernelDesignSpec, then the TPU Kernel Coding Subagent using KernelSage-ai/autoresearch, followed by the independent Verification Gate and Serving Plan Compiler. Replay the existing MoE investigations to exercise dominant buckets, TP/EP and engine dispatch, mathematical deletion, tiling, fusion/pipelining, compiler boundaries, model gaps, tails, and E2E verification. Treat a second chip as an adapter test: only its machine context and Kernel Coding Subagent should be new.**

# Appendix A. Evidence from prior work

| Prior result | Design requirement derived |
| :---- | :---- |
| Two of three MoE gathers were algebraically fake: token\_indices\[idx\] \= idx // top\_k, and a permuted range mask became scalar bounds; the remaining embedding gather moved real data. Routing gather fell from \~49.6% to \~1.8% of step. | The Kernel Engineering Subagent must question every materialization, prove the algebraic contract across random and boundary shapes, and prefer deletion over moving fake work to another engine; the Kernel Coding Subagent implements the proved contract. |
| At 512 concurrent 1,024/1,024 requests, optimized MoE arithmetic improved throughput \~7.4% and P99 TTFT \~74.8% vs the legacy bundle. | Full serving replay and tail metrics are required; kernel/device effects propagate nonlinearly through queuing. |
| Ragged gather traffic concentrated at a dominant 4,096-row shape; TensorCore won below 2,048 rows; reduce crossover depended on valid fraction; larger chunks lost \~0.58× below their block size and won \~1.13× above it. | Capture joint production modes; enumerate TensorCore/SparseCore and tiling options; learn size × validity crossovers; retain lane/register/VMEM dead ends; bind variants only with gated evidence. |
| Interleaving skewed partitions improved ragged\_gather\_reduce microbench 1.62×–5.45×, while isolated E2E impact varied by phase. | Represent routing skew and phase; separate microbench, isolated E2E, and bundled causality. |
| RPAmerged cut attention from 107 ms to 14 ms per step for short sequences and made CPU overhead 43% of step. | Re-profile after each win and allow the target to move from kernel to host/service. |
| Projection-attention v0 modeled and measured 545 µs; the pipelined plan modeled 438 µs and measured 424 µs. Reconstructing MXU/VPU/DMA swimlanes exposed idle windows, repeated K/V recomputation, and graph-side concatenation/zero-fill. | Require the mathematical timeline to reconcile with silicon. Treat both visible bubbles and prediction residuals as first-class opportunities, and learn from the remaining error after a win. |

**Internal sources:** Teaching the Machine to Read Its Own Trace; SparseCore in MoE: Early Optimization Lessons; RPAmerged: 1.77× Embedding Throughput on TPU; Teaching the Machine to Pipeline a TPU Kernel.

# Appendix B. Repository references

* **Token/request bucket construction:** [tpu\_inference/runner/utils.py](https://github.com/vllm-project/tpu-inference/blob/945630867052ffba328917b35911efd0a6364fc5/tpu_inference/runner/utils.py)
* **Shape precompilation:** [tpu\_inference/runner/compilation\_manager.py](https://github.com/vllm-project/tpu-inference/blob/945630867052ffba328917b35911efd0a6364fc5/tpu_inference/runner/compilation_manager.py)
* **MoE flow, chunking, and dispatch:** [tpu\_inference/layers/common/fused\_moe\_gmm.py](https://github.com/vllm-project/tpu-inference/blob/945630867052ffba328917b35911efd0a6364fc5/tpu_inference/layers/common/fused_moe_gmm.py)
* **SparseCore gather-reduce kernel:** [tpu\_inference/kernels/sparse\_core/ragged\_gather\_reduce\_v2.py](https://github.com/vllm-project/tpu-inference/blob/945630867052ffba328917b35911efd0a6364fc5/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py)
* **Tuned parameter mapping:** [tpu\_inference/kernels/sparse\_core/ragged\_gather\_reduce\_tuned\_params.py](https://github.com/vllm-project/tpu-inference/blob/945630867052ffba328917b35911efd0a6364fc5/tpu_inference/kernels/sparse_core/ragged_gather_reduce_tuned_params.py)
* **Kernel tuner:** [tools/kernel/tuner/v1/ragged\_gather\_reduce\_kernel\_tuner.py](https://github.com/vllm-project/tpu-inference/blob/945630867052ffba328917b35911efd0a6364fc5/tools/kernel/tuner/v1/ragged_gather_reduce_kernel_tuner.py)

*Repository references were inspected on September 7, 2026\. The linked revision is used for stable design grounding; implementation should revalidate against the chosen working commit before coding.*

# Appendix C. Review checklist

* Approve the Serving Plan, OptionMatrix, and ModelGapLesson as the primary deployable, search-space, and learning abstractions.
* Choose the first canonical scenario and workload trace.
* Set primary objective, protected SLOs, run budget, and variant budget.
* Approve the initial edit scope and human review boundary.
* Select Phase 0 instrumentation owners and the first replay case.
