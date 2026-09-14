---
document_id: autoresearch-phase0-scenario-serving-analysis
title: "Scenario Director and Serving Analysis Agent — Detailed Design"
format: agent-readable-markdown
status: design
canonical_source:
  type: google_doc
  title: "Scenario Director and Serving Analysis Agent Detailed Design"
  url: "https://docs.google.com/document/d/1ZwuZ5udWU3Xq_dM-Hbjbv8mRk0KGG7RooVXtRj1Dwn4/edit"
  revision: "ANLCKQmp-OP94hvf5009ceLuXyiyDaDTMC5aevD92mYFYBV-G3TtWyJAOMPFHce-MfLZ53MST4UpUiOOG0CDWx-qYiFI0o476Jp53lrDnlA"
repository_target:
  repository: windtara0619/tpu-inference
  baseline_commit: a4cfff0f10cee7525295236ce7cd13eba230633e
normative_scope: "Phase 0 implementation of Scenario Director and Serving Analysis Agent"
sync_policy: "When this file conflicts with the canonical source at the recorded revision, treat the Google Doc as authoritative. If its revision changed, review and resync before implementation."
---

> Agent context note: This is a text-first export for implementation agents. Embedded diagrams were replaced by their captions. Revalidate repository paths and APIs against the checked-out commit before changing code.
# Scenario Director and Serving Analysis Agent Detailed Design

## Phase 0 implementation design for the AutoResearch Kernel Agent

| Field | Value |
| :---- | :---- |
| Status | Proposed |
| Audience | Serving, compiler, performance and kernel engineers |
| Scope | Optimization-contract freezing, workload reconstruction, compiler and profiler attribution, static replay contexts and bottleneck ranking |
| Source architecture | [AutoResearch Kernel Agent for New Chips — Design Doc](https://docs.google.com/document/d/1eDtoOujHeuqTQauXiozcyFd2gp1RyDCcIwws42XBJU0/edit) |

# Executive summary

Phase 0 establishes a trustworthy boundary between a user's optimization request and downstream kernel research. The Scenario Director converts that request into one validated, immutable optimization contract. The Serving Analysis Agent then converts real service, scheduler, compiler and device evidence into a finite set of fully specified replay contexts and ranked bottlenecks. Every later mathematical model, generated implementation and benchmark is attributable to the same workload and machine conditions.

Ownership is strict. The Scenario Director defines and freezes what a run may optimize but does not interpret profiles. The Serving Analysis Agent observes, joins, aggregates and diagnoses evidence but does not redesign kernels, mutate source or promote results. Both emit typed immutable artifacts. The Research Orchestrator is the only writer of shared run state.

* Scenario Director output: one content-addressed ServingScenario and its validation report.
* Serving Analysis output: a graph-hit heatmap, selected static replay contexts, evidence graph, ranked bottleneck list and replay bundle.
* Phase 0 exit criterion: one run maps requests to graph buckets to hot operations to source, reproducibly and without relying on profiler category totals alone.

## Document map

* 1 Goals and boundaries
* 2 System context and ownership
* 3 Scenario Director design
* 4 Serving Analysis Agent design
* 5 Shared artifacts and storage
* 6 TPU first integration
* 7 Reliability security and observability
* 8 Verification plan
* 9 Delivery plan
* 10 Open decisions
* Appendices with schemas and canonical replay

# 1 Goals and boundaries

## 1.1 Goals

* Freeze model, quantization, topology, toolchain, repository revision, workload, objective, SLOs, budgets and permissions before evidence collection.
* Measure the real phase x padded-token x padded-request distribution over the exact executable ladder used by the server.
* Preserve actual counts, KV state, live rows, validity and routing skew inside compiled buckets.
* Join scheduler, host, compiler and device evidence through stable identifiers and expose graph-side costs around custom calls.
* Produce deterministic replay inputs and source-grounded bottlenecks for downstream engineering.
* Keep the contracts chip-neutral; each accelerator supplies machine-context and evidence adapters.

## 1.2 Non goals

* The Scenario Director does not propose algorithms, schedules or source changes.
* The Serving Analysis Agent does not emit KernelDesignSpec, write chip code or approve claims.
* Phase 0 does not search the OptionMatrix or compile a production Serving Plan.
* Missing configuration is never inferred from stale logs.
* Unobserved configured buckets remain compilation and correctness coverage, not tuning targets.

## 1.3 Invariants

| Invariant | Required behavior |
| :---- | :---- |
| One frozen contract | All observations, replay inputs and downstream tasks reference one scenario ID and schema version. |
| Evidence before interpretation | Every observation carries provenance and every diagnosis cites evidence IDs. |
| Logical-forward accounting | Hit weights count a logical compiled forward once. DP-rank events are child evidence and do not multiply workload share. |
| Exact executable ladders | Heatmap axes come from configured token and request ladders, never invented intermediate sizes. |
| Static research input | Each selected replay context fixes shapes, machine state, inputs and acceptance criteria. |
| No self promotion | Neither component may change SLOs, approve candidates or weaken verification gates. |
| Reproducible failure | Incomplete evidence produces a typed failure report rather than a speculative bottleneck. |

# 2 System context and ownership

## 2.1 Phase 0 dataflow

| Step | Owner | Input | Output |
| :---- | :---- | :---- | :---- |
| 1 | Scenario Director | Requested optimization contract | Validated normalized draft |
| 2 | Scenario Director | Validated draft and adapter probes | Frozen ServingScenario |
| 3 | Serving Analysis Agent | Frozen scenario and runtime evidence | Integrity-checked forward records |
| 4 | Serving Analysis Agent | Forward records and executable ladders | Graph-hit heatmap and selected contexts |
| 5 | Serving Analysis Agent | Selected contexts plus compiler and profile evidence | StaticSliceSpecs, evidence graph, bottlenecks and replay suite |
| 6 | Research Orchestrator | Phase 0 outputs | Kernel Engineering tasks |

## 2.2 Ownership boundaries

| Concern | Scenario Director | Serving Analysis Agent | Downstream owner |
| :---- | :---- | :---- | :---- |
| Objective and SLO | Validate and freeze | Read only | Verification Gate enforces |
| Machine context | Resolve and freeze adapter version | Use for attribution | Kernel Engineering models |
| Runtime evidence | Define required sources and permissions | Capture, validate and join | Verification reuses |
| Static replay context | Define schema constraints | Build and emit | Engineering and Coding consume |
| Bottleneck diagnosis | Out of scope | Rank with evidence | Engineering asks design questions |
| Source mutation | Forbidden | Forbidden | Kernel Coding in isolated worktree |
| Promotion | Forbidden | Forbidden | Serving Plan Compiler after verification |

## 2.3 Deterministic and agentic planes

The Serving Analysis Agent is not a free-form profiler assistant. Deterministic collectors own identifiers, joins, bucket assignment, aggregation, integrity checks and numerical ranking. Agentic reasoning may classify evidence, propose causal questions and request additional captures, but it cannot alter measured values or emit an unreferenced claim. Every diagnosis records evidence IDs, mapping method, ambiguity and confidence.

# 3 Scenario Director design

## 3.1 Responsibility

The Scenario Director is the control-plane entry point for one AutoResearch run. It turns a human or API request into a normalized, validated and immutable ServingScenario. The frozen artifact is the sole authority for the model, machine, software, workload, objective, resource and permission envelope used by Phase 0\.

## 3.2 Input groups

| Group | Required content | Resolution rule |
| :---- | :---- | :---- |
| Model and quantization | Model, checkpoint and tokenizer revisions; architecture dimensions; weight, activation, cache and accumulation dtypes; quantization format and quality constraints | Resolve immutable revisions and record digests. |
| Machine and toolchain | Chip and slice, device/interconnect topology, parallel mesh, engine inventory, memory hierarchy, compiler/runtime, kernel language and adapter version | Record declared and detected values; fail on material disagreement. |
| Traffic and SLO | Arrival or concurrency, ISL/OSL, phase mix, prefix reuse, KV behavior, precompiled ladders, target metrics and protected tails | Require a replayable workload source and measurement policy. |
| Service policy | Maximum batched tokens and sequences, scheduling, prefill, admission, caching and sharding | Freeze effective runtime values, not only requested flags. |
| Budgets | Wall-clock, accelerator, trial, compilation, cache and variant limits | Treat limits as immutable. |
| Permissions | Readable inputs, artifact destinations, profiling permission, source edit scope and deployment boundary | Phase 0 receives no source or production mutation permission. |
| Provenance | Repository remote and commit, dirty state, commands, environment hash and dependencies | Record a patch digest or reject untracked state by policy. |

## 3.3 ServingScenario schema

ServingScenario
  schema\_version: string
  scenario\_id: digest
  model: ModelQuantizationSpec
  machine: MachineContextRef
  toolchain: ToolchainSpec
  traffic\_slo: TrafficSLOSpec
  service\_policy: ServicePolicySpec
  objective: ObjectiveSpec
  budgets: BudgetSpec
  permissions: PermissionSpec
  provenance: ProvenanceSpec
  required\_evidence: EvidenceRequirement\[\]
  lifecycle: FROZEN

The scenario ID is the digest of canonical serialized content after aliases are resolved. Human-readable labels do not participate in equality. Credentials and raw prompts are stored behind protected references and are excluded from the hash payload.

## 3.4 Validation

| Class | Examples | Failure |
| :---- | :---- | :---- |
| Completeness | Model revision, machine, mesh, workload, objective and SLO exist | MISSING\_REQUIRED\_FIELD |
| Compatibility | Mesh product matches devices; dtype is supported; ladders cover limits | INCOMPATIBLE\_CONFIGURATION |
| Reproducibility | Repository commit, command, workload source and seed resolve | UNRESOLVED\_PROVENANCE |
| Safety | Budgets are valid; writable paths are scoped; deployment mutation is disabled | UNSAFE\_PERMISSION |
| Consistency | Declared and detected topology agree; flags match effective runtime state | RUNTIME\_CONFIG\_MISMATCH |
| Replayability | Requests can be regenerated or replayed without leaking private payloads | NON\_REPLAYABLE\_WORKLOAD |

## 3.5 Lifecycle and API

| State | Meaning | Transition |
| :---- | :---- | :---- |
| DRAFT | Mutable and possibly incomplete | VALIDATING or CANCELLED |
| VALIDATING | Adapters resolve versions and constraints | FROZEN or FAILED |
| FROZEN | Immutable and usable by analysis | SUPERSEDED through a new scenario |
| FAILED | Typed validation report is final | Corrected request creates a new draft |
| SUPERSEDED | Retained for provenance | None |

class ScenarioDirector:
    def validate(request: ScenarioRequest) \-\> ScenarioValidationReport: ...
    def freeze(request: ScenarioRequest) \-\> ServingScenarioRef: ...
    def get(scenario\_id: str) \-\> ServingScenario: ...
    def diff(left\_id: str, right\_id: str) \-\> ScenarioDiff: ...

**Idempotency.** Canonically identical content returns the existing scenario. Any change to objective, SLO, commit, topology or workload source creates a new ID.

**Concurrency.** Concurrent identical freezes converge on one digest. Conflicting contracts remain distinct.

**Auditability.** The validation report lists every default, adapter probe, resolved value and warning. Performance-sensitive defaults are never hidden.

## 3.6 Failure behavior

* Reject missing protected SLOs, quantization constraints or repository revisions rather than guessing.
* Permit declared topology without probing only when policy allows it and label the source as declared.
* Fail if a mutable model or branch resolves differently between validation and freeze.
* Store only anonymized replay handles when request payloads are sensitive.
* Supersede scenarios; never edit a frozen scenario in place.

# 4 Serving Analysis Agent design

## 4.1 Responsibility

The Serving Analysis Agent consumes one frozen scenario and evidence from the running service. It reconstructs which precompiled programs execute, identifies contexts that dominate workload cost or SLO risk, maps optimized operations and device events to source, and emits a finite research queue. Its output is diagnostic, not a kernel design.

## 4.2 Evidence inputs

| Source | Minimum fields | Use |
| :---- | :---- | :---- |
| Request stream | Request ID, arrival, input/output tokens, completion, latencies, replay seed or handle | Traffic and SLO attribution |
| Scheduler | Logical forward ID, request membership, phase, scheduled tokens, active requests, preemption and cache state | Batch reconstruction |
| Model execution | Forward ID, DP rank, actual/padded counts, executable fingerprint and selected paths | Graph-hit accounting |
| Compiler | Fingerprint, optimized IR operations, layouts, memory reports and source metadata | Graph and source mapping |
| Host profile | Preprocessing, dispatch, synchronization and launch events with correlation IDs | Host critical path |
| Device profile | Engine events, custom calls, collectives, transfers, dependencies and timestamps | Device critical path |
| Calibration | Primitive shapes, bytes, setup cost, latency distribution and confidence | Cost normalization |

## 4.3 Analysis pipeline

### 4.3.1 Capture

Attach scenario ID, analysis ID, logical forward ID, executable fingerprint and DP rank across scheduler, model and profile records. Each rank emits a child record; a coordinator emits one logical-forward record. Workload weights count the logical record once, while child events support skew and collective analysis.

### 4.3.2 Integrity gate

* Confirm nonzero artifacts, expected duration and expected steps.
* Detect event caps, truncation, dropped events and clock discontinuities.
* Reconcile wall time with scheduler steps, leaf self-time and engine timelines within declared tolerances.
* Require each analyzed forward to resolve to one scenario and one executable fingerprint.
* A failed integrity gate cannot produce a high-confidence bottleneck.

### 4.3.3 Graph-hit workload

Assign every logical compiled forward to the exact padded token and request buckets selected by the runtime. Preserve actual counts inside each cell. The primary key is phase x padded tokens x padded requests. Joint secondary views capture KV length, prefix reuse, MoE live rows, valid fraction, expert-load skew and merged-sequence occupancy.

cell\_hits(c) \= number of logical compiled forwards in context c
p(c) \= cell\_hits(c) / total logical compiled forwards
expected\_exposed\_cost(c) \= p(c) \* exposed\_time(c)

### 4.3.4 Context selection

* Do not tune zero-hit token buckets; retain them for compile and correctness coverage.
* Rank hit contexts by workload-weighted exposed cost, not frequency or single-step latency alone.
* Retain dominant modes until the configured coverage target is reached.
* Retain protected SLO-tail contexts regardless of small hit share.
* Retain measured neighborhoods around engine, sharding, tiling or algorithm crossovers.
* Publish the full heatmap even when only a bounded subset enters research.

### 4.3.5 Static replay contexts

StaticSliceSpec is the implementation record for one selected, fixed research context. It is created only after the workload is measured. The record binds the executable key, exact operator shapes, topology, layout/cache state, workload weight, deterministic replay inputs, selector eligibility and acceptance criteria. Downstream agents cannot widen the context silently.

### 4.3.6 Compiler and profiler evidence graph

Use an evidence graph rather than a flat table. Nodes represent forwards, executables, optimized operations, custom calls, host events, device events, buffers, source locations and artifacts. Typed edges represent compilation, containment, launch, dependency, allocation, aliasing, source provenance and temporal correlation. Each edge stores mapping method, confidence and ambiguity.

| Mapping | Preferred key | Fallback | Confidence |
| :---- | :---- | :---- | :---- |
| Forward to executable | Explicit fingerprint | Shape and cache entry | Inferred matches are not high confidence |
| Executable to operation | Compiler operation ID | Normalized signature | Signature collisions lower confidence |
| Operation to trace | Correlation ID | Name, shape and time | Uniqueness required for high confidence |
| Operation to source | Source metadata or target | Stack and symbol search | Multiple sites remain ambiguous |
| Buffer to memory event | Allocation or buffer ID | Shape, layout and lifetime | Shape-only bytes are labeled inferred |

### 4.3.7 Exposed critical path

Build a dependency DAG from program dependencies, resource serialization and synchronization. Baseline step time is the longest source-to-sink path. For event e, exposed contribution is the longest-path reduction when e's service time becomes zero while dependencies remain. This prevents overlapping MXU, VPU, DMA, SparseCore and collective work from being double-counted.

critical\_path \= longest\_path(event\_dag)
exposed(e) \= duration(critical\_path) \- duration(critical\_path with service\_time(e)=0)
workload\_impact(c,e) \= p(c) \* exposed(c,e)

### 4.3.8 Bottlenecks and replay

Emit up to five candidates per selected context, ordered by workload impact. Each candidate records symptom, exposed time, source and graph boundaries, engine/memory evidence, alternatives, confidence and the next discriminating experiment. The replay bundle contains exact or synthetic-equivalent inputs, expected graph key and shapes, reference outputs, server/client commands, commits, evidence links and inherited acceptance criteria.

## 4.4 Data contracts

### 4.4.1 ForwardExecutionRecord

ForwardExecutionRecord
  scenario\_id, analysis\_id, logical\_forward\_id
  rank\_event\_id, dp\_rank, phase
  actual\_tokens, padded\_tokens
  active\_requests, padded\_requests
  kv\_length\_bucket, prefix\_cache\_state
  executable\_fingerprint
  operator\_observations\[\]
  scheduler\_timestamps, latency\_links\[\], provenance

### 4.4.2 GraphHitCell

GraphHitCell
  graph\_key
  hit\_count, hit\_share, cumulative\_share
  actual\_token\_distribution
  active\_request\_distribution
  kv\_length\_distribution
  operator\_live\_row\_distribution
  validity\_and\_routing\_skew
  exposed\_time\_distribution
  slo\_risk\_summary

### 4.4.3 StaticSliceSpec

StaticSliceSpec
  slice\_id, scenario\_id, analysis\_id
  graph\_key, executable\_fingerprint
  exact operator shapes and dtypes
  topology, sharding, layout, cache state
  fixed or bucketed dynamic characteristics
  workload\_weight, replay\_bundle\_ref
  allowed\_selector\_features
  acceptance\_criteria, evidence\_refs\[\]

### 4.4.4 BottleneckCandidate and AnalysisReport

BottleneckCandidate
  bottleneck\_id, slice\_id
  operation, source boundary, engine, memory path
  observed\_cost, exposed\_cost, workload\_impact
  evidence\_refs\[\], mapping\_confidence
  alternative\_explanations\[\]
  next\_question, discriminating\_capture

AnalysisReport
  integrity\_report\_ref, heatmap\_ref
  selected\_slice\_refs\[\], evidence\_graph\_ref
  ranked\_bottleneck\_refs\[\], replay\_suite\_ref
  warnings\[\]

## 4.5 API and reruns

class ServingAnalysisAgent:
    def capture(scenario, sources) \-\> CaptureRef: ...
    def validate\_capture(capture) \-\> IntegrityReport: ...
    def reconstruct\_workload(capture) \-\> GraphHitHeatmapRef: ...
    def select\_contexts(heatmap, policy) \-\> list\[StaticSliceSpecRef\]: ...
    def map\_evidence(slices, capture) \-\> EvidenceGraphRef: ...
    def rank\_bottlenecks(graph) \-\> list\[BottleneckCandidateRef\]: ...
    def build\_replay(slices) \-\> ReplaySuiteRef: ...
    def analyze(scenario, sources) \-\> AnalysisReportRef: ...

**Idempotent reprocessing.** The same scenario, evidence digests and policy yield the same analysis ID.

**New evidence.** A new trace window creates a new analysis version under the unchanged scenario.

**Partial repair.** A failed profile may be replaced without repeating request capture only when forward IDs and time windows remain join-compatible.

**Rolling windows.** Preserve window identity so workload drift and tails are not hidden inside a lifetime average.

# 5 Shared artifacts and storage

## 5.1 Identity and layout

Artifacts are immutable and content-addressed. References carry schema version, producer version, scenario ID, parent IDs and payload digest. Mutable labels such as latest are convenience pointers and cannot appear in acceptance evidence.

runs/\<scenario\_id\>/\<analysis\_id\>/
  scenario/serving\_scenario.json
  scenario/validation\_report.json
  capture/capture\_manifest.json
  capture/forward\_records.parquet
  capture/raw\_profiles/
  compiler/executables.json
  compiler/ir/
  workload/graph\_hit\_heatmap.parquet
  slices/\<slice\_id\>/static\_slice\_spec.json
  evidence/evidence\_graph.parquet
  bottlenecks/ranked\_bottlenecks.json
  replay/replay\_manifest.json
  integrity/integrity\_report.json

## 5.2 Formats and evolution

| Artifact | Format | Reason |
| :---- | :---- | :---- |
| Contracts and summaries | Canonical JSON | Readable, hashable and schema-validatable |
| Forward and event records | Parquet | Efficient typed scans over large captures |
| Compiler IR | Native text or protobuf plus digest | Preserve source fidelity |
| Profiles | Raw xplane/native trace plus normalized Parquet | Retain strong evidence and support joins |
| Replay inputs | Protected object bundle plus manifest | Separate payloads from metadata |
| Figures | Derived SVG or PNG | Review aid, never sole evidence |

* Compatible optional-field additions increment a minor schema version; semantic changes increment the major version.
* Migrations create new artifacts and retain original digests.
* Readers reject unknown required fields and preserve unknown optional fields.
* Every analysis records producer code, analysis policy and chip-adapter versions.

# 6 TPU first integration

## 6.1 Existing code anchors

| Anchor | Current behavior | Phase 0 extension |
| :---- | :---- | :---- |
| runner/utils.py | Constructs token/request ladders and selects padded shapes | Export ladders and actual/padded counts for each forward. |
| runner/compilation\_manager.py | Precompiles token x attention-request shape combinations | Fingerprint executables and persist compile manifests. |
| Scheduler and input preparation | Select requests and scheduled tokens | Emit forward ID, membership, phase, KV/cache state and preemption. |
| JAX and XLA profiling | Capture host and TPU timelines | Attach scenario, analysis, forward and executable correlations. |
| MoE path selection | Select dense/SparseCore and related paths | Record live rows, validity, group sizes, skew and selected path. |

## 6.2 TPU adapter contract

* Report TPU generation and slice, chip/core counts, ICI topology and effective mesh.
* Describe MXU, VPU, DMA, SparseCore and collectives through chip-neutral engine interfaces.
* Report HBM/VMEM capacities, alignment constraints, compiler/runtime/Pallas/Mosaic versions and relevant XLA flags.
* Provide calibrated primitive records with exact shapes, bytes, latency distribution and confidence.
* Normalize TPU trace tracks without dropping engine-specific fields.

## 6.3 Required changes

| ID | Change | Done when |
| :---- | :---- | :---- |
| I1 | Stable logical\_forward\_id across scheduler, runner and profiler | One batch joins across host, compiler and every DP rank. |
| I2 | Actual/padded tokens and requests, phase and DP rank | Heatmap needs no parsing heuristic. |
| I3 | Export configured token and request ladders | Axes exactly match executables. |
| I4 | Fingerprint executables and persist compilation metadata | Runtime hits join uniquely to compiler artifacts. |
| I5 | Record MoE live rows, validity and skew without large host copies | Crossover evidence has bounded overhead. |
| I6 | Capture-integrity counters and event-drop detection | Incomplete traces fail automatically. |

Hot-path instrumentation uses fixed-size scalar metadata, asynchronous buffering and sampling for expensive events. The scenario declares an overhead limit, which is measured against an instrumentation-disabled baseline. Large payload arrays are never copied to host solely for analysis.

# 7 Reliability security and observability

## 7.1 Controls

* Bound telemetry queues and expose drop counters.
* Write capture manifests only after child artifacts are durable.
* Retain raw evidence whenever normalization or mapping may evolve.
* Enforce scenario-defined time, accelerator and storage budgets.
* Clean up profiler sessions without deleting completed evidence.
* Never merge records across scenario IDs, machines or incompatible clocks.

## 7.2 Security and privacy

| Risk | Control |
| :---- | :---- |
| Prompt leakage | Prefer counts, buckets and opaque IDs; protect replay payloads separately. |
| Credential capture | Exclude secrets and headers; store secret references only. |
| Broad source access | Enumerate readable repositories and artifact destinations. |
| Unintended mutation | Both Phase 0 components remain read-only for source and production. |
| Cross-customer contamination | Namespace and authorize artifacts by tenant and scenario. |
| Unsupported claim | Require evidence, confidence and integrity status for every bottleneck. |

## 7.3 Metrics

| Area | Metrics |
| :---- | :---- |
| Scenario Director | Validation success, failure class, freeze latency, declared/detected mismatches and deduplication |
| Capture | Drops, bytes, steps, logical forwards, rank coverage, timestamp skew and overhead |
| Workload | Join success, unmapped forwards, reconciliation, DP deduplication and coverage |
| Compiler mapping | Executable/operation mapping precision, ambiguity and missing artifacts |
| Critical path | Wall-time reconciliation error, unexplained time, exposed-time coverage and confidence |
| Replay | Replay success, graph-key match, output match and repeatability |

# 8 Verification plan

## 8.1 Scenario Director

| Level | Cases |
| :---- | :---- |
| Unit | Canonical serialization, IDs, defaults, schema validation, revision resolution, mesh arithmetic and permissions |
| Property | Field order does not change ID; identical inputs deduplicate; any material change changes ID |
| Integration | Probe TPU context, resolve repository/model revisions and compare requested/effective server configuration |
| Failure | Missing SLO, invalid dtype, topology mismatch, dirty unrecorded tree, unsafe path and non-replayable workload |

## 8.2 Serving Analysis Agent

| Level | Cases |
| :---- | :---- |
| Unit | Bucket boundaries, DP deduplication, shares sum to one, zero-hit exclusion and tail retention |
| Join | Missing rank, duplicate forward, ambiguous executable, clock skew, drops and cache aliasing |
| Critical path | Serialized, fully overlapped, partial overlap, collective straggler and nested custom-call cases |
| Replay | Graph-key reproduction, deterministic inputs, bucket coverage and reference equality |
| Adversarial | Truncated JSON, empty xplane, renamed custom call, layout-only graph change and multiple source sites |

## 8.3 Canonical end to end replay

Use Qwen3-30B-A3B on TPU v6e-8 with 1,024 input tokens, 1,024 output tokens, concurrency 512, attention DP8 and MoE expert parallelism. Run basic, sparse\_core\_only, moe\_gather\_only and improvement\_all under one scenario, with three trials per configuration and a fresh server for each configuration.

| Check | Pass condition |
| :---- | :---- |
| Scenario | Commands, commits, topology and flags are complete and immutable. |
| Workload | Each logical forward maps to one graph bucket; shares reconcile to one. |
| Evidence | MoE gather maps from scheduler through graph and trace to source. |
| Isolation | Variants differ only in declared MoE/SparseCore flags. |
| Statistics | Report medians of three valid trials and preserve raw results. |
| Negative evidence | A non-improving SparseCore-only result is retained and claims are revised. |
| Handoff | Kernel Engineering consumes one slice and bottleneck without raw-log interpretation. |

## 8.4 Phase 0 acceptance

* One canonical request produces exactly one frozen scenario and complete validation report.
* One run maps requests and scheduler activity to graph buckets, optimized operations, device events and source.
* The heatmap counts logical forwards under DP and preserves actual counts within padded buckets.
* At least one dominant and one SLO-tail context produce deterministic static records and replay bundles.
* The top bottleneck is ranked by exposed critical-path contribution with alternatives and evidence.
* Corrupt or truncated traces fail integrity and cannot produce high-confidence diagnoses.
* Identical inputs reproduce artifact digests and graph keys.
* Phase 0 performs no source mutation and cannot promote changes.

# 9 Delivery plan

## 9.1 Milestones

| Milestone | Deliverables | Exit |
| :---- | :---- | :---- |
| M0 Contracts | Core schemas and canonical examples | Compatibility tests pass. |
| M1 Scenario Director | Resolvers, validation, hashing, storage and CLI | Canonical TPU scenario freezes and intentional mismatches fail. |
| M2 Instrumentation | Forward IDs, buckets, fingerprints, compiler manifests and profile annotations | Scheduler, compiler and all DP ranks join without name heuristics. |
| M3 Analysis core | Integrity, heatmap, selection, evidence graph, critical path, ranking and replay | Canonical replay yields reconciled workload and source-grounded bottlenecks. |
| M4 Handoff | Stable artifact APIs and orchestrator adapter | Kernel Engineering consumes the output without raw logs. |

## 9.2 Repository structure

tpu\_inference/autoresearch/
  scenario.py
  machine\_context.py
  records.py
  artifact\_store.py
  serving\_analysis/
    agent.py
    collector.py
    integrity.py
    workload.py
    slice\_builder.py
    evidence\_graph.py
    critical\_path.py
    replay.py
  chips/tpu/
    machine\_context.py
    compiler\_adapter.py
    profiler\_adapter.py

tools/autoresearch/
  freeze\_scenario.py
  capture\_serving\_run.py
  analyze\_serving\_run.py
  render\_heatmap.py

tests/autoresearch/
  scenario/
  serving\_analysis/
  integration/
  replay/

## 9.3 Initial backlog

| ID | Task | Dependency |
| :---- | :---- | :---- |
| SD1 | Schemas and canonical serialization | None |
| SD2 | Model, repository and machine resolvers | SD1 |
| SD3 | Validation and typed failures | SD1 SD2 |
| SD4 | Freeze, digest and artifact storage | SD3 |
| SA1 | Forward ID and instrumentation payload | SD1 |
| SA2 | Scheduler and runner instrumentation | SA1 |
| SA3 | Compiler fingerprints and manifests | SA1 |
| SA4 | Capture integrity checks | SA2 SA3 |
| SA5 | Bucket aggregation and DP deduplication | SA2 |
| SA6 | Context selection and static records | SA4 SA5 |
| SA7 | Compiler/profile evidence graph | SA3 SA4 |
| SA8 | Critical-path attribution and ranking | SA7 |
| SA9 | Deterministic replay bundles | SA6 SA7 |
| E2E1 | Four-configuration canonical replay | SD4 SA1-SA9 |

# 10 Open decisions and recommended defaults

| Decision | Recommended MVP default | Configurable |
| :---- | :---- | :---- |
| Objective owner | Caller supplies; Scenario Director freezes | Weights, SLOs and minimum materiality |
| Workload source | Canonical benchmark first; production windows after joins are proven | Benchmark, production or customer replay |
| Context selection | Cover dominant weighted cost, then tails and crossover neighbors | Coverage and slice budget |
| Bottleneck count | Up to five candidates per selected context | Limit and confidence |
| Dynamic features | Observe live rows, validity and skew; do not authorize selection in Phase 0 | Allowed host-visible features |
| Raw evidence | Retain for public claims and unresolved failures | Duration, privacy class and budget |
| Mapping confidence | Explicit correlation IDs for high confidence; label heuristic joins | Downstream threshold |
| Instrumentation overhead | Measure against instrumentation-disabled baseline | Limit and sample rate |

These decisions are policy inputs rather than code constants. Schema and instrumentation work can proceed without embedding one workload or chip into the core contracts.

# Appendix A Canonical scenario example

model: Qwen/Qwen3-30B-A3B
hardware: TPU v6e-8
traffic:
  input\_tokens: 1024
  output\_tokens: 1024
  concurrency: 512
  request\_rate: uncapped
  fixed\_length\_ratio: 1.0
serving:
  attention\_dp: 8
  expert\_parallel: enabled
  max\_num\_batched\_tokens: 1024
  max\_num\_seqs: 128
measurement:
  measured\_requests: 5120
  warmups: 1024
  trials\_per\_configuration: 3
configurations:
  \- basic
  \- sparse\_core\_only
  \- moe\_gather\_only
  \- improvement\_all

# Appendix B Evidence checklist

* Scenario, analysis, logical-forward and DP-rank event IDs.
* Model, quantization, machine, mesh, compiler, runtime and repository revisions.
* Actual/padded token and request counts, phase, KV state and prefix-cache behavior.
* Executable fingerprint, graph key, operation ID, custom-call target and source.
* Engine, dependencies, service/exposed time, bytes, layout and lifetime evidence.
* Hit count/share, cumulative share, workload impact and SLO risk.
* Integrity status, mapping method, confidence, ambiguity and alternatives.
* Replay inputs, reference outputs, commands, seeds and acceptance criteria.

# Appendix C Source anchors

Architecture source: [AutoResearch Kernel Agent for New Chips — Design Doc](https://docs.google.com/document/d/1eDtoOujHeuqTQauXiozcyFd2gp1RyDCcIwws42XBJU0/edit)

TPU token and request bucket construction: [tpu\_inference runner utils](https://github.com/windtara0619/tpu-inference/blob/main/tpu_inference/runner/utils.py)

TPU precompilation shape loops: [tpu\_inference compilation manager](https://github.com/windtara0619/tpu-inference/blob/main/tpu_inference/runner/compilation_manager.py)
