# HRX structural fusion search

The HRX graph subsystem turns a GGML compute graph into a reusable executable
program. It deliberately separates three questions:

1. What computation and state transitions does the graph describe?
2. Which semantic regions should an HRX recipe own?
3. How are the selected recipes lowered to kernels, resources, and commands?

The separation is a correctness boundary. GGML operation order, view shape, and
tensor names are observations about one graph construction, not a physical
schedule. Likewise, a kernel name or binding ordinal is not evidence that a
semantic pattern exists. Graph recovery establishes a checked logical program;
fusion search chooses among legal implementations of that program; lowering
materializes the choice.

## Planning pipeline

Planning follows this data flow:

```text
ggml_cgraph
  -> normalized Graph
  -> GraphIndex
  -> domain analysis and logical program
  -> FusionProvider facts and candidates
  -> generic fusion search
  -> selected recipe regions
  -> Schedule with typed bindings
  -> ResourceProgram
  -> CommandProgram resolved against KernelCorpus
  -> frozen executable program
```

`ReactivePlanCache` performs this work on the first execution of a scheduler-
assigned graph UID and caches the resulting immutable plan and runtime tensor
bindings. Every subsequent execution of that UID is a direct lookup: it does
not re-import, normalize, fingerprint, or semantically compare the graph. A
zero UID is a contract error instead of a request for a generic fallback path.
Graph fingerprints remain cold-path provenance and integrity witnesses inside
the normalized graph, schedule, and command program; they are never cache
identities.

Runtime tensor pointers do not enter the normalized graph or immutable plan.
They are retained only in the UID cache entry used to instantiate execution
frames. This keeps plans serializable and captured graphs useful as test
fixtures while making steady-state lookup independent of graph size.

## Normalized graph semantics

`Graph` in `graph-ir.h` is the semantic input to planning. Import assigns stable
IDs to operations, values, and storage objects and records:

- operation inputs, outputs, parameters, and original ordinals;
- value type, shape, strides, byte offset, producer, and view source;
- storage identity, external/weight/mutable-state classification, and size;
- storage versions and explicit read/write effects;
- demanded graph roots.

Views and mutation therefore remain visible after import. Two values may alias
the same storage, and a write creates an ordering constraint even when ordinary
producer/consumer edges do not express it. Fusion code must not reconstruct
these semantics from tensor names or from adjacency in the GGML node array.

`GraphIndex` is the immutable query layer over `Graph`. It provides consumer,
predecessor, successor, storage-writer, and structural-key indices. It also
computes region boundaries and answers the generic legality questions used by
all domains:

- Are all operations and materialized outputs valid?
- Is ownership unique within the proposed region?
- Is the region connected, unless its recipe explicitly permits multiple roots?
- Are all values escaping the region materialized?
- Does contracting the region preserve a valid topological order, including
  storage-version dependencies?

Providers query `GraphIndex`; they do not maintain parallel versions of graph
legality.

## Logical program recovery

A domain analyzer raises normalized operations into a hierarchy that describes
the workload in terms useful to recipe authors. The hierarchy is an internal IR,
not a new GGML operation set and not a declaration of dispatch boundaries.

### RoutedTransformerModel

`RoutedTransformerModel` represents a routed transformer as:

- a program preamble;
- an ordered sequence of transformer blocks;
- typed components inside each block;
- a program endpoint;
- explicit one-operation `Atom` components for operations that were not raised.

Each block records operation and value roles as typed fields and carries facts
such as query/output/KV token counts, hidden width, query and KV widths, expert
count, and route count. A block is decomposed into these composition units:

- `AttentionPrepare`;
- `AttentionQkvPublication`;
- `Attention`;
- `AttentionOutputPrepare`;
- `RouterSelection`;
- `ExpertGateUp`;
- `ExpertDownPublication`.

These units give recipe authors stable semantic landmarks. They do not require
one kernel or one invocation per component. A recipe may implement part of a
component, several adjacent components, multiple roots joined by schedule-only
state, a whole block, or a boundary-spanning region such as result publication
plus the next normalization. Such choices remain ordinary recipe candidates;
the generic search does not need a special "transformer block op."

Recovery is query-and-validate rather than key searching. It starts from
structurally meaningful operations, follows dataflow and storage effects, and
records the evidence for each inferred role and fact. Repeated blocks provide
cross-checks: facts inferred in one location must agree with independently
observed instances. Unknown local structure is not silently folded into a
larger region.

`RoutedTransformerModel::verify` enforces the logical IR contract. In
particular, it verifies canonical component IDs, exact operation ownership,
component boundaries, region legality, repeated-block geometry, attention
query/KV/mask relationships, and router/expert facts. Every normalized
operation has exactly one logical owner. An operation that cannot be assigned a
typed role is owned by an explicit `Atom` component.

Preamble and endpoint regions are dataflow slices between block boundaries and
demanded roots. They are not positional prefixes or suffixes of the operation
array.

## Facts and evidence

`FactDatabase` is shared by providers during search. A fact consists of a key,
typed value, and one or more pieces of evidence identifying where it was
observed. Re-observing the same value adds evidence. Observing a conflicting
value rejects the analysis with `InconsistentFact` and identifies the
implicated graph locations.

Facts answer applicability questions; they are not a substitute for logical IR
roles. Code that needs an attention query, route-ID value, or block component
uses the typed logical program or a typed candidate payload. Human-readable
structural keys and report strings are diagnostic output and must never be
parsed to drive lowering.

## Providers and candidates

`FusionProvider` is the domain extension interface. A provider may:

- discover and validate domain facts;
- seed hero-owned candidate regions;
- expand a popped candidate into a bounded set of alternatives.

A provider does not mutate global ownership or commit a fusion. It describes
legal alternatives to the generic search engine.

A `FusionCandidate` carries everything needed to evaluate and later lower an
alternative:

- provider and recipe-family identity;
- a stable diagnostic key and hero operation;
- logical component provenance;
- the normalized operations it owns;
- values that remain materialized at its boundary;
- an explicit opt-in for disconnected multi-root recipes;
- whether it is a correctness baseline;
- cost evidence and physical dispatch economics;
- a provider-defined typed payload.

The hero is a stable anchor for discovery and expansion, not necessarily the
largest operation or the only root in the final region. Candidate expansion
should remain local, bounded, and deterministic. Providers must not enumerate
an unbounded power set of possible regions.

`RoutedTransformerProvider` exposes the routed-transformer logical program to
this interface. It seeds baseline ownership for every component and atom, then
offers optimized unions or replacements whose recipes are available in its
`RoutedTransformerRecipeCatalog`.

## Generic search

`SearchResult::search` owns policy common to every domain:

1. Ask each provider to discover facts.
2. Validate and enqueue its seeds in deterministic priority order.
3. Pop the best live candidate.
4. Reject stale, overlapping, unsupported, or illegal candidates with an
   explicit reason.
5. Commit a legal candidate and invalidate alternatives that conflict with its
   ownership.
6. Ask the provider for bounded expansions.
7. Verify coverage and topologically order the selected contracted regions.

Correctness baselines make coverage explicit and remain selectable without
claiming an optimization benefit. Optimized recipes compete with those
baselines only when their applicability, implementation availability, and
region legality are established. With complete coverage enabled, any operation
left without an owner is a planning error.

The generic engine sees operations, boundaries, dependencies, candidates, and
costs. It does not know about attention, experts, image convolutions, or any
other workload vocabulary.

## Recipe availability and economics

Logical recognition does not imply physical support. A recipe is offered only
when the provider's recipe catalog says that an emitter and corresponding
kernel implementations are available. Selection must not succeed and defer an
"unknown recipe" failure to command construction.

Candidate economics use one of two kinds of evidence:

- `StructuralDominance` proves that one alternative removes work or
  materialization without adding an incomparable cost.
- `Measured` records a measured time benefit for comparable alternatives.

The accounting records reference and planned dispatch counts, eliminated
materialization bytes, added scratch bytes, and measured time when applicable.
Arbitrary model-specific priority constants do not belong in the search.
Applicability and correctness are hard predicates, never score penalties.

Deterministic tie-breaking is part of the planner contract. Identical graphs,
provider revisions, recipe catalogs, and targets must produce identical
selected regions.

## Lowering selected recipes

`RoutedTransformerProgramProof::recover` connects the logical and physical
layers. It builds and verifies the logical program, runs complete-coverage
search, and lowers each selected candidate into a schedule `Invocation`.

An invocation retains:

- the selected recipe identity;
- every logical component it implements;
- every normalized operation it covers;
- its semantic stage and block ordinal;
- one or more specialized kernel dispatches;
- its dataflow boundary and dispatch dependencies.

Logical-component provenance is required even when a recipe spans components.
This makes it possible to compare search decisions with emitted programs and
prevents the emitter from rediscovering regions by positional convention.

The emitter derives kernel specialization from typed graph facts, operand
types, and semantic roles. Kernel family and variant names are physical recipe
identities; their spelling does not define the recognized model family.

`materialize_dispatch_bindings` is the role-based ABI adapter. It maps logical
values and storage effects to named kernel arguments, recovers compile and
runtime parameters, and adds synthetic scratch values and storage where a
recipe requires them. Binding code must not depend on layer-sized operation
offsets, fixed graph counts, or a particular capture's value IDs.

Schedule verification checks operation coverage, region boundaries, bindings,
dependencies, root disposition, and scratch contracts. A selected recipe with
no registered emitter or incomplete ABI binding is a hard planning error.

## Resources and commands

`build_resource_program` converts the verified schedule into storage-level
contracts. A `ResourceContract` records whether storage is imported, a weight,
mutable state, exported, or elidable; its aliases and final version; and its
first and last invocation. `ResourceUse` records versioned read/write access.
Resource verification checks those declarations against normalized graph
effects and schedule bindings.

`build_command_program` resolves every kernel specialization against an
embedded `KernelCorpus`. Resolution includes the family, variant, catalog ID,
target, and compile parameters. The command program then contains concrete
kernel, fill, copy, and barrier commands; named bindings; dependencies;
constant initialization; and a lifetime-packed transient arena.

The corpus is the executable compilation recipe. A schedule that names a
kernel absent from the corpus, supplies incompatible specialization metadata,
or violates its ABI fails command construction. No best-effort substitution is
performed.

Resource and command planning are separate from fusion search because device
memory placement, transient reuse, transfer commands, and synchronization are
properties of the physical program. They remain traceable back to selected
recipes through invocation and dispatch provenance.

## Fallback contract

Fallback is explicit at both structural levels:

- An unfamiliar operation inside an otherwise recognized routed transformer
  becomes a one-operation `Atom` component and candidate.
- A graph that is not recognized by a domain provider receives an all-atom
  schedule.

Atom plans preserve ownership, dependencies, resource accounting, diagnostics,
and the ability to add support incrementally. They are reported as unoptimized
fallbacks. An atom is executable only when the backend supplies a matching
native eager implementation and corpus entry; a missing implementation is a
hard error, never silent CPU execution or accidental absorption by a nearby
fusion.

This fallback contract allows a provider to recognize useful islands without
weakening correctness. It must not be used to hide failed validation of a
recipe that claimed a larger semantic region.

## Extending the subsystem

### Adding a graph domain

Add a domain-specific logical IR and a `FusionProvider`. The analyzer should
identify heroes structurally, recover typed roles by traversing dataflow and
effects, attach evidence to inferred facts, and verify exact ownership. Keep
domain vocabulary out of `GraphIndex` and the generic search.

Choose logical components that are stable semantic composition units, not a
mirror of one kernel schedule. They should be fine-grained enough for recipes
to form useful unions and strong enough to carry the facts needed by emitters.

### Adding or changing a logical component

Define its typed operation and value roles, boundary contract, and verifier
rules. Recover each required fact locally and validate agreement across
repeated structures. Update textual, JSON, and DOT diagnostics so an ownership
mistake is visible without running the model.

Do not identify components using model names, fixed layer counts, pinned token
sizes, tensor names, operation ordinals, or report-string parsing.

### Adding a recipe

A recipe addition includes all of the following:

1. A stable recipe identity in the provider catalog.
2. Explicit applicability predicates over typed roles and evidenced facts.
3. Candidate seeding or bounded expansion from existing components.
4. Region outputs, connectivity policy, and legality tests.
5. Economics supported by measurement or structural dominance.
6. An emitter that consumes the selected candidate and its typed payload.
7. ABI binding and specialization recovery.
8. A matching kernel corpus compilation recipe.
9. Tests that cover selection, rejection, lowering, bindings, and command
   verification.

Cross-component, cross-block, and whole-block recipes use the same path. They
list the logical components and normalized operations they own and preserve all
escaping values required by unclaimed consumers.

### Evolving heuristics

Heuristics belong either in a domain provider, where semantic knowledge is
required, or in generic search, where the rule applies to every candidate.
Domain modules may encode workload knowledge for language, image, audio, or
other graph families. They should communicate with search through facts,
candidates, costs, and typed payloads rather than special cases in the queue.

When two recipes are incomparable, add measurement or retain both baselines;
do not turn a correctness uncertainty into a score. Raise search limits only
after bounding the provider's expansion shape and adding a test for the
resulting candidate count.

## Diagnostics

Set `GGML_HRX_DUMP_GRAPH_DIR` before creating the backend context. The setting
is captured during context initialization; planning and steady-state dispatch
perform boolean checks and do not query environment variables in hot paths.

The dump separates each planning abstraction:

- `normalized/*.json` records imported graph semantics independently of live
  tensor pointers.
- `logical-program.txt`, `.json`, and `.dot` show recovered facts, blocks,
  components, boundaries, ownership, and atom fallbacks.
- `fusion-search.txt` and `.json` show fact evidence, candidate economics,
  queue decisions, rejection reasons, and selected regions.
- `fusion-regions.dot` shows the contracted DAG of selected candidates.
- `program.json` records recipe invocations, logical provenance, dispatches,
  bindings, and dependencies.
- `semantic-witness.txt` gives a stable semantic summary of graph-to-schedule
  coverage.
- `resources.txt` shows storage classification, versions, aliases, and
  lifetimes.
- `commands.txt`, `.json`, and `.dot` show corpus-resolved commands,
  synchronization, bindings, and transient allocation.
- `status.txt` summarizes planner identity, fallback count, command count, and
  validity.

For a planning failure, inspect the logical program before the search report,
then the schedule, resources, and commands. This order distinguishes a recovery
error from a selection error, an emitter error, an ABI error, and a physical
resource or command error.

The offline dump tools accept serialized normalized graphs, so most planner
work does not require loading or executing a model.

## Testing discipline

Tests should establish contracts at the smallest layer that can express the
failure:

- normalized-graph tests cover aliasing, views, mutation, and storage versions;
- `GraphIndex` tests cover boundaries, disconnected regions, missing
  materialization, contracted cycles, and region ordering;
- search tests cover fact disagreement, priority, overlap, stale candidates,
  bounded expansion, deterministic selection, and complete coverage;
- domain tests cover logical hierarchy, typed roles, fact validation, exact
  ownership, and atom isolation;
- lowering tests compare selected logical-component provenance with emitted
  invocations and verify schedule ABI contracts;
- resource and command tests cover versions, lifetimes, transient reuse,
  corpus resolution, dependencies, and roots;
- captured-graph tests prove end-to-end recovery and executable-plan validity;
- perturbation tests prove that an unfamiliar local operation becomes an
  explicit atom while unaffected optimized regions remain selectable, and that
  inconsistent geometry fails with an actionable diagnostic.

Assertions should target semantic behavior and derived invariants. Tests do
not pin capture digests, source revisions, serialized report text, fixed model
sizes, or total dispatch counts merely as change detectors. Dispatch-count
assertions are appropriate only when a test constructs the recipe alternatives
whose economics it is verifying.

## Source map

- `graph-ir.{h,cpp}` imports and serializes normalized graph semantics.
- `graph-index.{h,cpp}` implements generic graph queries and region legality.
- `fusion-search.{h,cpp}` defines facts, providers, candidates, economics, and
  deterministic selection.
- `routed-transformer.{h,cpp}` defines and verifies the routed-transformer
  logical IR and contributes its fusion candidates.
- `routed-transformer-program.{h,cpp}` lowers selected logical recipes into a
  schedule.
- `routed-transformer-bindings.{h,cpp}` materializes typed kernel ABI bindings
  and scratch.
- `schedule.{h,cpp}` defines and verifies recipe invocations and dispatches.
- `reactive-plan.{h,cpp}` builds resources and caches semantic plans.
- `command-program.{h,cpp}` resolves the kernel corpus and constructs physical
  commands and transient allocation.
- `../kernel-corpus.{h,cpp}` defines the target-specific kernel recipes and
  specialization metadata available to command construction.

The interfaces between these files are intentional. Extensions should add
domain knowledge to recovery, candidates, and emitters without collapsing
semantic analysis, search policy, and physical command construction into one
matcher.
