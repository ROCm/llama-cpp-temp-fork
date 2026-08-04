# HRX structural fusion search

The HRX planner separates graph recovery, region selection, and executable
materialization. This is intentional: a normalized GGML graph describes program
semantics, while a kernel recipe describes one physical way to execute a semantic
region. Neither should be inferred from the other's array offsets.

## Layers

`GraphIndex` is the generic, immutable graph substrate. It indexes value consumers,
operation dependencies, storage-version writers, structural keys, region boundaries,
and contraction legality. Mutation edges are first-class dependencies; views do not
make storage ordering disappear.

`FusionProvider` is the domain extension point. A provider discovers facts with
evidence, proposes hero-owned candidates, and offers bounded expansions. It does not
claim operations directly. `search_fusions` owns overlap resolution, deterministic
priority, legality, stale candidate invalidation, coverage, and final region order.

`RoutedTransformerProvider` is the first domain provider. It finds attention and MoE
heroes structurally, reconstructs repeated blocks and semantic roles, and partitions
each block into seven components:

- attention preparation;
- Q/K/V projection and cache publication;
- attention;
- attention output and feed-forward preparation;
- router selection;
- routed gate/up;
- routed down and result publication.

These are composition units, not promises of one dispatch each. A selected recipe
may emit several physical kernels. Conversely, an available recipe may grow across
two units, such as routed-down plus the following normalization. Whole-block recipes
remain representable as ordinary alternatives without adding a special block-op
concept to the generic engine.

`recover_structural_routed_transformer_program` materializes the currently available
Qwen-derived physical recipes. Kernel names retain their provenance, but dimensions,
quantization choices, operands, scratch, and specialization parameters are recovered
from semantic graph roles. `materialize_routed_transformer_dispatch_bindings` is the
role-based ABI adapter; it contains no layer offsets or fixed operation count.

## Recipe availability and costs

`RoutedTransformerRecipeCatalog` gates optimized alternatives. A family is not
selectable merely because the provider knows its shape: its physical implementation
must be present in the catalog. This lets new recipes land independently and compete
with correctness baselines.

Costs must be measured evidence or strict structural dominance. Current baselines
record their actual physical dispatch count. Publication-eliding alternatives record
the dispatches they replace and emit; the planner does not encode model-specific
priority numbers. Applicability and correctness remain hard predicates, never cost
penalties.

## Cutover and diagnostics

Reactive planning now uses structural recovery as the authority for recognized
routed-transformer graphs. On a cold plan, the old positional Qwen implementation is
retained as an independent oracle when it recognizes the same graph. Kernel families,
variants, runtime and compile parameters, ABI bindings, dependencies, scratch graph,
and root contracts must agree. Cached execution consults only the frozen structural
plan.

With `GGML_HRX_DUMP_GRAPH_DIR` set at backend initialization, each plan directory
contains:

- `fusion-search.txt` and `fusion-search.json`: facts, selected candidates,
  economics, queue decisions, and errors;
- `fusion-regions.dot`: the compact selected-region DAG;
- `program.json`, `commands.txt`, and `commands.dot`: physical schedule and command
  materialization;
- `status.txt`: planner identity and whether the legacy oracle agreed.

Environment variables are read when the backend context is created. Fusion search
and steady-state dispatch do not query them.

## Extension discipline

When adding a recipe:

1. Give it a semantic family and explicit applicability facts.
2. Gate it on physical recipe availability.
3. Grow from an existing component or form a union of components; do not parse
   diagnostic keys to recover semantic roles.
4. State reference and planned economics with evidence.
5. Test legality and overlap on a small graph, then compare complete ABI-bound
   schedules on captured graphs.
6. Preserve an actionable rejection reason for unsupported or inconsistent facts.

Corpus tests compare behavior and executable contracts. They deliberately do not pin
capture digests, source revisions, serialized report text, or fixed graph/dispatch
counts as change detectors.
