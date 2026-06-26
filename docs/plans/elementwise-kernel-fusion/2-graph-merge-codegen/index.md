# Graph Merge + Codegen Form — Design Summary

Scope: how a lazy elementwise DAG is partitioned into fusion groups and lowered to one CUDA kernel per group. Excludes laziness triggers, `data()` chokepoint, and backend registration (separate segment).

## Entities

**FusedGraph** — the per-group artifact. One maximal elementwise region → one FusedGraph → one generated kernel. Carries the group's membership (which nodes are inputs vs interior vs outputs), the SSA naming, and the three lowering methods (`topo_sort`, `codegen`, `evaluate`). Not a graph node itself; it's the isolated subcomponent extracted from the larger DAG for processing.
**Boundary** — any node that terminates group growth: non-elementwise op (matmul, reduction), shape-incompatible op, or an already-realized leaf. Unary ops are currently boundaries too (v0 simplicity — flagged as open below). A boundary is not absorbed; it's realized through its own path and enters the group as an input.
**Escaping node** — an interior fusible node whose value is also live outside the group (live Python handle, saved-for-backward, or in-place target). It stays in-kernel as a temp but additionally gets a store → multi-output fusion. Escape never stops fusion.

## Control flow: mutual recursion

`realize(n)` and `visit(n, g)` are mutually recursive. This is the core of the partitioning — neither can be expressed without the other.

```cpp
void realize(TensorImpl* n) {
  if (!n->is_lazy()) return;
  if (is_fusible(n->lazy())) {
    FusedGraph g;
    visit(n, g, /*is_root=*/true);
    g.topo_sort();
    g.evaluate();
  } else {
    for (auto* op : n->lazy()->operands) realize(op);
    run_eager(n);                              // existing dispatch path
  }
}
void visit(TensorImpl* n, FusedGraph& g, bool is_root = false) {
  if (g.seen.count(n)) return;                 // DAG memo
  g.seen.insert(n);
  if (!n->is_lazy()) { g.add_input(n); return; }            // realized leaf → input
  if (!is_fusible(n->lazy()) || !shape_ok(n, g)) {
    realize(n);                                // force boundary FIRST
    g.add_input(n);                            // now realized → input
    return;
  }
  for (auto* op : n->lazy()->operands) visit(op, g);
  g.interior.insert(n);
  if (is_root || escapes(n)) g.outputs.push_back(n);         // root always; escapes → multi-output
}
```

Load-bearing invariant: "all leaves are evaluated when the DFS ends" holds only because `visit` calls `realize` on each non-fusible operand on the way down. Boundaries are materialized during the walk, not after. The whole DAG unwinds bottom-up by demand: each non-fusible op recurses into its operands, which may build their own FusedGraphs.
`escapes(n)` checks liveness signals — no reverse edges needed, since the recorded graph only has operand edges.

## FusedGraph shape

```cpp
struct FusedGraph {
  // ---- membership (filled by visit) ----
  std::vector<const TensorImpl*>        inputs;    // realized leaves, deduped  → in0,in1,...
  std::vector<TensorImpl*>              outputs;   // escaping nodes, ≥1         → out0,out1,...
  std::unordered_set<const TensorImpl*> interior;  // fusible nodes (unordered)
  std::unordered_set<const TensorImpl*> seen;      // DAG memo for visit
  // ---- leaf indexing (P1: TensorImpl* identity → input slot) ----
  std::unordered_map<const TensorImpl*, int> slot; // dedup key → index into `inputs`
                                                   // assigned on first encounter; slot == arg position == source name
  // ---- ordering + naming (filled by topo_sort) ----
  std::vector<const TensorImpl*>                      topo;  // interior in dep order → t0,t1,...
  std::unordered_map<const TensorImpl*, std::string> name;  // identity → SSA var (v*/t*)
  void add_input(const TensorImpl* n) {            // P1 dedup
    auto [it, fresh] = slot.try_emplace(n, (int)inputs.size());
    if (fresh) inputs.push_back(n);
  }
  void        topo_sort();      // STUB -- see below
  std::string codegen() const;  // flat emit over `topo`
  void        evaluate();       // codegen → hash → NVRTC cache → gather ptrs → launch → writeback
};
```

`slot` is the P1 decision made concrete: dedup key is `TensorImpl*` identity, value is the input's positional index. Worst-case false-negative (two views over one buffer) is a redundant load, never a wrong result. The pointer is hashed, never emitted.

- Merge semantics:
  - If the shape matches
  - If the OP is safe (unary or binary ops)
  - If the device matches
- Data type does NOT need to match

## topo_sort stub

```cpp
// STUB -- do not implement yet.
// (1) Orders `interior` so every node's operands precede it, writing into `topo`.
//     post-order `visit` already yields this; the seam stays explicit so the
//     collection strategy can change without touching codegen.
// (2) Assigns `name`: inputs[i] → "v"+i (load var), topo[k] → "t"+k -- in this exact
//     deterministic order. Canonical naming lives HERE, not in visit, so structurally
//     identical graphs produce byte-identical source → module-cache hits.
void FusedGraph::topo_sort() { /* TODO */ }
```

## codegen

Per-op template map, `OpKind` → format string with positional placeholders, one SSA statement per node. Abnormal ops are just different strings — no special-casing:

```
Add → "{0} + {1}"     Mul  → "{0} * {1}"      Relu  → "fmaxf(0.0f, {0})"
Sub → "{0} - {1}"     Exp  → "expf({0})"      Clamp → "fminf({2}, fmaxf({1}, {0}))"
Neg → "-{0}"
```

Emission is a flat walk over `topo` (no recursion, no precedence/paren handling — post-order already linearized it):
(btw, codegen will not be hardcoded float, there will be a mechanism to change per the tensors dtype)

```
params/loads:  for i in inputs   → "const float* __restrict__ in{i}",  "float v{i} = in{i}[idx];"
body:          for n in topo     → "float {name[n]} = " + emit(kind, [name[op] for op in operands]) + ";"
stores:        for o in outputs  → "float* __restrict__ out{o}",       "out{o}[idx] = {name[out]};"
assemble:      extern "C" __global__ + grid-stride loop
```

`extern "C"` gives a stable symbol for `cuModuleGetFunction`; grid-stride loop decouples launch config from n.

## evaluate

`codegen()` → `hash(src)` → NVRTC module cache (get-or-compile) → allocate `Storage` for each output `TensorImpl` → gather device ptrs in `[inputs…, outputs…]` order → launch → clear `lazy_` on outputs. Arg-gather order and param-emit order are both built from the same inputs/outputs vectors, so they cannot drift.

## Division of labor

`visit` decides membership (input/interior/output, dedup via `slot`, DAG memo via `seen`). `topo_sort` decides order + names. `codegen` is a flat emit. Each stage owns one concern.

# Context

- Define "elementwise" ops and "nonelementwise" ops, and ALSO check out binary ops
- We need to fusion group all the fusible kernels, and STOP at "BARRIERS". This is a dfs walk from the root.

We traverse the graph.

Big question:

- What is the codegen going to look like
  - Need to support arbitrary ops.
  - In the graph traversal, we save
  - How to map the tensor name
- In a big graph,

<https://claude.ai/chat/35f94e52-f79a-4584-96b0-030fef6bf32a>

here's the scoping for part 1, capturing the graph:

* I think going through the graph and grouping mergable elements is trivial. the part part: codegen, how to map from tensor names to elements in the codegen, what the codegen will look like, , and how to go from graph walk to codegen.
* lets say we have a graph (all mergable, for simplicity) of the nodes + ops. here's my idea/ algo
  * we count the number of leaf nodes that have data, and they all get an id, which is their variable name in the kernel???
  * we go from top to bottom recursively, and for each op there is a "template" defined, something like this, for multiply: "(static_cast <outtype>" + v + ") - (static cast outytpe<>" + u + ")"
  * we recursively build the "template", and just write a random kernel for them

NEXT PART: spec out how boundaries are held, when we evaluate and generate the graph, and how to isolate a subcomponent of the graph to run processing and generating on.
