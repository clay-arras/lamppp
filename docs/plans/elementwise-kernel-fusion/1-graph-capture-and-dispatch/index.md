# Todo changelog

- All `TensorImpl*` must be shared pointers
- There needs to be a reshape and view ops refactor to make them proper
  `ops::`, so that they can be tracked in the inductor layer
- DECIDED — Option A: capture EVERY op regardless of fusibility. The
  fusible/non-fusible split happens at merge+partition (Part 2), not at
  capture. Non-fusible ops record a boundary node and realize via a
  `run_eager` bridge to their `*_stub()`.

# Graph Capturing + Dispatch — Design Summary

We add another module called "inductor". We break up the existing tensor
module, where we'll depend on `tensorImpl` in the inductor module, but
we'll have `Tensor` reference some abstract inductor implementations
(abstract inductor will live in tensor modulus)

## Entities

**LazyNode** — one vertex in the deferred-computation DAG. Two flavors:
a leaf (wraps an already-realized operand, has `Storage`, terminates
DFS) or an op node (deferred elementwise op pointing at operand nodes).
Abstract base declared in tensor; concrete subclasses defined in
inductor. `Tensor` only holds `shared_ptr<LazyNode>` to the base.

- Fields:
  - Stores `shared_ptr<LazyFunction>` OR a `shared_ptr<TensorImpl>`. One
    of the pointers will be `nullptr`

`@ABSTRACT`
**LazyFunction** — what an op-node carries: `OpKind`, operand node
pointers, and output metadata (shape/dtype/device) computed eagerly at
record time. Fusion analogue of the autograd `Function`; separate lib.
Describes "what computation is pending," not differentiation.

- Fields:
  - Substitution string / codegen info
  - Vector of its children (`std::unique_ptr<variable_list> saved_inputs;`???)
  - Meta info (i.e. other fields, etc. like for clamp)
- There are different implementations of `LazyFunction`

**AbstractInductorBackend** — abstract interface declared in tensor,
exposing record + realize. `Tensor` calls through it and never includes
inductor.

**NVRTCInductorBackend** — concrete subclass in inductor implementing
the interface for CUDA via NVRTC codegen + module cache. Registered into
tensor at startup. Leaves room for a future `OpenMPInductorBackend` on
the same seam.

- `Lazynode.cpp`
- `Lazyfunction.cpp`
- `nvrtc/` (or `cuda/`) which contains the `realize()` code.

- Fields:
  - TODO: Is realization backend a singleton?

## TensorImpl field

TODO: should I use Union? !!!!

```cpp
struct TensorImpl {
  Storage data_;                       // realized bytes
  std::shared_ptr<LazyNode> lazy_;     // deferred graph node (fwd-decl only)
  // invariant: exactly one of {data_ populated, lazy_ set}
  bool is_lazy() const { return lazy_ != nullptr; }
};
```

- Storage is either empty (zero bytes) with a device field (CUDA), or it
  is filled with actual data, still with device field

Eager path never touches `lazy_` → one null-check on the hot path.
`LazyNode`'s virtual dtor + `shared_ptr`'s type-erased deleter let
tensor destroy a concrete inductor node whose definition it can't see.

## Realization backend

```cpp
struct RealizationBackend {                  // in tensor/
  virtual ~RealizationBackend() = default;
  virtual std::shared_ptr<TensorImpl>
      record(OpKind, std::vector<const TensorImpl*> ins) = 0;
  virtual void realize(TensorImpl*) = 0;
};
RealizationBackend* backend();               // null unless registered
void register_backend(RealizationBackend*);  // backend calls at startup
```

**record** — reads operand identity, computes output metadata now
(`type_upcast`, shape), allocates a `TensorImpl` with empty `Storage` +
a fresh `LazyNode` wrapping a `LazyFunction`. No bytes.

**realize** — DFS backward from the node, stop at leaves/non-fusible (=
fusion group), codegen one NVRTC kernel (hash → module cache), launch,
write `Storage` back into the impl(s), clear `lazy_`. Tensors become
ordinary realized leaves.

- Is there ever a case where we realize + delete, and then LATER we call
  `data()` on an EARLIER node? Maybe we should just hardcode an
  assumption that we don't do that.

## Op changes

```cpp
Tensor ops::add(const Tensor& a, const Tensor& b) {
  auto* B = backend();
  bool defer = B && a.device()==CUDA;   // Option A: no op-kind/shape test
  if (!defer) return primitives::add(a, b);          // today's eager path
  return Tensor{ B->record(OpKind::Add, {a.impl(), b.impl()}) };
}
```

Two-arm branch keyed on device + backend-registered. CPU and no-backend
cases always take the eager arm (unchanged). CUDA-with-backend records
instead of executing.

## data() note (out of scope, recorded)

`data()` is itself a breakpoint. All raw-byte access (`to_vector`,
`.item()`, indexing, autograd reading saved tensors) bottoms out in
`data()`, so the realize trigger belongs inside it:

```cpp
T* TensorImpl::data() {
  if (is_lazy()) backend()->realize(this);   // force, then fall through
  return data_.ptr();
}
```

Makes laziness transparent to every host-side consumer and catches the
whole host-read class at one chokepoint. Non-elementwise consumers and
backward still need their own triggers, as they may not funnel through
`data()`.

<https://claude.ai/chat/1f0a2191-3687-46a1-9ce1-8beb61d6f62e>

# Context

Refactoring eager evaluation into capturing a graph.

- When to save versus when to execute eagerly? What are the times where
  we'll be forced to evaluate?
  - Lets define "breakpoints" as times where we HAVE to evaluate. Maybe
    when we call backward? Or maybe ???
- What will the graph representation look like? What are the important
  information to save?

To save the representation, I'm thinking we return some "scaffold"
variable after the function resolves (like a `Tensor` with some flag set
where it's "not resolved"). There's a "trigger", lets assume it's just a
function call for now, that triggers the graph to actually be parsed and
the kernel to generate.

**Graph representation** We need to add another field. If the data field
is empty (lazy fusion evaluation), then we need a pointer to a
`LazyFunction`, which holds two `LazyTensors`? Or something like that.
We have `Tensor` and `LazyTensor`

- Decision: is `lazyFunction` separate from the autograd function?
  Answer: yes, different libs
- Open Decision: should fusion be done in a separate SECTION (from
  tensor, autograd, nets?) – probably? Called "inductor" – so do we have
  to build a completely separate tensor engine than the eager mode?

<https://github.com/clay-arras/lamppp/blob/main/csrc/include/lamppp/tensor/tensor.hpp#L46>

**Barrier definition:**
`.item()`, `.numpy()`, `.tolist()`, host reads
`backwards()` – how will this work with AUTOGRAD?

Intuitively, the logic should live in TENSOR module if we use inductor or not
In inductor, we work off of `TensorImpl`.
How does inductor know when ???

- DO THIS LATER, may show up

1. Static-registration drop. You've already eaten one "linker error fix"
   (commit 1cd2975) around exactly this. If inductor is a separate
   shared lib and nothing in `tensor_core` references its symbols, the
   linker will discard the `register_backend()` static initializer TU.
   You'll need `--whole-archive`/`-u`/an explicit `lmp_inductor_init()`
   call, or to make the registrar part of a TU that's transitively
   referenced. Plan for it now; it's the same fiasco the macro comment
   warns about, but across a `.so` boundary where the trick is weaker.

# Part 1 scoping (Option A)

- Record on `backend && device==CUDA` only — no fusibility gate in the
  op shim.
- Every op (not just elementwise) gets a `LazyFunction` subclass:
  record-time metadata + a `run_eager` bridge to its `*_stub()`.
  Boundary ops realize via `run_eager` in Part 2.
- Split metadata-inference from allocation in `meta_handler.cpp` (ctor
  currently does both); record uses inference-only, eager/realize keeps
  the fused path.
- Shape/view ops (`reshape`/`squeeze`/`expand_dims`) must go through
  `ops::` so they record — today they call `impl_->` directly and share
  Storage (double-realize risk).
- Broadcast/non-contiguous ops stay non-fusible boundaries (flat-indexed
  codegen needs contiguous same-shape).
- `data()` is the realize chokepoint; drop its `const`/`noexcept`. Add
  public `realize()`/`sync()`.
- Lazy `TensorImpl` carries a 0-byte Storage on-device; the ctor size
  `LMP_CHECK` needs a lazy bypass.
