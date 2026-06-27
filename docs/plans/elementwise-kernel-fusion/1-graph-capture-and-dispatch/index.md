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

> **Decided model lives in `2.md §0`.** The entities below are kept for
> context but reflect the final decisions: no `LazyNode`, no `OpKind`,
> `LazyFunction` in `tensor`, metadata on `TensorImpl`.

## Entities

**TensorImpl is the vertex** — no separate `LazyNode`. `is_lazy()`
(`lazy_ != nullptr`) distinguishes a realized leaf (has `Storage`,
terminates DFS) from an op node (0-byte `Storage`, real metadata, holds a
`shared_ptr<LazyFunction>`).

**LazyFunction** — what an op node carries. A `tensor` type (not inductor),
fusion analogue of the autograd `Function`. Describes "what computation is
pending," not differentiation.

- Fields:
  - `inputs` — `vector<shared_ptr<TensorImpl>>` (≈ `Function::saved_inputs`)
  - op-specific params (e.g. clamp bounds) — like `ClampBackward::min_val_`
- Virtuals: `infer_output()`, `run_eager()`, `codegen_expr()`,
  `is_fusible()`. **No `OpKind`. No output metadata** — that lives on the
  `TensorImpl` the op produces.
- One concrete subclass per op (shape shared on a category base).

**LazyBackend** — the *only* abstract seam, declared in tensor,
exposing **realize** (record is a free fn in tensor, not a backend method).
`Tensor` calls through it and never includes inductor.

**NVRTCInductorBackend** — concrete subclass in inductor implementing
realize for CUDA via NVRTC codegen + module cache. Registered into tensor
at startup. Leaves room for a future `OpenMPInductorBackend` on the same
seam.

- `lazy_function.cpp` (+ per-op subclasses) — in **tensor**.
- `nvrtc/` (or `cuda/`) which contains the `realize()` code — in inductor.
- Singleton: **yes**, `backend()` returns the single registered instance.

## TensorImpl field

Keep both members (no union) — `Storage` is cheap when 0-byte.

```cpp
struct TensorImpl {
  Storage data_;                         // realized bytes
  std::shared_ptr<LazyFunction> lazy_;   // the pending op; null on eager path
  // invariant: exactly one of {data_ populated, lazy_ set}
  bool is_lazy() const { return lazy_ != nullptr; }
};
```

- Storage is either empty (zero bytes) with a device field (CUDA), or it
  is filled with actual data, still with device field

Eager path never touches `lazy_` → one null-check on the hot path.
`LazyFunction` is a `tensor` type, so no cross-module type-erasure is
needed; only the `LazyBackend` is abstract.

## Lazy backend

`record` is a free fn in tensor (no backend); the backend exposes realize.

```cpp
std::shared_ptr<TensorImpl>                  // free fn in tensor/
    record(std::shared_ptr<LazyFunction> fn);

struct LazyBackend {                  // in tensor/
  virtual ~LazyBackend() = default;
  virtual void realize(TensorImpl*) = 0;
};
LazyBackend* backend();               // null unless registered
void register_backend(LazyBackend*);  // backend calls at startup
```

**record** — takes the op's `LazyFunction`, calls `fn->infer_output()`
(computes output metadata now — `type_upcast`, shape — and allocates a
`TensorImpl` with empty `Storage`), sets `lazy_ = fn`. No bytes.

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
  bool defer = backend() && a.device()==CUDA;  // Option A: device-only gate
  if (!defer) return primitives::add(a, b);          // today's eager path
  return Tensor{ record(std::make_shared<AddFn>(
      std::vector{a.impl(), b.impl()})) };
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
