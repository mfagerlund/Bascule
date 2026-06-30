# Proposal: scoped backends for Tensotron (retire the hand-written CPU inference mirror)

**Status:** **deferred — scoped backends don't address the consumer's real constraint** (see Verdict
below, 2026-06-30). Targets the **Tensotron engine** (sibling repo), not this repo. The body below is
the original proposal; the Verdict is the authoritative current decision.

**Author context:** raised while building `Tensotron.Rl` (in-process PPO for Godot). The RL core
ports the engine showcase's `CpuMlp`/`CpuActorCritic` — a hand-written scalar-C# forward pass that
duplicates a network already defined as Tensotron `Linear`/`Sequential` modules. This proposal is
about removing that duplication at the engine level.

---

## Verdict (2026-06-30): deferred — scoped backends don't address the consumer's real constraint

A measured investigation settles this **against** the change for now. Keep `CpuMlp`/`CpuActorCritic`;
leave the engine's singleton alone. The body below is the original proposal.

### The numbers reframe the premise

The proposal assumed the real-module forward is too heavy for batch-1. It isn't. Batch-1 forward,
4→32→32→1 tanh net, ~10⁵ calls:

| path | µs/call | vs scalar |
|---|---|---|
| GPU `Sequential.Forward` (CUDA/ILGPU) | ~200 | 73× |
| SIMD `Sequential.Forward` (NoGrad) | ~3.6 | ~2× |
| hand-rolled scalar `CpuMlp` | ~1.75 | 1× |

The SIMD real module runs within ~2× of the hand-roll and ~55× faster than the GPU path, so
**Open question 1 resolves yes: piece (3) is unnecessary.** If scoped backends are ever built they are
pieces (1)+(2) only.

### The consumer's governing constraint is concurrency, not backend-switching

The first real consumer is `Evolvatron.Walker` (design a body → train it → watch it), and
"train on GPU, infer on CPU, interleaved, single-agent" **is the product loop**, not a demo. But its
rollout constraint is not the one this proposal targets:

- Walker's rollout is **thread-per-env**: `Parallel.For` over ~16 envs, each thread a private-scratch
  `CpuActorCritic.CloneShared()` forward — measured **11.5×** on 16 envs / 32 cores.
- The NN forward is **~56% of rollout compute** (73 ms forward vs 130 ms serial inner work) — the
  larger half, not a footnote.
- That parallelism requires the forward to run **outside the runtime**. The engine is single-threaded
  across ops (shared mutable state: allocator pool, `_intCache`), so the real module **cannot** be
  called from 16 concurrent threads. `CloneShared` (private scratch, shared read-only weights) is what
  makes the parallel rollout safe — `CpuMlp` is load-bearing here, not legacy.

A backend *scope* still routes to that one shared-state runtime, so it would force **serializing** the
rollout forward — a regression on the larger half of rollout cost. Walker's M0 also trains on CPU
(GPU deferred for tiny nets), so there is no mid-training backend split to bridge yet. Scoped backends
solve neither the concurrency need nor the open-architecture need.

### Piece (1) is also the highest-blast-radius change available

It overturns a documented hard constraint ("`TensorRuntime` is a process-wide singleton, one backend
per process"), rewrites the `Instance` every op dispatches through, turns storage-type into a de-facto
device tag, and needs the cross-backend operand rule — to remove a forward that is currently
load-bearing for the parallel rollout. Upside-down at this scope.

### Retiring the hand-roll, when it bites — neither lever is scoped backends

`CpuMlp` is real-but-narrow: it handles any width/depth but hardcodes tanh and an MLP shape, and can
drift from `Linear` semantics. When the first non-tanh activation or non-MLP creature arrives:

- **Option A — keep `CpuMlp`, close its gaps.** Carry a per-layer activation (torch-named:
  tanh/relu/softsign/identity) instead of hardcoded tanh, and add a parity test asserting
  `CpuActorCritic.Forward` matches `ActorCritic.PolicyMean` so it can't drift. Keeps the 11.5×, zero
  engine change. Right while creatures stay MLP-shaped.
- **Option B — batched two-phase rollout through the real module.** Per tick: (1) gather live envs'
  obs → one `[N, obs]` `Sequential.Forward` (real module, NoGrad, single runtime call, any
  architecture) with `CpuMatMulThreads` on (the engine's "turn it on for one big-batch trainer" case);
  (2) `Parallel.For` envs to sample actions + step physics. Uses the real module → generalizes to any
  layer type, no drift. Right when architectures go structurally open (non-MLP). Cost: done/reset
  masking + a rollout-loop rewrite. This is the "batched-device rollout" alternative below — for this
  consumer it is the right lever.

### Preferred shape if scoped backends are revisited

Smaller than the ambient-scope design below:

- **The singleton stays a singleton, but its active backend is mutable** — flipped explicitly at phase
  boundaries (CPU for rollout, GPU for the update). Both backend runtimes coexist; one is current.
- **A tensor used under the wrong backend crashes** — no silent cross-backend copy, no on-demand
  host-copy. Severe but simple, and no worse than today's single-backend reality. Getting params onto
  the inference backend stays an explicit `MirrorTo`/snapshot; accidental cross-use fails loudly.
- **Alternating only.** A global flip holds one value at a time, so it serves rollout/update
  alternation, not concurrent train-and-infer on two threads. Concurrency (a live preview inferring
  while the GPU trains) needs a thread-static active backend plus verified two-runtime safety, and is
  itself gated on whether GPU training starves the render thread — a separate, larger change.
- **The hand-roll coexists.** This capability is additive; it does not require removing `CpuMlp`.

### Revisit trigger

Training moves to GPU (nets grow) **and** the batched CPU rollout is wanted in the same process — and
even then, keeping both phases on one backend (batch the rollout on whichever backend training uses)
may still beat overturning the singleton. Until then, deferred.

---

## Problem

`TensorRuntime` is a **process-wide singleton: one backend, chosen once** (`Auto` → CUDA if a CUDA
GPU is present, else the SIMD CPU backend). There is no per-tensor device and no `tensor.to(device)`.

That model is great for the common cases (all-GPU, or all-CPU) and buys real API simplicity — no
`device` parameter anywhere, so an entire class of "tensor on the wrong device" bugs can't exist.

It breaks for **one workload: in-process RL training.**

- **Training** (batched GEMMs + autograd + optimizer) wants the **GPU**.
- **Rollout / in-game inference** is hundreds of **batch-1** forwards, latency-sensitive. On the GPU
  backend each is a kernel launch + `Synchronize` round-trip — the measured bottleneck. It wants the
  **host / CPU**.

Because the backend is global and fixed-once, you cannot use the CPU backend for rollout *while* the
GPU backend holds the training weights. The showcase works around this with `CpuMlp`/`CpuActorCritic`:
a plain scalar-C# forward that runs **outside the runtime entirely**, fed by a per-iteration weight
snapshot pulled to the host (`ActorCritic.SnapshotCpu`).

### Two root causes (only one is the singleton)

1. **The singleton** prevents running the CPU backend mid-training without abandoning the GPU one.
2. **Framework-forward weight for batch-1.** Even *with* per-tensor devices, an autograd-capable,
   tensor-allocating forward is heavy for a single sample — which is why PyTorch shops still hand-roll
   / TorchScript / ONNX their latency paths. `CpuMlp` dodges this too (no tensor alloc, no autograd
   tape, buffer reuse).

A fix that only addresses (1) lets you run the *real* module on CPU mid-training, but won't fully
retire `CpuMlp` unless (2) is also addressed — i.e. the SIMD backend's batch-1 forward is lean enough.

## Non-goals / what we explicitly do **not** need

- **Not concurrency.** Rollout and update *alternate*; they never run at the same instant. We need
  cheap phase switching, not two backends executing simultaneously. (True overlap would need two
  runtime contexts on two threads — out of scope, and the engine is single-threaded across ops anyway.)
- **Not per-tensor devices.** Reintroducing `tensor.to(device)` / a device tag on every tensor is the
  context-threading we want to avoid, and undoes the singleton's main ergonomic win.
- **Not relevant to shipped inference.** A shipped game does pure CPU inference with the singleton
  pinned to SIMD CPU — already correct. This is a *training-time* concern only.

## Why this is tractable here

The PPO rollout/update boundary is **already a clean `float[]` handoff under `NoGrad`**:

- Rollout emits plain `float[]` experience buffers (`bStates`, `bActions`, `bLogp`, …).
- The update calls `Tensor.FromShaped(buffers)` to build fresh autograd tensors.

So the backend boundary lands exactly where **nothing backprops across it** and no tensor is aliased
between phases. Scoped backends drop in without touching autograd semantics.

## Proposal

Three pieces, smallest-blast-radius first.

### 1. Ambient, swappable backend scope (mirrors the existing `NoGradScope`)

The engine already ships `Tensor.NoGradScope()` — ambient, stack-scoped, no parameter threading. Add
the same shape for backend selection:

```csharp
// default stays Auto (CUDA if present) — unchanged
using (TensorRuntime.On(Backend.Cpu))
{
    // every op on this call stack executes on the SIMD CPU backend
}
// outside the scope: back to the default backend
```

No `device` on any op. No context object passed around. Just an ambient scope, exactly like NoGrad.

### 2. Engine-managed, single-source parameter mirroring

A module keeps (or lazily creates) a resident copy of its parameters on a second backend, refreshed
from the authoritative one — this is what `SnapshotCpu` already does, but **keeping the same module**
instead of producing a hand-written mirror class:

```csharp
ac.MirrorTo(Backend.Cpu);   // pull params to host once per iteration (cheap; small nets)
```

Architecture is defined **once** (`ac`'s `Sequential`s). The engine manages residency. No
`CpuActorCritic`, no re-implemented forward.

### 3. (If needed) a lean batch-1 path on the SIMD backend

Root cause (2): running a real `Sequential.Forward` for batch-1 under `NoGrad` must be allocation-light
to match hand-rolled `CpuMlp`. The CLAUDE.md already claims the SIMD backend is "the fast path for
small-net batch-1 in-game inference" — **so this may already hold and `CpuMlp` is partly legacy.**
Needs measurement (see Open questions). If it doesn't hold, expose an engine-built compiled forward
(`module.CompileInference()` → `delegate(ReadOnlySpan<float>, Span<float>)`) — still single-source,
generated from the same weights/structure.

### Worked example — PPO with scopes (replaces the `CpuMlp` path)

```csharp
var ac = new ActorCritic(obs, controls.Count);   // params on the default (GPU) backend
for (int iter = 0; iter < iterations; iter++)
{
    ac.MirrorTo(Backend.Cpu);                     // one host pull per iteration (today's SnapshotCpu)
    using (TensorRuntime.On(Backend.Cpu))
    using (Tensor.NoGradScope())
    {
        // rollout: run the SAME ac.PolicyMean(...) on the host — batch-1, launch-free
    }
    // update: default (GPU) scope, full autograd, batched minibatches — unchanged
}
```

One network definition; one `MirrorTo` per iteration as the only explicit cross-backend act; zero
device parameters anywhere; `CpuMlp`/`CpuActorCritic` deleted.

## Cross-backend operand rule

An op under scope *B* whose operand is resident only on backend *A* must either host-copy on demand
(cached) or error in a strict mode. For the rollout this **never fires** — every operand (mirrored
params + `float[]` inputs) is CPU-resident inside the scope. The rule only matters for accidental
mixing, where a clear error is preferable to a silent copy.

## Alternatives considered

- **Per-tensor device (`tensor.to(device)`)** — rejected. Reintroduces device threading on every
  tensor and undoes the singleton's ergonomic win. Explicitly the thing to avoid.
- **Status quo (hand-written `CpuMlp`)** — works today, but duplicates the forward (maintenance +
  drift risk vs. the device path) and can't run the real module on CPU mid-training.
- **Batched-device rollout instead** — gather all N arenas into one `[N, obs]` forward per tick → 1
  launch/tick amortized over N (the "one cuBLAS GEMM per tick" the Godot design mandates). This
  attacks the launch+sync cost *without* leaving the GPU and is independently worth doing for the
  headless trainer — but it does **not** help the shipped single-agent CPU inference case, so it
  complements scoped backends rather than replacing them.

## Open questions

1. **Does the SIMD batch-1 forward (under NoGrad) already match `CpuMlp`?** **Answered (2026-06-30):**
   measured ~3.6 µs for the real `Sequential.Forward` on SIMD vs. ~1.75 µs for the scalar `CpuMlp`
   (~2×, on a small tanh-MLP). Piece (3) is unnecessary. But the delta is the lesser point: even at
   parity, a backend *scope* runs through the single-threaded runtime, so it can't be called from the
   consumer's N concurrent rollout threads the way the run-outside-the-runtime `CpuMlp` can. See Verdict.
2. **Mirror staleness ergonomics** — explicit `MirrorTo` (predictable, matches today) vs. lazy
   auto-refresh on first CPU-scope use (fewer footguns, more magic). Lean explicit.
3. **Memory** — GPU params + CPU mirror doubles parameter memory. Negligible for control-policy nets;
   note it for larger models.

## Impact on `Tensotron.Godot`

None required for v1 — the ported `Ppo` works today. When the engine gains scoped backends, delete
`CpuInference.cs` (`CpuMlp`/`CpuActorCritic`) and run the rollout through the real `ActorCritic` under
a CPU scope. Until then, the current code is the correct adaptation of the existing engine pattern.
