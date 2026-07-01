# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Bascule is a Godot 4 (.NET) integration for **Tensotron** — a PyTorch-faithful tensor +
autograd library for .NET — that trains game AI (PPO) entirely in-process: no Python, no socket
bridge, no ONNX/native blob. It is the thin Godot-facing RL layer on top of the engine.

## Status: spec stage — no code yet

`README.md` **is the design spec**, not just docs. Read it in full before writing anything — every
architectural decision below is justified there. The repo currently contains only `README.md` and is
not yet a git repo or a .NET solution. When you scaffold, follow the planned layout in the README's
"What's in the box" section; do not invent a different structure without reason.

## The one load-bearing idea: interface-driven discovery

The trainer composes everything from three interfaces a node advertises and **never needs to know
what the thing actually is** — a gun, a car, a leg joint, and a shader param are all "just controls":

```csharp
public interface IObservationSource { int Size { get; } void Write(Span<float> dst); }
public interface IControlSurface    { ControlSpec Spec { get; } void Apply(ReadOnlySpan<float> action, float dt); }
public interface IRewardSource      { float Reward { get; } bool Done { get; } void ResetEpisode(); }

public sealed record ControlChannel(string Name, float Min, float Max, bool IsDiscrete = false);
public sealed record ControlSpec(ControlChannel[] Channels);   // the keystone
```

Editor UX is the product: drop `LearningAgent` + an observation/control/reward source on a node,
press **Train**, save a `ModelResource` (`.tres`), flip to **Inference**, ship. Keep this surface
small and composable — the README is explicit that gold-plating (algorithm zoo, graph editor,
curricula, Python bridge) is **out of scope for v1**.

## Two boundaries that must not be crossed

These are the whole reason the project is split into layers — respect them in every change:

1. **Tensotron stays pure tensors.** PPO, environments, and rollout buffers are *not* torch, so they
   live in `src/Bascule.RL/`, **never** in the Tensotron engine. The engine remains
   "PyTorch-faithful tensors," full stop.
2. **The RL core is Godot-free.** `src/Bascule.RL/` depends only on `Tensotron` and knows nothing
   about Godot. The `addons/bascule/` layer is what adapts Godot nodes to `Bascule.RL`'s
   `IEnvironment`. This keeps the RL code reusable (console sims, other engines) and unit-testable
   without launching an editor. If you find yourself adding `using Godot;` to `Bascule.RL`, stop.

## Why the design is batched, not threaded

Godot physics is **single-threaded, fixed-step**, and the Tensotron engine is **single-threaded
across ops** (calling tensor ops from multiple threads concurrently is unsupported and unsafe). So
"128 arenas in parallel" means 128 agents stepping in *one* `_physics_process` tick — **not** 128
threads. The correct shape, and the one Tensotron is fastest at:

> Gather every `TrainingArea`'s observations into a single `[N, obs]` batch → run **one** policy
> forward pass per tick (one cuBLAS GEMM on GPU / one row-parallel SIMD matmul on CPU) → scatter the
> actions back.

Do not introduce per-agent threads or per-agent forward passes. Batch across arenas.

## The Tensotron engine dependency

The engine is wired in as a git submodule under **`lib/Tensotron`**, referenced by `Bascule.RL`
via a relative-path `ProjectReference`; `git clone --recursive` is what pulls it. Read
**`lib/Tensotron/CLAUDE.md`** before touching anything tensor-related — it is authoritative for the
engine.

**Reference RL code already exists in the engine** and is the blueprint `Bascule.RL` should adapt
(not reinvent) — the README calls the RL core "extractable as its own package," and these are what
gets extracted/generalized:

- `lib/Tensotron/showcase/Tensotron.Showcase/Rl/` — `ContinuousPpo`, `ActorCritic`, `CpuInference`
- `lib/Tensotron/showcase/Tensotron.Showcase/Environments/IEnvironment.cs` — the env contract to
  generalize (showcase version is single continuous action; this project needs the multi-channel
  `ControlSpec` shape above).

Engine constraints that directly shape RL/Godot code:

- **float32 only.** No other dtype exists in the engine.
- **`TensorRuntime` is a process-wide singleton — one backend per process, chosen once.** `Auto`
  (default) = CUDA if a CUDA GPU is present, else the fast hand-written **SIMD CPU backend** (not the
  slow ILGPU CPU accelerator). There is **no per-tensor device / `tensor.to(device)`**.
- **SIMD row-parallel matmul defaults OFF and should stay off per-agent in multi-env RL** (cores get
  saturated by the many agents, and parallelism backfires). Turn it **on** only for a single
  big-batch headless trainer on otherwise-idle cores (`TensorRuntime.CpuMatMulThreads` /
  `TENSOTRON_CPU_THREADS`). The SIMD backend is the *fast* path for small-net batch-1 in-game
  inference — prefer it for shipped Inference mode on CPU-only machines.

## Build / run / test

Prerequisites: the **.NET/mono build of Godot 4.7** (the .NET editor, *not* the standard build) and a
**.NET SDK** that can build `net8.0` (the engine targets `net8.0`; any 8.x+ SDK works). A CUDA GPU is
optional (engine falls back to SIMD CPU). This green-field project has no SDK pin yet; target
`Godot.NET.Sdk/4.7.0` to match the editor.

```bash
# once code exists — there is no solution yet
dotnet build                              # build the solution (use --verbosity quiet, not -q)
dotnet test                               # run Bascule.RL unit tests (Godot-free, no editor needed)
dotnet test --filter "FullyQualifiedName~Ppo"   # scope to a subset

# Godot project — use the .NET/mono build of Godot 4.7; $GODOT = its install dir.
# Use the *_console.exe* binary for --headless so stdout/test output appears on Windows.
"$GODOT/Godot_v4.7-stable_mono_win64.exe" --path . --editor            # open the project; enable the Bascule plugin under Project Settings -> Plugins
"$GODOT/Godot_v4.7-stable_mono_win64_console.exe" --headless --path .  # headless / accelerated training
```

## Three training modes (all from v1)

- **Direct control** — `rotation += action * speed * dt`. Cleanest learning signal; the MVP default.
- **Physics control** — `body.ApplyTorque(...)`. Harder/realistic; opt-in once the direct loop works.
- **Headless / accelerated** — `godot --headless`, many `TrainingArea`s per process, raised
  `PhysicsTicksPerSecond`; for real runs you don't watch.

## v1 scope (don't build past it)

**In:** runtime PPO training loop, `ModelResource` save/load, `ControlSpec`/`ObservationSpec`,
direct-control surfaces, the editor dock (live loss/reward/episode plots + Save Model), and the
pole-cart + turret example scenes.

**Out:** distributed training, Python bridge, visual graph editor, RL algorithm zoo, automatic
curricula.
