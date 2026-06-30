# Tensotron.Godot

**Train game AI inside Godot — in pure C#, no Python, no native runtime.**

A Godot 4 (.NET) integration for [Tensotron](https://github.com/mfagerlund/Tensotron), the
PyTorch-faithful tensor + autograd library for .NET. Drop a few components onto a node, declare
what it can *see*, *do*, and *be rewarded for*, press **Train**, and watch it learn — all in-engine,
in-process, on the same machine you build your game on.

> **Status: planning.** This README is the design/spec. No code yet — it describes what
> `Tensotron.Godot` is meant to contain. The underlying engine ([Tensotron](https://github.com/mfagerlund/Tensotron))
> exists and trains continuous-control policies with PPO (see its pole-cart and
> walker showcases); this project is the thin Godot-facing layer on top.

---

## Why this exists

Every "C# ML for games" story today still trains in **Python**. Unity ML-Agents runs the agent in
C# but trains over a socket to a Python PyTorch process. `godot_rl_agents` is the same shape —
Godot collects experience, Python (Stable-Baselines3 / CleanRL) does the learning.

Tensotron.Godot is the one that doesn't phone home: **collect, train, and infer all happen in your
Godot/.NET process.** No Python install, no ONNX export dance, no native blob. The whole loop is
managed C#, GPU-accelerated through ILGPU when you have a CUDA card and a fast hand-written SIMD CPU
backend when you don't.

That makes the killer workflow possible: **edit a reward, press Train, see the result in the same
editor session.**

---

## The juicy part: things you can teach it

These are the showcase scenes the project will ship. Each is a few components on a node and a reward
function — the *learning arc* is what makes them fun to watch.

### 1. Pole-cart — the 60-second proof
A cart on a rail balancing a pole. The "hello world": it converges so fast you watch it happen live.

- **Sees:** cart position & velocity, pole angle & angular velocity (4 floats).
- **Does:** push left / right (1 continuous control).
- **Reward:** `+1` every physics tick the pole stays up; episode ends when it falls.
- **Arc:** *episode 0* — flails, pole drops in half a second. *~episode 200* — dead steady, cart
  parked under the pole. Good for proving the loop on any machine in under a minute.

### 2. The Turret — leading a moving target
A turret learns to track and hit a target that strafes and juke. The example everyone asks for.

```csharp
// Declare the control surface — the trainer never needs to know it's a gun.
ControlSpec Spec => new(new[] {
    new ControlChannel("YawDelta",   -1, 1),
    new ControlChannel("PitchDelta", -1, 1),
    new ControlChannel("Fire",        0, 1, IsDiscrete: true),
});
```

- **Sees:** target position & velocity relative to the muzzle, current aim vector, distance,
  cooldown-ready flag.
- **Does:** yaw/pitch delta + fire.
- **Reward:** `+1` hit, `−0.1` fire-and-miss, `+0.05` aim got closer, `−0.01`/step (encourage speed).
- **Arc:** *early* — spins wildly and spams fire. *later* — holds fire, **leads** the moving target,
  squeezes the shot at the intercept point. It discovers lead pursuit without being told it exists.

### 3. Homing seeker — proportional navigation, from scratch
A thrust-vectoring projectile chasing an evading target. No guidance law coded — it grows one.

- **Sees:** line-of-sight vector & rate to the target, own velocity, closing speed.
- **Does:** steer thrust (2 continuous controls).
- **Reward:** shaped on closing distance, big bonus on intercept, penalty on time/fuel.
- **Arc:** *early* — overshoots, loops, runs out of fuel. *later* — smooth interception curves that
  look exactly like the proportional-navigation guidance real missiles use. Emergent, not authored.

### 4. The Walker — emergent gait
A 2- or 4-legged ragdoll learns to walk to a goal flag. The most visually satisfying one.

- **Sees:** body orientation & velocity, joint angles, foot-contact flags, goal direction.
- **Does:** joint torques (one continuous control per joint).
- **Reward:** forward progress toward the goal, upright bonus, energy penalty, big fall penalty.
- **Arc:** *episode 0* — collapses in a heap. *mid* — a desperate inchworm shuffle. *late* — a
  committed, balanced gait. Every training run finds a slightly different walk — that's the fun.
  (Tensotron trains exactly this headlessly; here you watch it emerge in the viewport.)

### 5. Racing line — driving from raycasts
A car learns a track from whisker sensors, with no waypoints fed to the policy.

- **Sees:** a fan of raycast distances (its "whiskers"), speed, heading error to track direction.
- **Does:** steer + throttle/brake (2 continuous controls).
- **Reward:** progress along the track centerline, penalty for wall contact, bonus for speed.
- **Arc:** *early* — hugs walls, stalls in corners. *later* — carries speed, **brakes late**, and
  carves a clean racing line through the apex. Train 64 cars at once and watch the pack converge.

### More the same components unlock
Steering/flocking swarms (one shared policy, many bodies), NPC cover-seeking under fire,
aim-assist / trajectory prediction, drone stabilization, procedural creature animation, crane/arm
reaching tasks. **If you can write a `Reward()` for it, you can train it.**

---

## How it works — the core idea

The whole design is *interface-driven discovery*. A node advertises three things; the trainer
composes everything else from them and never needs to know what the thing actually is.

```csharp
public interface IObservationSource { int Size { get; } void Write(Span<float> dst); }
public interface IControlSurface    { ControlSpec Spec { get; } void Apply(ReadOnlySpan<float> action, float dt); }
public interface IRewardSource      { float Reward { get; } bool Done { get; } void ResetEpisode(); }
```

`ControlSpec` is the keystone — it lets a control surface declare its channels and ranges, so a gun,
a car, a leg joint, and a shader parameter are all *just controls* to the optimizer:

```csharp
public sealed record ControlChannel(string Name, float Min, float Max, bool IsDiscrete = false);
public sealed record ControlSpec(ControlChannel[] Channels);
```

**The editor UX is the point:** drop a `LearningAgent` on a node, add an `ObservationSource`, a
`ControlSurface`, and a `RewardSource`, expose their ranges in the inspector, press **Train**. Save
the result as a `ModelResource` (`.tres`), flip to **Inference**, ship it.

**Why it's fast in Godot:** Godot physics is single-threaded fixed-step, so "128 arenas in parallel"
means 128 agents stepping in *one* `_physics_process` tick — not 128 threads. Tensotron.Godot
gathers every arena's observations into a single `[N, obs]` batch and runs **one** policy forward
pass per tick (a single cuBLAS GEMM on GPU, or the row-parallel SIMD matmul on CPU), then scatters
the actions back. That's the shape Tensotron is fastest at.

---

## What's in the box (planned layout)

```
Tensotron.Godot/
├── addons/tensotron/              # the Godot editor plugin (enable in Project Settings -> Plugins)
│   ├── plugin.cfg
│   ├── TensotronPlugin.cs
│   ├── nodes/                     # LearningAgent, TrainingArea, ControlSurface,
│   │                              #   ObservationSource, RewardSource
│   ├── resources/                 # ControlSpec, ObservationSpec, ModelResource (.tres-backed)
│   └── editor/                    # training dock: live loss/reward/episode plots, Save Model
├── src/Tensotron.Rl/              # engine-agnostic RL core — NO Godot dependency:
│   │                              #   PPO trainer, IEnvironment, ControlSpec, rollout buffers.
│   │                              #   Depends only on Tensotron. Extractable as its own package.
├── examples/                      # the showcase scenes above (turret, walker, racer, ...)
├── lib/Tensotron/                 # Tensotron engine — git submodule, pinned to a commit (build dependency)
└── README.md
```

Two deliberate boundaries:

- **Tensotron stays pure.** PPO, environments, and replay buffers are *not* torch, so they live in
  `Tensotron.Rl`, never in Tensotron itself. Tensotron remains "PyTorch-faithful tensors," full stop.
- **The RL core is Godot-free.** `Tensotron.Rl` knows nothing about Godot — the `addons/tensotron`
  layer adapts Godot nodes to its `IEnvironment`. That keeps the RL code reusable (console sims,
  other engines) and testable without an editor.

### Training modes (all three from day one)
- **Direct control** — `rotation += action * speed * dt`. Cleanest learning signal; the MVP default.
- **Physics control** — `body.ApplyTorque(...)`. Harder, more realistic; opt-in once the loop works.
- **Headless / accelerated** — `godot --headless`, many `TrainingArea`s per process, raised
  `PhysicsTicksPerSecond`. For real training runs you don't watch.

---

## Getting started

**Prerequisites:** the .NET build of Godot 4 and the .NET 8 SDK. A CUDA GPU is optional — Tensotron
uses it through ILGPU when present and falls back to its SIMD CPU backend otherwise, so training and
inference run on any machine.

```bash
git clone --recursive https://github.com/mfagerlund/Tensotron.Godot.git
```

`--recursive` pulls the Tensotron engine into `lib/Tensotron`; `Tensotron.Rl` references it by
relative path, so the solution builds with no extra setup. Open the project in Godot, enable the
**Tensotron** plugin under Project Settings -> Plugins, and the training dock loads.

## MVP scope

**In v1:** the runtime training loop (PPO), `ModelResource` save/load, `ControlSpec`/`ObservationSpec`,
direct-control surfaces, the editor dock (loss / reward / episode count / Save), and the turret +
pole-cart example scenes.

**Explicitly not in v1:** distributed training, a Python bridge, a visual graph editor, a large RL
algorithm zoo, automatic curricula. Godot's community likes small composable tools — this stays one.

---

## Relationship to Tensotron

Tensotron.Godot is a **thin wrapper**: all the tensor math, autograd, optimizers, and the
GPU/CPU backends come from [Tensotron](https://github.com/mfagerlund/Tensotron). This repo adds the
RL trainer, the environment abstraction, and the Godot bindings.

Tensotron is wired in as a **git submodule** under `lib/Tensotron`, and `Tensotron.Rl` references it
by relative path — so `git clone --recursive` builds the whole solution on any machine, pinned to an
exact engine commit. When Tensotron ships on nuget.org the reference becomes a versioned
`PackageReference`, and the submodule stays only for working on the engine in place.

Both are **MIT, free, forever.** The Godot layer is the killer demo for the library, not a paywall.

---

## Consulting & support

Free, MIT, and best-effort — open an issue or a discussion and I'll get to it when I can; PRs welcome.
There's no SLA and no obligation, by design.

**Mattias Fagerlund** is available for paid consulting and custom integration work
(`Mattias.Fagerlund@carretera.se`) — though in all honesty, the wrapper is thin enough that most
teams will fit it to their own game with an AI coding assistant like **Claude Code** faster than a
contract would take to sign. That's the intended experience, not an accident: small surface, clear
interfaces, easy to bend. Reach out if you want it done *with* you or done *for* you.

---

## License

MIT.
