# Bascule.RL

The Godot-free reinforcement-learning core behind [Bascule](https://github.com/mfagerlund/Bascule) —
batched PPO for .NET, built on the [Tensotron](https://github.com/mfagerlund/Tensotron) tensor +
autograd engine. No Python, no socket bridge, no native runtime.

## What it is

`Bascule.RL` is the reusable RL layer that Bascule's Godot addon adapts to game nodes — but it knows
nothing about Godot, so you can drive it from a console sim, a test, or another engine. It provides:

- **Batched, not threaded, PPO** — gather `N` environments' observations into one `[N, obs]` batch,
  run **one** policy forward pass per step (a single GEMM on GPU / one row-parallel SIMD matmul on
  CPU), scatter actions back. One matmul, not `N`.
- **Multi-channel `ControlSpec`** — mixed continuous + discrete action channels described by data, so
  a gun, a car, and a joint are all "just controls" to the optimizer.
- **Two trainer shapes** — `Ppo` owns its own rollout loop over an `IEnvironment` factory (console /
  headless); `BatchedPpoTrainer` is *host-driven* (`Tick`), for a game loop that owns the clock.
- **Stability built in** — GAE, clipped surrogate, KL early-stop, and reactive LR backoff.
- **Model save/load** — byte-exact serializer round-trip for shipping trained policies.

## Quick shape

```csharp
using Bascule.RL;

// 1. Describe what the agent can do — one channel per action, each normalized to [-1, 1].
var controls = new ControlSpec(new[] { new ControlChannel("thrust", -1f, 1f) });

// 2. Implement IEnvironment for your sim (ObservationSize, Controls, Reset, GetState, Step).
//    Then train:
var net = new ActorCritic(stateSize: 4, controls, hidden: 64);
var ppo = new Ppo(net, () => new MyEnv(), new Random(0)) { NumEnvs = 16, Horizon = 256 };
ppo.Train(iterations: 200, (iter, meanReturn) => Console.WriteLine($"{iter}: {meanReturn:F1}"));
```

For a game loop that owns the clock, use `BatchedPpoTrainer` and call `Tick(observations, rewards,
dones, actions)` once per physics tick for all agents at once.

## Status

`0.1.0-alpha`. Continuous / discrete / mixed action heads. Depends on `Tensotron` (float32-only;
CUDA when a GPU is present, otherwise a hand-written SIMD CPU backend — the fast path for shipped,
CPU-only inference).

## License

MIT
