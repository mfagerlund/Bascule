# Changelog

All notable changes to Bascule are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha] - 2026-07-01

### Added
- First public release. In-process PPO training for Godot 4 (.NET) via the Tensotron engine — no
  Python, no socket bridge, no native runtime.
- `Bascule.RL` (published to NuGet) — the Godot-free RL core: batched PPO (`Ppo`,
  `BatchedPpoTrainer`), `ActorCritic`, multi-channel `ControlSpec`, `CompositeAgent`, and byte-exact
  `ModelSerializer`.
- Godot addon (`addons/bascule`): `LearningAgent`, training dock with live loss/reward/episode plots,
  in-game HUD, and `.tres` `ModelResource` save/load.
- Example scenes: cartpole, turret (mixed continuous/discrete), physics arm, drift racer, PuckWorld.
- Training stability: KL early-stop and reactive LR backoff, both on by default.

[Unreleased]: https://github.com/mfagerlund/Bascule/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/mfagerlund/Bascule/releases/tag/v0.1.0-alpha
