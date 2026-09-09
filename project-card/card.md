---
oneliner: In-process reinforcement learning for Godot 4 — train PPO agents in pure C#, no Python
tags: [reinforcement-learning, ppo, godot, csharp, dotnet, game-ai, tensotron, autograd, drift-racer, puckworld]
stack: [Godot 4 .NET, C#, Tensotron]
generated: 2026-09-09
commit: 279e9c1
placeholder: false
---
Bascule is a Godot 4 (.NET) addon built on Tensotron, a PyTorch-faithful tensor/autograd library
for .NET, that trains PPO agents entirely in-engine: drop observation/control/reward components on
a node, press Train, then flip to Inference. No Python process, socket bridge or ONNX export.
Status is v1, working — the RL core, Godot addon, example scenes (pole-cart, turret, drift racer,
puckworld) and unit-test suite all build and run today.
