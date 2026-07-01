# Contributing to Bascule

Thanks for your interest! Bascule is a small, composable tool — contributions that keep it that way
are very welcome.

## Ground rules — the two boundaries

Bascule is layered on purpose; please respect these:

1. **`src/Bascule.RL` is Godot-free.** The RL core depends only on the Tensotron engine and knows
   nothing about Godot. Godot adapters live in `addons/bascule`. If you find yourself adding
   `using Godot;` to `Bascule.RL`, stop.
2. **Tensotron stays pure tensors.** PPO, environments, and rollout buffers live in `Bascule.RL`,
   never in the engine.

## Building and testing

```bash
git clone --recursive https://github.com/mfagerlund/Bascule.git
cd Bascule
dotnet test          # builds the RL core + runs the Godot-free unit tests (no editor needed)
```

For the Godot side, use the **.NET/mono build of Godot 4.7+**, enable the **Bascule** plugin under
*Project Settings → Plugins*, and open an example scene under `examples/`.

## Pull requests

- Keep the build and tests green (`dotnet test`) and warning-free. CI runs on Linux and Windows.
- Add or update tests for behaviour changes in `Bascule.RL` (the core is unit-tested without an editor).
- Keep changes focused; discuss larger features in an issue first. See the README's **v1 scope** for
  what is intentionally out of scope.
- Add an entry under `## [Unreleased]` in `CHANGELOG.md`.

## Versioning & releases

Versions are tag-driven via [MinVer](https://github.com/adamralph/minver): tagging `v0.1.1` and
pushing it builds and publishes `Bascule.RL 0.1.1` through the trusted-publishing workflow. Don't
hand-edit a `<Version>` in the csproj.

## License

By contributing, you agree that your contributions are licensed under the project's MIT license.
