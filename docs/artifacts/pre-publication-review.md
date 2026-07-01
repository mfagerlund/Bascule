# Pre-publication review — Bascule

Four independent reviewers (2× Claude Opus, 2× OpenAI Codex), each with a distinct lens, plus a
ground-truth `dotnet test` run (22/22 green). Every finding below was **verified against the actual
code** before being accepted; reviewer claims that didn't hold up are listed under "Rejected / downgraded".

## Methodology note — why cross-checking mattered

Codex B's scariest finding was a MAJOR "continuous actions are clipped before log-prob, so PPO scores
them wrong." The Opus RL reviewer independently traced the same path and found the rollout
(`ActionLayout.Sample`) and the update (`ActionLayout.LogpAndEntropy`) compute the Gaussian density on
the **same stored clamped action** — they're mutually consistent, so the importance ratio and gradient
are well-defined. It's a minor statistical bias (the known "clip-then-score" issue in many PPO impls),
**not a gradient bug**. Rewriting it to a tanh-squashed Gaussian would risk destabilizing a green,
tested trainer right before launch. **Downgraded to known-minor, not changed.**

The Opus RL reviewer also positively verified the algorithm core: GAE indexing (`row = t*E+e`), clipped
surrogate signs, the two-layer NaN crash guard, return-RMS normalization (both directions),
`Ppo` ≡ `BatchedPpoTrainer` semantics, mixed continuous/discrete heads, and byte-exact serializer
round-trip. **No wrong-math BLOCKER exists.**

---

## BLOCKERS

| # | Finding | Status |
|---|---------|--------|
| B1 | No top-level `LICENSE` file despite README claiming MIT 3× | **FIXED** — added MIT `LICENSE` (© 2026 Mattias Fagerlund) |
| B2 | README said "Status: planning. No code yet" while shipping built code + 22 passing tests | **FIXED** — README rewritten to "v1, working" |
| B3 | `PpoUpdate.cs:95` — `MinibatchSize` (user-exported) ≤ 0 → infinite loop (`start += minibatchSize`) | **FIXED** — guard in `RunUpdateEpochs` + `LearningAgent.ValidateTrainingConfig` |

## MAJOR

| # | Finding | Status |
|---|---------|--------|
| M1 | README advertised **Homing seeker** + **Walker** showcases that don't exist in `examples/` | **FIXED** — relabeled as "not yet in the box"; showcase list now matches the 4 real examples |
| M2 | README described racing as **raycast/whisker**-based; actual obs is centerline progress + look-ahead | **FIXED** — rewritten to match `RaceCar` |
| M3 | `ObservationSpec` referenced in README (layout + v1 scope) but **no such type exists** | **FIXED** — removed from README |
| M4 | "What's in the box" layout stale (`nodes/`, `editor/` subfolders; omits `physics`; lists `walker`) | **FIXED** — regenerated from actual tree |
| M5 | README "press **Train**" / "flip to **Inference**" implies dock buttons that don't exist (dock only has **Save Model**; real flow is `Mode` enum + **Play**) | **FIXED** — README now describes the real workflow |
| M6 | Prerequisites said "Godot 4"; addon needs **4.7** (`EditorDock` API + SDK pin `4.7.0`) | **FIXED** — README states 4.7+ |
| M7 | Training collapse undisclosed where users see it (`docs` artifact framed "FIXED"; README "~ep 200 dead steady" no caveat) | **FIXED** — README "Training stability" section + scope note in the divergence artifact |
| M8 | `LearningAgent` accepts **partial arenas** (missing obs/control/reward) → later crash or silent no-learn | **FIXED** — `DiscoverAgents` now requires all three roles + ≥1 action channel |
| M9 | Homogeneity check compared only obs/action **sizes**, not the actual `ControlSpec` (discrete vs continuous, order) | **FIXED** — `ControlSpec.ChannelsMatch` + full compare in `ValidateHomogeneous` |
| M10 | Inference compat checked only sizes — a continuous model could drive a same-width discrete surface | **FIXED** — full `ControlSpec` compare in `SetupInference` |
| M11 | Empty `ControlSpec` → `entropyVar!.Mean()` NRE deep in the update | **FIXED** — zero-channel guard in `ActionLayout` ctor |
| M12 | `examples/racing/` is **untracked in git** — a fresh clone wouldn't get the racing showcase | **PENDING — needs `git add`** (not committing without your go-ahead) |
| M13 | `ModelSerializer.Load` trusts blob counts/dims; truncated/corrupt `.tres` → cryptic exception/huge alloc | **DEFERRED** — round-trip + magic/version/shape checks already verified present; low threat (you load your own models). Recommend wrapping load in a clear `InvalidOperationException` + bounding counts. |
| M14 | Non-finite reward/observation from a user node flows into buffers before the guard runs | **DEFERRED** — NaN-reward recovery is already tested; a fail-fast finite-check at the Godot boundary is the right add, but per-step cost + harshness needs a design call. |

## MINOR / NIT

- `ActionLayout` discrete category count silently rounded non-integral spans (0..1.4 → 2 cats). **FIXED** — `Categories` now rejects fractional/non-finite spans.
- `.gitmodules` submodule URL embedded a personal username (`https://mfagerlund@github.com/...`) — forces it on every cloner. **FIXED** — cleaned to `https://github.com/mfagerlund/Tensotron.git`.
- `examples/physics` (arm, the physics-control demo) existed but was undocumented. **FIXED** — added to README as showcase #3.
- README "expose ranges in the inspector" — ranges are declared in the `Spec` (C#). **FIXED**.
- v1 scope undercounted examples. **FIXED**.
- `TurretDemo.cs:60` assumes child `_Ready` order; empty `_arenas` → misleading 0% HUD if order/AutoStart changes. **DEFERRED** (MINOR) — gather arenas lazily/after `StartRun`.
- `LearningAgent.Mode` defaults to `Idle` (silent no-op on Play). **DEFERRED** (NIT) — a `GD.Print` hint would close the loop.
- Telemetry: episodes straddling a rollout-segment boundary undercount mean-return (`BeginSegment` clears `_running`). **DEFERRED** (NIT) — plots only, buffers unaffected.

---

## Training stability — IMPLEMENTED + validated (2026-06-30)

The Opus RL reviewer's headline: the core is correct, but several defaults made this PPO **more
collapse-prone than textbook** — the peak-then-collapse you saw (`skipped` climbing to 1808 by
iter ~200). Two of the cures are now built in and validated with an actual A/B run:

1. **KL early-stop** (`TargetKl`, default 0.02) — **DONE.** Epoch loop bails the instant a minibatch's
   approx-KL `(oldLogp − logp).mean()` exceeds `1.5 × TargetKl`. In `PpoUpdate.RunUpdateEpochs`, threaded
   through both trainers. **Validated:** at an aggressive LR pushed to 130 iters (best-checkpoint off),
   took guarded updates from **1861 → 0**.
2. **Reactive LR backoff** (`LrBackoffOnInstability`, default on) — **DONE.** Halves the LR (floored at
   `LrBackoffMinLr`) on a crash-guard skip (catastrophic) *or* `LrBackoffKlStreak` consecutive KL
   early-stops (slow drift). **Key finding from validation:** KL early-stop *suppresses* the skip
   signal, so a skip-only trigger never fires once KL early-stop is on — the policy then slow-drifts
   with the LR pinned (run ended at **−2.7**). Reworked to also watch the KL-early-stop streak; the
   re-run **held its ~30 peak** through iter 130 with `skipped=0`, the LR auto-decaying to its floor (a
   soft auto-freeze).
3. **`EntropyCoef`** — already exposed and the racing demo already sets it to 0.005 (non-zero).

Still future work (not blocking publish): **raise the σ floor** `LogStdMin -5 → ~-3`, and
**value-function clipping** (PPO2-style) for the critic-drift channel. Blanket **LR annealing** is
largely subsumed by the reactive backoff above (a demand-driven anneal). Offer stands to add the
remaining two with a validating run.

## Deferred but recommended — test coverage

- Deterministic `ComputeGae` fixture (2 envs × 3 steps, terminal at mid + final, hand-computed adv/ret).
- `Ppo` vs `BatchedPpoTrainer` rollout-equivalence test (fixed RNG, sampling off).
- Serializer: truncated/negative/huge counts, duplicate tensor name, mixed-head round-trip with discrete not in slot 0.
- Categorical sample/log-prob/entropy parity against a fixed-logits fixture.

---

## Needs your decision (judgment, not bugs)

- **`CLAUDE.md` in the public repo** — it's AI-assistant guidance and currently says "spec stage — no
  code yet" (stale). Keep + update, or strip before publish?
- **Consulting pitch + email in README** (Credits/Consulting sections) — kept as-is; looks intentional.
- **`docs/proposals/tensotron-scoped-backends.md`** — internal design proposal; keep in public docs or
  move under a clearly-labeled `design-notes/`?
- **CONTRIBUTING.md + a training gif** — both missing; real adoption wins for a "watch it learn" product.

## Verified clean (no action)

- No tracked `.godot/`, `bin/`, `obj/`, secrets, or `Evolvatron`/internal-path leaks in source/docs (`.gitignore` covers the generated dirs).
- Layering boundaries honored (`Bascule.RL` has no Godot reference).
- Example `.tscn` ext_resources all resolve; `dotnet build` clean (0/0); submodule present + pinned.
