# PPO value-function divergence (loss → 1e8, return climbs then collapses)

A portable write-up of a PPO failure mode found and fixed in **Bascule** and the
**Tensotron showcase `ContinuousPpo`** (both **FIXED** — see the status table in §5). If your
PPO uses an MSE value loss on the raw bootstrapped return with no return/value normalization,
you have this bug latent — it just hasn't bitten yet.

This is a *different* failure from the NaN crash; both are covered below because they look
similar from the outside (training "falls apart") and a robust PPO wants both fixes.

> **Scope — what is and isn't fixed.** This document covers two failures that ARE fixed: the
> value-loss divergence (→ return normalization) and the NaN crash (→ logStd clamp + non-finite
> guard). There is a **third**, separate way a run can still "climb then collapse" that these fixes
> do **not** eliminate: **entropy/σ collapse + critic drift** on long runs (the policy sharpens to a
> near-deterministic σ, importance ratios then explode on any nudge, and the unclipped critic drifts
> on the narrowing state distribution). At a too-high learning rate this can still drag a converged
> policy off its peak — observed as a rising `TotalSkippedUpdates` and a slow return decay (NOT a
> NaN crash; the guard prevents that). It is **mitigated** by saving the best-smoothed checkpoint
> rather than the last and freezing training at the peak (`LearningAgent.Stop()`), and now actively
> fought by two on-by-default levers: a **KL early-stop** (`TargetKl`) that caps how far a single
> update moves the policy, and a **reactive LR backoff** that shrinks the learning rate when the policy
> persistently strains against that trust region (a streak of KL early-stops) or trips the crash guard.
> Measured on the drift-racer pushed well past its peak at an aggressive LR (best-checkpoint disabled),
> the two levers took the guarded-update count from **1861 → 0** and turned a peak-then-collapse
> (~38 → ~11) into a stable plateau that held its ~30 peak. A subtle interaction worth knowing: KL
> early-stop *suppresses* the skip signal, so an LR backoff gated only on skips never fires once KL
> early-stop is on — the backoff must also watch the KL-early-stop streak (see
> [[ppo-value-divergence-crossrepo]]). Remaining levers (a higher σ floor, value-function clipping)
> are future work; a small positive entropy coefficient is already exposed (`EntropyCoef`).

---

## 1. The symptom

- `meanReturn` improves for a good while, then **falls WELL down** and doesn't recover.
- `loss` grows without bound — observed **~1e8** — but stays **finite**, so any
  `IsNaN`/`IsInfinity` guard never fires. The training looks "alive" the whole time.
- **Intermittent across runs even with a fixed seed.** On CUDA, reductions use atomics and
  are not bitwise-deterministic (see `lib/Tensotron/CLAUDE.md` → "Not bitwise-deterministic"),
  so whether/when a given run tips into divergence varies. This is why it feels random.

Distinguishing it from the NaN crash (section 4): here loss is a **huge finite number** and
the policy degrades *gradually*. In the NaN crash loss is literally `NaN`, actions become
`NaN`, and the agent visibly teleports/disappears immediately.

---

## 2. Root cause

The critic is trained to predict the **bootstrapped GAE return**:

```
ret[t] = adv[t] + value[t]          // GAE target = advantage + own current value estimate
valueLoss = (value(s) - ret)^2      // MSE regression onto that target
```

The regression target depends on the critic's **own current output** (that's what
"bootstrap" means). With `gamma = 0.99`, the only thing pinning the *absolute level* of `V`
to reality is the reward signal, and it enters through a `(1 - gamma) ≈ 0.01` term — an
extremely weak anchor. So the absolute value level is almost free to **random-walk**. Once
`V` drifts to a large magnitude, the TD target `ret` follows it up, and `(V - ret)^2` grows
**quadratically** into the 1e8 range.

Then the coupling kills the policy: in a shared-trunk actor-critic the giant value loss
dominates the total loss, and its gradient flows back through the shared layers, dragging the
policy weights. `meanReturn` collapses. (Even with separate heads, polluted/huge advantages
from a bad critic degrade the policy.)

### Why gradient clipping and Huber loss do NOT fix it

This is the trap. People reach for `MaxGradNorm` / `clip_grad_norm` or a Huber value loss and
are surprised it keeps diverging. **Adam renormalizes gradient magnitude per-parameter**
(it divides by a running RMS of the gradient), so a uniformly huge gradient is scaled back to
roughly a unit step anyway. Grad clipping caps the *step size*; Huber caps the *per-sample
loss slope*. **Neither changes the regression TARGET SCALE**, and the target scale is the
actual problem. The critic is being asked to regress onto a number whose natural magnitude is
unbounded.

---

## 3. The fix: value-target normalization (running return RMS)

Make the critic predict a **unit-scale** quantity. Keep a running RMS of returns `σ_ret`; the
value head learns `ret / σ_ret`. Crucially, **scale the predicted value back up by `σ_ret`
everywhere the value is consumed by GAE**, so advantages — and therefore the policy — are
completely unchanged. Only the critic's regression target moves to unit scale, which is
exactly what stops the divergence.

This is the same idea as PopArt / SB3's return normalization, done minimally.

Pseudization (see real code in `src/Bascule.RL/PpoUpdate.cs`, `Ppo.cs`,
`BatchedPpoTrainer.cs`):

```csharp
// running scale, calibrated on the first healthy batch then EMA'd
float _retStd = 1f; bool _retStdInit = false; const float RetStdBeta = 0.1f;

float ReturnRms(float[] ret) {                       // sqrt(mean(ret^2)), double accum
    double s = 0; foreach (var r in ret) s += (double)r * r;
    return (float)Math.Sqrt(s / ret.Length);
}

void UpdateReturnScale(float[] ret) {
    float rms = ReturnRms(ret);
    if (!float.IsFinite(rms) || rms <= 0f) return;   // degenerate/NaN batch keeps prior σ
    _retStd = _retStdInit ? (1 - RetStdBeta) * _retStd + RetStdBeta * rms : rms;  // calibrate-then-EMA
    _retStd = MathF.Max(_retStd, 1e-4f);             // floor
    _retStdInit = true;
}

// --- during rollout: value comes out normalized, scale it UP to raw units for GAE ---
bValues[row]   = predictedValue[e] * _retStd;
finalValues[e] = bootstrapValue[e] * _retStd;        // bootstrap term too

// --- after GAE, before the update ---
ComputeGae(...);                                     // advantages + raw returns
UpdateReturnScale(ret);                              // recalibrate σ_ret from THIS batch's raw returns
NormalizeAdvantages(adv);                            // unchanged — adv never touched _retStd
for (int i = 0; i < ret.Length; i++) ret[i] /= _retStd;  // value target now ~unit scale
// ... value head regresses onto ret/_retStd; loss stays O(1) forever
```

Key invariants that make it correct:

1. **Advantages are never divided by `σ_ret`.** The policy objective is byte-for-byte what it
   was before the fix. Only the value target changed.
2. **The bootstrap/final value is scaled up too** (`* _retStd`), or GAE mixes normalized and
   raw values and corrupts the advantage recurrence.
3. **Calibrate on the first batch, then EMA** (`β = 0.1`). Starting from a hard `1.0` and
   slowly EMA-ing toward the true scale would let the critic diverge during the warm-up before
   σ catches up.
4. **A NaN/degenerate batch keeps the prior σ** (don't let a bad batch zero or NaN the scale).

After this, value loss stayed O(1) for the entire run where it previously hit 1e8; verified by
test `Value_target_normalization_keeps_the_critic_bounded` (asserts `ReturnScale > 5` and
`maxLoss < 25` on cart-pole).

---

## 4. The sibling bug: NaN crash (σ collapse / gradient explosion)

Same project, separate fix, mentioned because any PPO codebase can have it too and it
looks like "training crashed."

- **Symptom:** loss/meanReturn go literally `NaN`; actions become `NaN`; agent teleports to
  NaN coordinates (renders invisible) and the process slows to a crawl (denormal floats).
- **Cause:** the Gaussian policy's `logStd` collapses to `-∞` (σ → 0 → divide-by-~0 in the
  log-prob → ±∞) **or** a gradient spike writes `NaN`/`∞` into the weights, which then poisons
  every subsequent forward pass.
- **Fix — three cheap layers:**
  1. **Clamp `logStd` to a band** (e.g. `[-5, 2]`) everywhere it's exponentiated — both at
     sample time and inside the log-prob/entropy computation. Stops σ collapse and σ blow-up.
  2. **Non-finite update guard:** compute the scalar loss, and if it isn't finite, **skip the
     whole minibatch** (grads stay zeroed, weights stay intact) instead of back-propagating
     NaN. After backward, also require a **finite global grad-norm** before `opt.Step()`.
  3. **Skip telemetry:** count guarded minibatches (`TotalSkippedUpdates`) and surface it, so a
     "recovered" run is distinguishable from a healthy one.

See `ActionLayout.ClampLogStd` and `PpoUpdate.UpdateMinibatch(..., out bool stepped)`.

---

## 5. How to check another codebase

Grep for the value loss and ask two questions:

```
# the regression target — anything like this is the suspect line
(value - ret).Square()          # C# / Tensotron
(value - returns) ** 2          # python
F.mse_loss(value, returns)      # python
```

1. **Is the value loss MSE on the raw bootstrapped return?** If yes → candidate.
2. **Is there ANY return/value normalization?** (running mean/std of returns, `VecNormalize`,
   `RewardNormalizer`, PopArt, or dividing the value target by a running scale.) If **no** →
   it has this bug latent. It will surface as soon as a run random-walks far enough, which on
   non-deterministic GPU reductions is "eventually."

Confirmed status at time of writing:

| Codebase | Location | Status |
|---|---|---|
| Bascule | `src/Bascule.RL/{Ppo,BatchedPpoTrainer,PpoUpdate}.cs` | **FIXED** (value-target norm + NaN guards) |
| Tensotron showcase | `lib/Tensotron/showcase/Tensotron.Showcase/Rl/ContinuousPpo.cs` | **FIXED** (value-target norm + NaN guards) — running return-RMS normalization (`ReturnScale`), `logStd` clamped to [-5,2] at sample + update, non-finite-loss skip + finite-grad-norm gate (`LastSkippedUpdates`). Regression test `ShowcaseSmokeTests.Ppo_ValueTargetNormalization_KeepsCriticBounded` (σ_ret calibrates ~10→40, total loss stays <0.3, 0 skipped). |

A quick health metric to log while you decide: track **value loss** (or explained variance)
each iteration. A correct critic's value loss has a *fixed scale* and is flat-ish; a value
loss that trends upward over training with no ceiling is this bug in progress.
