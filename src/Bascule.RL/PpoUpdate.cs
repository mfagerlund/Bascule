namespace Bascule.RL;

/// <summary>
/// The PPO algorithm's shared, driver-agnostic core: Gaussian action sampling, GAE advantages,
/// advantage normalization, and the clipped-surrogate minibatch update. It operates purely on a
/// filled rollout buffer laid out as <c>row = t * E + e</c> (timestep-major over envs), so it knows
/// nothing about <em>how</em> the rollout was collected.
///
/// Two drivers fill that buffer differently and then call into here:
/// <list type="bullet">
///   <item><see cref="Ppo"/> — synchronous: it owns the loop and steps each <see cref="IEnvironment"/>.</item>
///   <item><see cref="BatchedPpoTrainer"/> — inverted: Godot owns the frame loop and feeds it one
///   batched tick at a time.</item>
/// </list>
/// Keeping the bug-prone tensor math (the update) and the GAE recurrence in one place is the point —
/// the drivers differ only in collection, never in the algorithm.
/// </summary>
internal static class PpoUpdate
{
    internal static readonly float Log2Pi = MathF.Log(2f * MathF.PI);
    internal static readonly float HalfLog2PiE = 0.5f * MathF.Log(2f * MathF.PI * MathF.E);

    /// <summary>One standard normal sample (Box–Muller). Centralized so every driver draws from
    /// <paramref name="rng"/> identically (two <see cref="Random.NextDouble"/> calls per sample).</summary>
    public static float NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>
    /// Generalized Advantage Estimation over a <c>row = t * E + e</c> buffer. <paramref name="finalValues"/>
    /// is the bootstrap value of each env's state <em>after</em> the last stored step. Writes both the
    /// advantages and the value targets (returns = advantage + value).
    /// </summary>
    public static void ComputeGae(float gamma, float lambda, int T, int E,
        float[] rewards, float[] values, float[] dones, float[] finalValues,
        float[] adv, float[] ret)
    {
        for (int e = 0; e < E; e++)
        {
            float lastGae = 0f;
            for (int t = T - 1; t >= 0; t--)
            {
                int row = t * E + e;
                float nextValue = t == T - 1 ? finalValues[e] : values[(t + 1) * E + e];
                float nextNonTerminal = 1f - dones[row];
                float delta = rewards[row] + gamma * nextValue * nextNonTerminal - values[row];
                lastGae = delta + gamma * lambda * nextNonTerminal * lastGae;
                adv[row] = lastGae;
                ret[row] = lastGae + values[row];
            }
        }
    }

    /// <summary>Root-mean-square of the returns — the running scale a driver normalizes value targets by,
    /// so the value head predicts a unit-scale quantity and can't drift to huge (loss-exploding) values.</summary>
    public static float ReturnRms(float[] ret)
    {
        double sumSq = 0;
        for (int i = 0; i < ret.Length; i++) sumSq += (double)ret[i] * ret[i];
        return (float)Math.Sqrt(sumSq / ret.Length);
    }

    /// <summary>Normalize advantages to zero mean / unit std in place (standard PPO variance control).</summary>
    public static void NormalizeAdvantages(float[] adv)
    {
        float advMean = adv.Average();
        float advStd = MathF.Sqrt(adv.Select(x => (x - advMean) * (x - advMean)).Sum() / adv.Length) + 1e-8f;
        for (int i = 0; i < adv.Length; i++) adv[i] = (adv[i] - advMean) / advStd;
    }

    /// <summary>
    /// Run the full PPO update: <paramref name="epochs"/> passes over the rollout, each reshuffled and
    /// split into minibatches, with a clipped-surrogate + value + entropy step per minibatch. The
    /// rollout arrays are indexed by flat row; advantages must already be normalized.
    /// Returns the mean total loss across every <em>applied</em> minibatch update (the telemetry the
    /// dock/HUD plots), and reports via <paramref name="skippedUpdates"/> how many minibatches were
    /// guarded out — see <see cref="UpdateMinibatch"/>. Returns NaN only if every update was skipped.
    ///
    /// <para><b>KL early-stop.</b> When <paramref name="targetKl"/> is &gt; 0, the update bails out the
    /// instant a minibatch's approximate KL divergence between the rollout policy and the current
    /// policy — <c>(oldLogp − logp).mean()</c> — exceeds <c>1.5 × targetKl</c>. This is the textbook
    /// PPO trust-region guard: it caps how far one update can move the policy <em>before</em> the move
    /// lands, so the importance ratios can't explode into the NaN guard. <paramref name="targetKl"/>
    /// ≤ 0 disables it (legacy behavior — run every epoch). The mean approx-KL over the applied
    /// minibatches is reported via <paramref name="approxKl"/> for telemetry.</para>
    /// </summary>
    public static float RunUpdateEpochs(ActorCritic ac, Adam opt, Random rng,
        int epochs, int minibatchSize, float clipEps, float valueCoef, float entropyCoef, float maxGradNorm,
        int B, int S, int A,
        float[] bStates, float[] bActions, float[] bLogp, float[] adv, float[] ret,
        out int skippedUpdates, float targetKl, out float approxKl, out bool klEarlyStopped)
    {
        // A non-positive minibatch size makes `start += minibatchSize` never advance — an infinite loop.
        if (minibatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(minibatchSize), minibatchSize, "minibatchSize must be > 0.");
        if (epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(epochs), epochs, "epochs must be > 0.");
        klEarlyStopped = false;
        var idx = Enumerable.Range(0, B).ToArray();
        float lossSum = 0f;
        int lossCount = 0;
        int skipped = 0;
        float klSum = 0f;
        int klCount = 0;
        float klLimit = targetKl > 0f ? 1.5f * targetKl : 0f;   // 0 = disabled
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(rng, idx);
            for (int start = 0; start < B; start += minibatchSize)
            {
                int mb = Math.Min(minibatchSize, B - start);
                var mbStates = new float[mb * S];
                var mbActions = new float[mb * A];
                var mbLogp = new float[mb];
                var mbAdv = new float[mb];
                var mbRet = new float[mb];
                for (int j = 0; j < mb; j++)
                {
                    int r = idx[start + j];
                    Array.Copy(bStates, r * S, mbStates, j * S, S);
                    Array.Copy(bActions, r * A, mbActions, j * A, A);
                    mbLogp[j] = bLogp[r];
                    mbAdv[j] = adv[r];
                    mbRet[j] = ret[r];
                }
                float loss = UpdateMinibatch(ac, opt, clipEps, valueCoef, entropyCoef, maxGradNorm,
                    mb, S, A, mbStates, mbActions, mbLogp, mbAdv, mbRet, out bool stepped, out float mbKl);
                if (stepped && float.IsFinite(loss)) { lossSum += loss; lossCount++; }
                else skipped++;
                if (float.IsFinite(mbKl)) { klSum += mbKl; klCount++; }

                // Trust-region early-stop: one minibatch moving the policy too far ends the whole update.
                if (klLimit > 0f && float.IsFinite(mbKl) && mbKl > klLimit)
                {
                    klEarlyStopped = true;
                    approxKl = klCount > 0 ? klSum / klCount : float.NaN;
                    skippedUpdates = skipped;
                    return lossCount > 0 ? lossSum / lossCount : float.NaN;
                }
            }
        }
        approxKl = klCount > 0 ? klSum / klCount : float.NaN;
        skippedUpdates = skipped;
        return lossCount > 0 ? lossSum / lossCount : float.NaN;
    }

    /// <summary>
    /// One clipped-surrogate + value + entropy step. Returns the scalar total loss it computed and sets
    /// <paramref name="stepped"/> to whether the optimizer actually stepped. This is the crash guard:
    /// a non-finite loss is never back-propagated, and a step is applied only when the (post-clip) global
    /// gradient norm is finite — so a single diverged/NaN minibatch can never write NaN into the weights.
    /// <paramref name="approxKl"/> receives this minibatch's mean <c>(oldLogp − logp)</c>, the trust-region
    /// signal the caller's KL early-stop watches (NaN if the loss was guarded out, since then the policy
    /// did not move).
    /// </summary>
    private static float UpdateMinibatch(ActorCritic ac, Adam opt,
        float clipEps, float valueCoef, float entropyCoef, float maxGradNorm,
        int mb, int S, int A,
        float[] states, float[] actions, float[] oldLogp, float[] adv, float[] ret,
        out bool stepped, out float approxKl)
    {
        var st = Tensor.FromShaped(states, new[] { mb, S });
        var oldLogpT = Tensor.FromShaped(oldLogp, new[] { mb });
        var advT = Tensor.FromShaped(adv, new[] { mb });
        var retT = Tensor.FromShaped(ret, new[] { mb });

        // Mixed continuous (Gaussian) / discrete (categorical) log-prob + entropy, per the layout.
        var policyOut = ac.PolicyOutput(st);                 // (mb, PolicyOutSize)
        var (logp, entropy) = ac.Layout.LogpAndEntropy(policyOut, actions, mb, ac.LogStd);

        var ratio = (logp - oldLogpT).Exp();                 // (mb,)
        var surr1 = ratio * advT;
        var surr2 = ratio.Clamp(1f - clipEps, 1f + clipEps) * advT;
        var policyLoss = TensorOps.Minimum(surr1, surr2).Mean().Neg();

        var value = ac.Value(st);                            // (mb,)
        var valueLoss = (value - retT).Square().Mean();

        // approx-KL(old‖new) ≈ mean(oldLogp − logp); read off the same graph before backward.
        var loss = policyLoss + valueCoef * valueLoss - entropyCoef * entropy;
        float lossValue = loss.ToArray()[0];
        approxKl = (oldLogpT - logp).Mean().ToArray()[0];

        opt.ZeroGrad();
        if (!float.IsFinite(lossValue))
        {
            // A non-finite loss would back-propagate NaN/∞ into every weight. Skip the whole minibatch;
            // grads stay zeroed and the next minibatch (or iteration) trains from intact weights.
            stepped = false;
            approxKl = float.NaN;
            return lossValue;
        }

        loss.Backward();
        // Pull the pre-clip global grad norm: ∞ grads were already scaled toward 0 by the clip, but a
        // NaN norm means NaN grads — applying either-and-especially-NaN would corrupt the weights.
        float gradNorm = GradUtils.ClipGradNorm(ac.Parameters(), maxGradNorm, returnTotalNorm: true);
        stepped = float.IsFinite(gradNorm);
        if (stepped) opt.Step();
        return lossValue;
    }

    private static void Shuffle(Random rng, int[] a)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
