namespace Tensotron.Rl;

/// <summary>
/// Maps a <see cref="ControlSpec"/> onto a policy head and centralizes every place the two action
/// kinds are sampled, scored, and made greedy — the one spot that knows the difference between a
/// continuous channel and a discrete one.
///
/// <para><b>Continuous</b> channels are Gaussian (one mean column in the policy output, a shared
/// state-independent log-σ), exactly as before. <b>Discrete</b> channels are categorical: a channel
/// with <c>K</c> options occupies <c>K</c> logit columns in the policy output, is sampled from the
/// softmax, and contributes the chosen category's log-probability to the action log-prob. A discrete
/// channel's option count is <c>round(Max − Min) + 1</c>, and its value in the env-facing action
/// vector is the 0-based <em>category index</em> (a gun's Fire channel with Min=0/Max=1 yields 0 or 1).</para>
///
/// <para>The env-facing action vector stays one float per channel (<see cref="ChannelCount"/>), so
/// <see cref="IControlSurface.Apply"/>, <see cref="IEnvironment.Step"/>, and the rollout buffer layout
/// are unchanged; only a discrete channel's float now means "which option" instead of "how much".
/// The policy network is wider than the action vector when discrete channels are present
/// (<see cref="PolicyOutSize"/> ≥ <see cref="ChannelCount"/>).</para>
///
/// Godot-free: pure layout + tensor/host math, no engine-of-record knowledge beyond Tensotron tensors.
/// </summary>
public sealed class ActionLayout
{
    public ControlSpec Controls { get; }

    /// <summary>Safe band for the Gaussian log-σ. Keeps σ ∈ [e^-5, e^2] ≈ [0.0067, 7.4], so the
    /// log-prob never divides by a near-zero σ (→ ±∞) nor explodes it — the two main numeric blow-up
    /// paths in continuous PPO. Clamping at point of use also zeroes the gradient outside the band, so
    /// the parameter self-limits instead of drifting away. Applied identically in sampling and scoring.</summary>
    public const float LogStdMin = -5f;
    public const float LogStdMax = 2f;

    /// <summary>Env-facing action-vector length: one float per channel.</summary>
    public int ChannelCount => Controls.Count;

    /// <summary>Width of the policy network's output: 1 column per continuous channel, K per discrete.</summary>
    public int PolicyOutSize { get; }

    /// <summary>Number of continuous (Gaussian) channels — equals the length of <c>LogStd</c>.</summary>
    public int ContinuousCount { get; }

    /// <summary>True when every channel is continuous (the cart-pole case): the policy output <em>is</em>
    /// the mean vector, so the update keeps the original direct Gaussian math with no gather.</summary>
    public bool AllContinuous { get; }

    private readonly bool[] _discrete;     // [channel]
    private readonly int[] _polOffset;     // [channel] -> first column in the policy output
    private readonly int[] _polLen;        // [channel] -> 1 (continuous) or K (discrete)
    private readonly int[] _contIndex;     // [channel] -> index into LogStd/std, or -1 for discrete

    private readonly int[] _contChannels;  // channel indices that are continuous (length ContinuousCount)
    private readonly int[] _contPolCols;   // policy columns of continuous means (length ContinuousCount)
    private readonly int[] _discreteChannels;     // channel indices that are discrete
    private readonly int[][] _discretePolCols;    // [discreteIdx] -> its K policy columns

    public ActionLayout(ControlSpec controls)
    {
        Controls = controls;
        int n = controls.Count;
        _discrete = new bool[n];
        _polOffset = new int[n];
        _polLen = new int[n];
        _contIndex = new int[n];

        var contCh = new List<int>();
        var contCols = new List<int>();
        var discCh = new List<int>();
        var discCols = new List<int[]>();

        int col = 0, contIdx = 0;
        for (int k = 0; k < n; k++)
        {
            var ch = controls.Channels[k];
            _polOffset[k] = col;
            if (ch.IsDiscrete)
            {
                int categories = Categories(ch);
                _discrete[k] = true;
                _polLen[k] = categories;
                _contIndex[k] = -1;
                discCh.Add(k);
                var cols = new int[categories];
                for (int i = 0; i < categories; i++) cols[i] = col + i;
                discCols.Add(cols);
                col += categories;
            }
            else
            {
                _discrete[k] = false;
                _polLen[k] = 1;
                _contIndex[k] = contIdx;
                contCh.Add(k);
                contCols.Add(col);
                col += 1;
                contIdx++;
            }
        }

        PolicyOutSize = col;
        ContinuousCount = contIdx;
        AllContinuous = ContinuousCount == n;
        _contChannels = contCh.ToArray();
        _contPolCols = contCols.ToArray();
        _discreteChannels = discCh.ToArray();
        _discretePolCols = discCols.ToArray();
    }

    /// <summary>Clamp a host log-σ array into <see cref="LogStdMin"/>..<see cref="LogStdMax"/> in place.
    /// The rollout drivers call this before exp() so the noise they sample with matches the band the
    /// autograd update enforces.</summary>
    public static void ClampLogStd(float[] logStd)
    {
        for (int i = 0; i < logStd.Length; i++)
            logStd[i] = Math.Clamp(logStd[i], LogStdMin, LogStdMax);
    }

    /// <summary>Option count of a discrete channel: <c>round(Max − Min) + 1</c> (≥ 2).</summary>
    public static int Categories(ControlChannel ch)
    {
        int k = (int)MathF.Round(ch.Max - ch.Min) + 1;
        if (k < 2)
            throw new ArgumentException(
                $"Discrete channel '{ch.Name}' must span at least two options (Min={ch.Min}, Max={ch.Max}).");
        return k;
    }

    /// <summary>
    /// Host sampler used by both rollout drivers. Reads one agent's policy output (length
    /// <see cref="PolicyOutSize"/>), writes its env-facing action (length <see cref="ChannelCount"/>),
    /// and returns the action log-prob. Continuous channels draw a Gaussian (one
    /// <see cref="PpoUpdate.NextGaussian"/> each, in channel order — so an all-continuous spec consumes
    /// the RNG identically to the original inline code); discrete channels draw a categorical sample
    /// (one <see cref="Random.NextDouble"/> each).
    /// </summary>
    public float Sample(ReadOnlySpan<float> policyOut, Span<float> envAction,
        ReadOnlySpan<float> std, ReadOnlySpan<float> logStd, Random rng)
    {
        float logp = 0f;
        for (int k = 0; k < ChannelCount; k++)
        {
            int p = _polOffset[k];
            if (!_discrete[k])
            {
                int c = _contIndex[k];
                float mean = policyOut[p];
                float act = mean + std[c] * PpoUpdate.NextGaussian(rng);
                act = Math.Clamp(act, -1f, 1f);
                envAction[k] = act;
                float d = (act - mean) / std[c];
                logp += -0.5f * d * d - logStd[c] - 0.5f * PpoUpdate.Log2Pi;
            }
            else
            {
                int kK = _polLen[k];
                float max = float.NegativeInfinity;
                for (int i = 0; i < kK; i++) max = MathF.Max(max, policyOut[p + i]);
                float sum = 0f;
                for (int i = 0; i < kK; i++) sum += MathF.Exp(policyOut[p + i] - max);

                double u = rng.NextDouble();
                int chosen = kK - 1;
                float cum = 0f;
                for (int i = 0; i < kK; i++)
                {
                    cum += MathF.Exp(policyOut[p + i] - max) / sum;
                    if (u <= cum) { chosen = i; break; }
                }
                envAction[k] = chosen;
                logp += (policyOut[p + chosen] - max) - MathF.Log(sum);
            }
        }
        return logp;
    }

    /// <summary>Greedy (deterministic) action for inference: continuous → mean clamped to [-1,1];
    /// discrete → the argmax category index. No RNG, no exploration.</summary>
    public void Greedy(ReadOnlySpan<float> policyOut, Span<float> envAction)
    {
        for (int k = 0; k < ChannelCount; k++)
        {
            int p = _polOffset[k];
            if (!_discrete[k])
            {
                envAction[k] = Math.Clamp(policyOut[p], -1f, 1f);
            }
            else
            {
                int kK = _polLen[k];
                int best = 0;
                float bv = policyOut[p];
                for (int i = 1; i < kK; i++)
                    if (policyOut[p + i] > bv) { bv = policyOut[p + i]; best = i; }
                envAction[k] = best;
            }
        }
    }

    /// <summary>
    /// Differentiable action log-prob and policy entropy for a minibatch, the autograd half of the PPO
    /// update. <paramref name="policyOut"/> is the policy head output (mb, <see cref="PolicyOutSize"/>);
    /// <paramref name="actionsFlat"/> is the stored env-facing actions (mb × <see cref="ChannelCount"/>,
    /// constants); <paramref name="logStd"/> is the (<see cref="ContinuousCount"/>,) Gaussian log-σ.
    /// Returns per-sample log-prob (mb,) and the scalar entropy term used by the loss.
    /// </summary>
    public (Tensor logp, Tensor entropy) LogpAndEntropy(Tensor policyOut, float[] actionsFlat, int mb, Tensor logStd)
    {
        Tensor? logp = null;
        Tensor? entropyConst = null;   // scalar: continuous (state-independent) entropy
        Tensor? entropyVar = null;     // (mb,): discrete (per-sample) entropy

        if (ContinuousCount > 0)
        {
            // Continuous means: the policy output itself when all-continuous, else gather the mean columns.
            Tensor mean = AllContinuous ? policyOut : TensorOps.IndexSelect(policyOut, 1, _contPolCols);

            var contAct = new float[mb * ContinuousCount];
            for (int r = 0; r < mb; r++)
                for (int c = 0; c < ContinuousCount; c++)
                    contAct[r * ContinuousCount + c] = actionsFlat[r * ChannelCount + _contChannels[c]];
            var actT = Tensor.FromShaped(contAct, new[] { mb, ContinuousCount });

            var logStdC = logStd.Clamp(LogStdMin, LogStdMax);        // bound σ so the Gaussian stays finite
            var std = logStdC.Exp();                                 // (C,)
            var diff = (actT - mean) / std;                          // (mb, C) broadcast std
            var terms = -0.5f * diff.Square() - logStdC - 0.5f * PpoUpdate.Log2Pi;
            logp = terms.Sum(new[] { 1 });                           // (mb,)
            entropyConst = (logStdC + PpoUpdate.HalfLog2PiE).Sum();  // scalar
        }

        for (int j = 0; j < _discreteChannels.Length; j++)
        {
            int dch = _discreteChannels[j];
            var logits = TensorOps.IndexSelect(policyOut, 1, _discretePolCols[j]);  // (mb, K)
            var logsm = logits.LogSoftmax(1);                                       // (mb, K)

            var idx = new int[mb];
            for (int r = 0; r < mb; r++)
                idx[r] = (int)MathF.Round(actionsFlat[r * ChannelCount + dch]);
            var chosen = TensorOps.Gather(logsm, 1, idx, new[] { mb, 1 }).Reshape(mb);  // (mb,)
            logp = logp == null ? chosen : logp + chosen;

            var ent = (logits.Softmax(1) * logsm).Sum(new[] { 1 }).Neg();           // (mb,)
            entropyVar = entropyVar == null ? ent : entropyVar + ent;
        }

        Tensor entropy =
            entropyConst != null && entropyVar != null ? entropyConst + entropyVar.Mean()
            : entropyConst ?? entropyVar!.Mean();

        return (logp!, entropy);
    }
}
