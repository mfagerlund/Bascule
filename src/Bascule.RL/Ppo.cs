namespace Bascule.RL;

/// <summary>
/// Continuous-action PPO (clipped surrogate + GAE), built entirely on Tensotron tensors. Adapted
/// from the engine showcase's ContinuousPpo and widened from a single scalar action to the
/// multi-channel <see cref="ControlSpec"/> shape: continuous channels emit a normalized value in
/// [-1,1], discrete channels a category index, and the full action vector is handed to
/// <see cref="IEnvironment.Step"/>. Sampling, scoring, and greedy evaluation all defer to
/// <see cref="ActionLayout"/>. Rollouts run under a one-shot CPU weight snapshot (launch-free); only
/// the update builds an autograd graph.
/// </summary>
public sealed class Ppo
{
    public float Gamma = 0.99f;
    public float Lambda = 0.95f;
    public float ClipEps = 0.2f;
    public int Epochs = 10;
    public int MinibatchSize = 512;
    public float LearningRate = 2e-3f;
    public float EntropyCoef = 0.0f;
    public float ValueCoef = 0.5f;
    public float MaxGradNorm = 0.5f;
    public int NumEnvs = 16;
    public int Horizon = 256;

    /// <summary>Trust-region KL early-stop target; when &gt; 0 an update bails out once a minibatch's
    /// approx-KL exceeds <c>1.5 × TargetKl</c>. ≤ 0 disables it. See <see cref="BatchedPpoTrainer.TargetKl"/>.</summary>
    public float TargetKl = 0.02f;

    /// <summary>Reactive LR backoff on instability — crash-guard skips (catastrophic) or a streak of KL
    /// early-stops (slow drift); see <see cref="BatchedPpoTrainer"/> for the full rationale.</summary>
    public bool LrBackoffOnInstability = true;
    public float LrBackoffFactor = 0.5f;
    public float LrBackoffMinLr = 1e-5f;
    public int LrBackoffSkipThreshold = 0;
    public int LrBackoffKlStreak = 3;
    private int _klHotStreak;

    private readonly ActorCritic _ac;
    private readonly Func<IEnvironment> _envFactory;
    private readonly Adam _opt;
    private readonly Random _rng;

    private float _retStd = 1f;          // running RMS of returns: the value head learns ret/_retStd (bounded)
    private bool _retStdInit;
    private const float RetStdBeta = 0.1f;

    /// <summary>Mean total PPO loss of the most recently completed iteration (0 before the first).</summary>
    public float LastLoss { get; private set; }
    /// <summary>Minibatch updates the crash guard skipped in the most recent iteration (0 = healthy).</summary>
    public int LastSkippedUpdates { get; private set; }
    /// <summary>Mean approx-KL of the most recent update (NaN before the first); the trust-region read-out.</summary>
    public float LastApproxKl { get; private set; } = float.NaN;
    /// <summary>True if the most recent update hit KL early-stop and ran fewer than <see cref="Epochs"/> passes.</summary>
    public bool LastKlEarlyStopped { get; private set; }
    /// <summary>Running RMS of returns the value targets are normalized by (keeps the critic from diverging).</summary>
    public float ReturnScale => _retStd;

    public Ppo(ActorCritic ac, Func<IEnvironment> envFactory, Random rng)
    {
        _ac = ac;
        _envFactory = envFactory;
        _rng = rng;
        _opt = new Adam(ac.Parameters(), lr: LearningRate);

        // A single minibatch update (3-layer MLP forward+backward + PPO loss + clip + Adam) is
        // well under 1024 launches and makes no host pull. FlushEvery=1024 spans a whole update so
        // the safety drain doesn't sync mid-update; drains still fire across minibatches, keeping
        // the in-flight queue bounded. Pure MLP: no parked data-dependent index buffers, so nothing
        // accumulates between drains.
        TensorRuntime.Instance.FlushEvery = 1024;
    }

    /// <summary>Run training for the given number of PPO iterations. The callback (if any)
    /// receives (iteration, meanRolloutEpisodeReturn) for logging.</summary>
    public void Train(int iterations, Action<int, float>? onIteration = null)
    {
        int S = _ac.StateSize, A = _ac.ActionSize, P = _ac.PolicyOutSize;
        var envs = Enumerable.Range(0, NumEnvs).Select(_ => _envFactory()).ToArray();
        var cur = envs.Select(e => e.Reset()).ToArray();

        for (int iter = 0; iter < iterations; iter++)
        {
            _opt.LearningRate = LearningRate;   // re-read each iteration so reactive LR backoff can take effect
            int T = Horizon, E = NumEnvs, B = T * E;
            var bStates = new float[B * S];
            var bActions = new float[B * A];
            var bLogp = new float[B];
            var bRewards = new float[B];
            var bValues = new float[B];
            var bDones = new float[B];

            var logStd = _ac.LogStd.ToArray();
            ActionLayout.ClampLogStd(logStd);                       // sample with the enforced σ band
            var std = logStd.Select(MathF.Exp).ToArray();

            float epReturnSum = 0f; int epCount = 0;
            var running = new float[E];

            // ---- rollout: launch-free CPU inference from a one-shot weight snapshot ----
            // (per-step device forwards would be hundreds of tiny launch+sync round-trips).
            var cpu = _ac.SnapshotCpu();
            for (int t = 0; t < T; t++)
            {
                var (policyOut, values) = ForwardCpu(cpu, cur, E);
                for (int e = 0; e < E; e++)
                {
                    int row = (t * E + e);
                    Array.Copy(cur[e], 0, bStates, row * S, S);
                    bValues[row] = values[e] * _retStd;             // scale normalized value back to raw for GAE

                    // mixed continuous/discrete sampling via the layout (one float per channel into bActions).
                    bLogp[row] = _ac.Layout.Sample(
                        new ReadOnlySpan<float>(policyOut, e * P, P),
                        new Span<float>(bActions, row * A, A),
                        std, logStd, _rng);

                    // multi-channel: hand the whole action vector to the env.
                    var (reward, done) = envs[e].Step(new ReadOnlySpan<float>(bActions, row * A, A));
                    bRewards[row] = reward;
                    bDones[row] = done ? 1f : 0f;
                    running[e] += reward;

                    if (done)
                    {
                        epReturnSum += running[e]; epCount++;
                        running[e] = 0f;
                        cur[e] = envs[e].Reset();
                    }
                    else
                    {
                        cur[e] = envs[e].GetState();
                    }
                }
            }

            // bootstrap value for the final state of each env (same snapshot)
            var (_, finalValues) = ForwardCpu(cpu, cur, E);
            for (int e = 0; e < E; e++) finalValues[e] *= _retStd;   // raw-return units, like bValues

            // ---- GAE advantages + returns, then the PPO update (shared core) ----
            var adv = new float[B];
            var ret = new float[B];
            PpoUpdate.ComputeGae(Gamma, Lambda, T, E, bRewards, bValues, bDones, finalValues, adv, ret);
            UpdateReturnScale(ret);                                  // recalibrate σ_ret from the raw returns
            PpoUpdate.NormalizeAdvantages(adv);
            for (int i = 0; i < ret.Length; i++) ret[i] /= _retStd;  // value target in normalized return units
            LastLoss = PpoUpdate.RunUpdateEpochs(_ac, _opt, _rng, Epochs, MinibatchSize, ClipEps, ValueCoef, EntropyCoef,
                MaxGradNorm, B, S, A, bStates, bActions, bLogp, adv, ret,
                out int skipped, TargetKl, out float approxKl, out bool klEarlyStopped);
            LastSkippedUpdates = skipped;
            LastApproxKl = approxKl;
            LastKlEarlyStopped = klEarlyStopped;

            // Reactive backstop (see BatchedPpoTrainer): shrink LR on catastrophic skips OR a streak of KL
            // early-stops (slow drift). Takes effect next iteration via `_opt.LearningRate = LearningRate`.
            _klHotStreak = klEarlyStopped ? _klHotStreak + 1 : 0;
            if (LrBackoffOnInstability && LearningRate > LrBackoffMinLr
                && (skipped > LrBackoffSkipThreshold
                    || (LrBackoffKlStreak > 0 && _klHotStreak >= LrBackoffKlStreak)))
            {
                LearningRate = MathF.Max(LearningRate * LrBackoffFactor, LrBackoffMinLr);
                _klHotStreak = 0;
            }

            float meanReturn = epCount > 0 ? epReturnSum / epCount : running.Average();
            onIteration?.Invoke(iter, meanReturn);
        }
    }

    // Track the running RMS of returns so value targets stay ~unit scale (the critic can't diverge).
    // Calibrated directly on the first healthy batch, then EMA; a NaN/degenerate batch keeps the prior.
    private void UpdateReturnScale(float[] ret)
    {
        float rms = PpoUpdate.ReturnRms(ret);
        if (!float.IsFinite(rms) || rms <= 0f) return;
        _retStd = _retStdInit ? (1f - RetStdBeta) * _retStd + RetStdBeta * rms : rms;
        _retStd = MathF.Max(_retStd, 1e-4f);
        _retStdInit = true;
    }

    // Launch-free rollout inference: evaluate the snapshotted policy+value on the host for E
    // states. No tensors, no kernel launches, no Synchronize — the whole point of the split.
    private static (float[] policyOut, float[] values) ForwardCpu(CpuActorCritic cpu, float[][] states, int E)
    {
        int P = cpu.PolicyOutSize;
        var outBuf = new float[E * P];
        var values = new float[E];
        var m = new float[P];
        for (int e = 0; e < E; e++)
        {
            cpu.Forward(states[e], m, out float v);
            for (int k = 0; k < P; k++) outBuf[e * P + k] = m[k];
            values[e] = v;
        }
        return (outBuf, values);
    }

    /// <summary>Greedy evaluation (mean action, no exploration). Returns the average number
    /// of steps survived across the given number of episodes.</summary>
    public float EvaluateMeanSteps(int episodes, int maxSteps)
    {
        var cpu = _ac.SnapshotCpu();
        int P = cpu.PolicyOutSize, A = _ac.ActionSize;
        var policyOut = new float[P];
        var act = new float[A];
        float total = 0f;
        for (int ep = 0; ep < episodes; ep++)
        {
            var env = _envFactory();
            var s = env.Reset();
            int steps = 0;
            for (; steps < maxSteps; steps++)
            {
                cpu.Forward(s, policyOut, out _);
                _ac.Layout.Greedy(policyOut, act);
                var (_, done) = env.Step(act);
                s = env.GetState();
                if (done) { steps++; break; }
            }
            total += steps;
        }
        return total / episodes;
    }
}
