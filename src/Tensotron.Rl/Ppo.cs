namespace Tensotron.Rl;

/// <summary>
/// Continuous-action PPO (clipped surrogate + GAE), built entirely on Tensotron tensors. Adapted
/// from the engine showcase's ContinuousPpo and widened from a single scalar action to the
/// multi-channel <see cref="ControlSpec"/> shape: the policy emits one normalized value in [-1,1]
/// per channel and the full action vector is handed to <see cref="IEnvironment.Step"/>. Rollouts run
/// under a one-shot CPU weight snapshot (launch-free); only the update builds an autograd graph.
///
/// Discrete channels are not yet modelled — every channel is treated as a continuous Gaussian.
/// Per v1 scope that's deferred until after the continuous loop (pole-cart) is proven.
/// </summary>
public sealed class Ppo
{
    private static readonly float Log2Pi = MathF.Log(2f * MathF.PI);
    private static readonly float HalfLog2PiE = 0.5f * MathF.Log(2f * MathF.PI * MathF.E);

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

    private readonly ActorCritic _ac;
    private readonly Func<IEnvironment> _envFactory;
    private readonly Adam _opt;
    private readonly Random _rng;

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

    private float NextGaussian()
    {
        // Box–Muller
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>Run training for the given number of PPO iterations. The callback (if any)
    /// receives (iteration, meanRolloutEpisodeReturn) for logging.</summary>
    public void Train(int iterations, Action<int, float>? onIteration = null)
    {
        _opt.LearningRate = LearningRate;
        int S = _ac.StateSize, A = _ac.ActionSize;
        var envs = Enumerable.Range(0, NumEnvs).Select(_ => _envFactory()).ToArray();
        var cur = envs.Select(e => e.Reset()).ToArray();

        for (int iter = 0; iter < iterations; iter++)
        {
            int T = Horizon, E = NumEnvs, B = T * E;
            var bStates = new float[B * S];
            var bActions = new float[B * A];
            var bLogp = new float[B];
            var bRewards = new float[B];
            var bValues = new float[B];
            var bDones = new float[B];

            var logStd = _ac.LogStd.ToArray();
            var std = logStd.Select(MathF.Exp).ToArray();

            float epReturnSum = 0f; int epCount = 0;
            var running = new float[E];

            // ---- rollout: launch-free CPU inference from a one-shot weight snapshot ----
            // (per-step device forwards would be hundreds of tiny launch+sync round-trips).
            var cpu = _ac.SnapshotCpu();
            for (int t = 0; t < T; t++)
            {
                var (means, values) = ForwardCpu(cpu, cur, E);
                for (int e = 0; e < E; e++)
                {
                    int row = (t * E + e);
                    Array.Copy(cur[e], 0, bStates, row * S, S);
                    bValues[row] = values[e];

                    float logp = 0f;
                    for (int k = 0; k < A; k++)
                    {
                        float mean = means[e * A + k];
                        float act = mean + std[k] * NextGaussian();
                        act = Math.Clamp(act, -1f, 1f);
                        bActions[row * A + k] = act;
                        float d = (act - mean) / std[k];
                        logp += -0.5f * d * d - logStd[k] - 0.5f * Log2Pi;
                    }
                    bLogp[row] = logp;

                    // multi-channel: hand the whole normalized action vector to the env.
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

            // ---- GAE advantages + returns ----
            var adv = new float[B];
            var ret = new float[B];
            for (int e = 0; e < E; e++)
            {
                float lastGae = 0f;
                for (int t = T - 1; t >= 0; t--)
                {
                    int row = t * E + e;
                    float nextValue = t == T - 1 ? finalValues[e] : bValues[(t + 1) * E + e];
                    float nextNonTerminal = 1f - bDones[row];
                    float delta = bRewards[row] + Gamma * nextValue * nextNonTerminal - bValues[row];
                    lastGae = delta + Gamma * Lambda * nextNonTerminal * lastGae;
                    adv[row] = lastGae;
                    ret[row] = lastGae + bValues[row];
                }
            }

            // normalize advantages
            float advMean = adv.Average();
            float advStd = MathF.Sqrt(adv.Select(x => (x - advMean) * (x - advMean)).Sum() / adv.Length) + 1e-8f;
            for (int i = 0; i < B; i++) adv[i] = (adv[i] - advMean) / advStd;

            // ---- PPO update epochs ----
            var idx = Enumerable.Range(0, B).ToArray();
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                Shuffle(idx);
                for (int start = 0; start < B; start += MinibatchSize)
                {
                    int mb = Math.Min(MinibatchSize, B - start);
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
                    UpdateMinibatch(mb, S, A, mbStates, mbActions, mbLogp, mbAdv, mbRet);
                }
            }

            float meanReturn = epCount > 0 ? epReturnSum / epCount : running.Average();
            onIteration?.Invoke(iter, meanReturn);
        }
    }

    // Launch-free rollout inference: evaluate the snapshotted policy+value on the host for E
    // states. No tensors, no kernel launches, no Synchronize — the whole point of the split.
    private static (float[] means, float[] values) ForwardCpu(CpuActorCritic cpu, float[][] states, int E)
    {
        int A = cpu.ActionSize;
        var means = new float[E * A];
        var values = new float[E];
        var m = new float[A];
        for (int e = 0; e < E; e++)
        {
            cpu.Forward(states[e], m, out float v);
            for (int k = 0; k < A; k++) means[e * A + k] = m[k];
            values[e] = v;
        }
        return (means, values);
    }

    private void UpdateMinibatch(int mb, int S, int A,
        float[] states, float[] actions, float[] oldLogp, float[] adv, float[] ret)
    {
        var st = Tensor.FromShaped(states, new[] { mb, S });
        var actT = Tensor.FromShaped(actions, new[] { mb, A });
        var oldLogpT = Tensor.FromShaped(oldLogp, new[] { mb });
        var advT = Tensor.FromShaped(adv, new[] { mb });
        var retT = Tensor.FromShaped(ret, new[] { mb });

        var mean = _ac.PolicyMean(st);                       // (mb, A)
        var logStd = _ac.LogStd;                             // (A,)
        var stdT = logStd.Exp();                             // (A,)

        // log N(act; mean, std) summed over action dims -> (mb,)
        var diff = (actT - mean) / stdT;                     // (mb, A) broadcast std
        var logpTerms = -0.5f * diff.Square() - logStd - 0.5f * Log2Pi;
        var logp = logpTerms.Sum(new[] { 1 });               // (mb,)

        var ratio = (logp - oldLogpT).Exp();                 // (mb,)
        var surr1 = ratio * advT;
        var surr2 = ratio.Clamp(1f - ClipEps, 1f + ClipEps) * advT;
        var policyLoss = TensorOps.Minimum(surr1, surr2).Mean().Neg();

        var value = _ac.Value(st);                           // (mb,)
        var valueLoss = (value - retT).Square().Mean();

        // differential entropy of the Gaussian policy = sum_k (logStd_k + 0.5 log 2πe)
        var entropy = (logStd + HalfLog2PiE).Sum();

        var loss = policyLoss + ValueCoef * valueLoss - EntropyCoef * entropy;

        _opt.ZeroGrad();
        loss.Backward();
        GradUtils.ClipGradNorm(_ac.Parameters(), MaxGradNorm, returnTotalNorm: false);
        _opt.Step();
    }

    /// <summary>Greedy evaluation (mean action, no exploration). Returns the average number
    /// of steps survived across the given number of episodes.</summary>
    public float EvaluateMeanSteps(int episodes, int maxSteps)
    {
        var cpu = _ac.SnapshotCpu();
        int A = cpu.ActionSize;
        var mean = new float[A];
        var act = new float[A];
        float total = 0f;
        for (int ep = 0; ep < episodes; ep++)
        {
            var env = _envFactory();
            var s = env.Reset();
            int steps = 0;
            for (; steps < maxSteps; steps++)
            {
                cpu.Forward(s, mean, out _);
                for (int k = 0; k < A; k++) act[k] = Math.Clamp(mean[k], -1f, 1f);
                var (_, done) = env.Step(act);
                s = env.GetState();
                if (done) { steps++; break; }
            }
            total += steps;
        }
        return total / episodes;
    }

    private void Shuffle(int[] a)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = _rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
