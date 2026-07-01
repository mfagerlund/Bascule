namespace Bascule.RL;

/// <summary>Per-iteration training summary, emitted by <see cref="BatchedPpoTrainer.OnIterationComplete"/>
/// when a rollout segment fills and a PPO update runs. This is what the editor dock plots.
/// <paramref name="Loss"/> is the mean total PPO loss (clipped-surrogate + value − entropy) across the
/// iteration's minibatch updates.</summary>
public readonly record struct TrainingStats(
    int Iteration, float MeanEpisodeReturn, int EpisodesCompleted, long TotalAgentSteps, float Loss);

/// <summary>
/// PPO turned inside out for a host that owns the clock. Where <see cref="Ppo"/> drives its own
/// rollout loop and pulls each <see cref="IEnvironment"/> forward, this trainer is <em>driven</em>:
/// the caller (Godot's <c>_physics_process</c>, or a console/test harness) advances the world and,
/// once per tick, hands over every agent's current observation plus the reward/done that resulted
/// from the <em>previous</em> tick's action, and gets back this tick's actions to apply.
///
/// This is the batched-not-threaded shape the README mandates: all <c>N</c> agents are stepped in one
/// tick, so a tick is exactly <b>one</b> policy forward over an <c>[N, obs]</c> batch (a single GEMM
/// on GPU / one row-parallel SIMD matmul on CPU) under <see cref="Tensor.NoGradScope"/> — never N
/// forwards, never per-agent threads. Transitions accumulate into a rollout buffer of
/// <see cref="Horizon"/> × <see cref="AgentCount"/> steps; when it fills, one PPO update runs (the same
/// <see cref="PpoUpdate"/> core the synchronous trainer uses) and the next segment begins.
///
/// Reward attribution is deferred by one tick because, in an externally-clocked loop, the physical
/// consequence of an action is only observable after the host integrates the next step. So the
/// <c>reward</c>/<c>done</c> passed to <see cref="Tick"/> describe the action returned by the previous
/// <see cref="Tick"/>; the very first call ignores them. <c>done</c> follows Gym auto-reset semantics:
/// it marks the just-completed transition as terminal (GAE stops bootstrapping across it), and the
/// host is expected to have already reset that agent so the observation it passes is the new episode's
/// start. Godot-free by construction.
/// </summary>
public sealed class BatchedPpoTrainer
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

    /// <summary>Trust-region KL early-stop target. When &gt; 0, an update bails out the instant a
    /// minibatch's approx-KL exceeds <c>1.5 × TargetKl</c> — the preventive half of the stability story:
    /// it bounds how far one update moves the policy, so ratios can't explode. ≤ 0 disables it.</summary>
    public float TargetKl = 0.02f;

    /// <summary>Reactive learning-rate backoff: shrink the LR (×<see cref="LrBackoffFactor"/>, floored at
    /// <see cref="LrBackoffMinLr"/>) when the policy strains against its trust region. Two triggers:
    /// (1) any iteration that trips the crash guard (<c>skipped &gt; <see cref="LrBackoffSkipThreshold"/></c>) —
    /// the catastrophic backstop; (2) <see cref="LrBackoffKlStreak"/> <em>consecutive</em> iterations that
    /// hit the KL early-stop — the slow-drift signal that survives KL early-stop (which suppresses the
    /// skips trigger #1 would otherwise watch). A streak resets on any healthy (full-epoch) iteration, so
    /// the occasional early-stop during productive learning doesn't throttle the LR. Set false for a fixed LR.</summary>
    public bool LrBackoffOnInstability = true;
    public float LrBackoffFactor = 0.5f;
    public float LrBackoffMinLr = 1e-5f;
    public int LrBackoffSkipThreshold = 0;
    public int LrBackoffKlStreak = 3;          // consecutive KL-early-stops before backing off; ≤ 0 disables this trigger
    private int _klHotStreak;

    private readonly ActorCritic _ac;
    private readonly ActionLayout _layout;
    private readonly Adam _opt;
    private readonly Random _rng;
    private readonly int _policyOutSize;

    /// <summary>Number of agents stepped together each tick (the rollout's env count, "E").</summary>
    public int AgentCount { get; }
    /// <summary>Steps collected per agent before a PPO update fires (the rollout length, "T").</summary>
    public int Horizon { get; }
    public int ObservationSize { get; }
    public int ActionSize { get; }
    /// <summary>The control layout the policy drives; saved alongside the weights for inference.</summary>
    public ControlSpec Controls { get; }
    /// <summary>The trained network — pass with <see cref="Controls"/> to ModelSerializer / ModelResource to save.</summary>
    public ActorCritic Network => _ac;

    /// <summary>Completed PPO iterations (rollout segments) so far.</summary>
    public int Iteration { get; private set; }
    /// <summary>Total agent-steps fed through <see cref="Tick"/> (AgentCount per tick).</summary>
    public long TotalAgentSteps { get; private set; }
    /// <summary>Mean episode return of the most recently completed iteration (0 before the first).</summary>
    public float LastMeanReturn { get; private set; }
    /// <summary>Mean total PPO loss of the most recently completed iteration (0 before the first).</summary>
    public float LastLoss { get; private set; }
    /// <summary>Minibatch updates the crash guard skipped in the most recent iteration (0 = healthy).</summary>
    public int LastSkippedUpdates { get; private set; }
    /// <summary>Total minibatch updates the crash guard has skipped since the run started.</summary>
    public long TotalSkippedUpdates { get; private set; }
    /// <summary>Mean approx-KL between the rollout policy and the post-update policy for the most recent
    /// iteration — the trust-region health read-out KL early-stop watches (NaN before the first update).</summary>
    public float LastApproxKl { get; private set; } = float.NaN;
    /// <summary>True if the most recent iteration's update hit the KL early-stop and ran fewer than
    /// <see cref="Epochs"/> full passes.</summary>
    public bool LastKlEarlyStopped { get; private set; }
    /// <summary>Running RMS of returns the value targets are normalized by (keeps the critic from diverging).</summary>
    public float ReturnScale => _retStd;

    /// <summary>Raised once per completed rollout segment, right after its PPO update. For live plots/logging.</summary>
    public event Action<TrainingStats>? OnIterationComplete;

    private readonly int _b;                  // Horizon * AgentCount
    private readonly float[] _states, _actions, _logp, _rewards, _values, _dones;
    private readonly float[] _adv, _ret, _finalValues;
    private readonly float[] _running;        // per-agent episode-return accumulator (per segment)
    private readonly float[] _scratchValues;  // per-tick value output (AgentCount)

    private float[] _logStd = System.Array.Empty<float>();
    private float[] _std = System.Array.Empty<float>();
    private float _retStd = 1f;          // running RMS of returns: the value head learns ret/_retStd (bounded)
    private bool _retStdInit;
    private const float RetStdBeta = 0.1f;
    private int _filled;                       // completed steps in the current segment (0..Horizon)
    private bool _hasPending;                  // an action has been issued whose reward/done isn't recorded yet
    private float _segReturnSum;
    private int _segEpisodes;

    public BatchedPpoTrainer(ActorCritic ac, ControlSpec controls, int agentCount, int horizon, Random rng)
    {
        if (agentCount <= 0) throw new ArgumentOutOfRangeException(nameof(agentCount));
        if (horizon <= 0) throw new ArgumentOutOfRangeException(nameof(horizon));
        if (controls.Count != ac.ActionSize)
            throw new ArgumentException(
                $"ControlSpec has {controls.Count} channels but the network emits {ac.ActionSize} actions.",
                nameof(controls));

        _ac = ac;
        _layout = ac.Layout;
        _rng = rng;
        Controls = controls;
        AgentCount = agentCount;
        Horizon = horizon;
        ObservationSize = ac.StateSize;
        ActionSize = ac.ActionSize;
        _policyOutSize = ac.PolicyOutSize;
        _opt = new Adam(ac.Parameters(), lr: LearningRate);

        // Same rationale as Ppo: a single minibatch update stays under the launch cap, and FlushEvery
        // spanning a whole update keeps the safety drain from syncing mid-update. The per-tick rollout
        // forward pulls to host (ToArray) each tick, so the in-flight queue stays bounded regardless.
        TensorRuntime.Instance.FlushEvery = 1024;

        int b = horizon * agentCount;
        _b = b;
        _states = new float[b * ObservationSize];
        _actions = new float[b * ActionSize];
        _logp = new float[b];
        _rewards = new float[b];
        _values = new float[b];
        _dones = new float[b];
        _adv = new float[b];
        _ret = new float[b];
        _finalValues = new float[agentCount];
        _running = new float[agentCount];
        _scratchValues = new float[agentCount];

        BeginSegment();
    }

    /// <summary>
    /// Advance the trainer by one tick for all <see cref="AgentCount"/> agents at once.
    /// </summary>
    /// <param name="observations">Current observations, agent-major: length AgentCount × ObservationSize.</param>
    /// <param name="rewards">Reward each agent earned from the <em>previous</em> tick's action (length AgentCount).
    /// Ignored on the very first call.</param>
    /// <param name="dones">Whether each agent's previous transition ended its episode (length AgentCount).
    /// The host should have already reset any done agent, so its <paramref name="observations"/> slice is the
    /// new episode's start. Ignored on the very first call.</param>
    /// <param name="actions">Receives this tick's actions to apply, agent-major: length AgentCount × ActionSize.</param>
    /// <returns>True if this tick completed a rollout segment and ran a PPO update.</returns>
    public bool Tick(
        ReadOnlySpan<float> observations,
        ReadOnlySpan<float> rewards,
        ReadOnlySpan<float> dones,
        Span<float> actions)
    {
        int n = AgentCount, s = ObservationSize, a = ActionSize;
        if (observations.Length < n * s) throw new ArgumentException("observations too short.", nameof(observations));
        if (actions.Length < n * a) throw new ArgumentException("actions too short.", nameof(actions));
        if (_hasPending)
        {
            if (rewards.Length < n) throw new ArgumentException("rewards too short.", nameof(rewards));
            if (dones.Length < n) throw new ArgumentException("dones too short.", nameof(dones));
        }

        bool optimized = false;

        // 1. Close out the previous tick's step with the reward/done it produced.
        if (_hasPending)
        {
            int step = _filled;
            for (int e = 0; e < n; e++)
            {
                int row = step * n + e;
                _rewards[row] = rewards[e];
                bool done = dones[e] != 0f;
                _dones[row] = done ? 1f : 0f;
                _running[e] += rewards[e];
                if (done)
                {
                    _segReturnSum += _running[e];
                    _segEpisodes++;
                    _running[e] = 0f;
                }
            }
            _filled++;
            _hasPending = false;
        }

        // 2. A full segment: bootstrap from the current (post-last-step) state, update, then restart.
        //    The bootstrap value MUST use the rollout-time weights, so compute it before optimizing.
        if (_filled == Horizon)
        {
            ForwardBatch(observations, policyOut: null, _finalValues);
            Optimize();
            EmitStats();
            BeginSegment();
            optimized = true;
        }

        // 3. Open a new step at the current state: store it, forward once, sample, scatter actions.
        int cur = _filled;
        observations.Slice(0, n * s).CopyTo(_states.AsSpan(cur * n * s, n * s));

        var stepPolicyOut = new float[n * _policyOutSize];
        ForwardBatch(observations, stepPolicyOut, _scratchValues);
        for (int e = 0; e < n; e++)
        {
            int row = cur * n + e;
            _values[row] = _scratchValues[e];
            // mixed continuous/discrete sampling via the layout; writes the env action for this agent.
            _logp[row] = _layout.Sample(
                new ReadOnlySpan<float>(stepPolicyOut, e * _policyOutSize, _policyOutSize),
                actions.Slice(e * a, a),
                _std, _logStd, _rng);
            for (int k = 0; k < a; k++) _actions[row * a + k] = actions[e * a + k];
        }
        _hasPending = true;
        TotalAgentSteps += n;
        return optimized;
    }

    // One batched policy/value forward over the [N, obs] batch under no-grad (no rollout graph).
    // Writes the raw policy head into `policyOut` (if non-null, length N*PolicyOutSize) and values
    // into `valuesOut` (length N).
    private void ForwardBatch(ReadOnlySpan<float> observations, float[]? policyOut, float[] valuesOut)
    {
        int n = AgentCount, s = ObservationSize, p = _policyOutSize;
        var flat = new float[n * s];
        observations.Slice(0, n * s).CopyTo(flat);
        using (Tensor.NoGradScope())
        {
            var st = Tensor.FromShaped(flat, new[] { n, s });
            var valT = _ac.Value(st);                 // (N,) in normalized return units
            var v = valT.ToArray();
            for (int e = 0; e < n; e++) valuesOut[e] = v[e] * _retStd;   // scale back to raw returns for GAE
            if (policyOut != null)
            {
                var outT = _ac.PolicyOutput(st);      // (N, PolicyOutSize)
                var m = outT.ToArray();
                for (int i = 0; i < n * p; i++) policyOut[i] = m[i];
            }
        }
    }

    private void Optimize()
    {
        _opt.LearningRate = LearningRate;
        PpoUpdate.ComputeGae(Gamma, Lambda, Horizon, AgentCount,
            _rewards, _values, _dones, _finalValues, _adv, _ret);
        UpdateReturnScale();                                          // recalibrate σ_ret from the raw returns
        PpoUpdate.NormalizeAdvantages(_adv);
        for (int i = 0; i < _ret.Length; i++) _ret[i] /= _retStd;     // value target in normalized return units
        LastLoss = PpoUpdate.RunUpdateEpochs(_ac, _opt, _rng, Epochs, MinibatchSize, ClipEps, ValueCoef, EntropyCoef,
            MaxGradNorm, _b, ObservationSize, ActionSize, _states, _actions, _logp, _adv, _ret,
            out int skipped, TargetKl, out float approxKl, out bool klEarlyStopped);
        LastSkippedUpdates = skipped;
        TotalSkippedUpdates += skipped;
        LastApproxKl = approxKl;
        LastKlEarlyStopped = klEarlyStopped;

        // Track how long the policy has been straining against the trust region. A healthy iteration (no
        // early-stop) resets it; a persistent streak means the policy keeps wanting to bolt — the slow-drift
        // signal that survives KL early-stop (which suppresses the crash-guard skips trigger #1 watches).
        _klHotStreak = klEarlyStopped ? _klHotStreak + 1 : 0;

        // Reactive LR backoff: shrink the step so we descend gently instead of bouncing off the wall. The
        // best-checkpoint (host side) keeps the peak; this just keeps the run from drifting worse. Monotonic
        // down (never anneals back up) and floored, so it converges rather than freezing dead.
        if (LrBackoffOnInstability && LearningRate > LrBackoffMinLr)
        {
            bool catastrophic = skipped > LrBackoffSkipThreshold;
            bool persistentStrain = LrBackoffKlStreak > 0 && _klHotStreak >= LrBackoffKlStreak;
            if (catastrophic || persistentStrain)
            {
                LearningRate = MathF.Max(LearningRate * LrBackoffFactor, LrBackoffMinLr);
                _klHotStreak = 0;   // require another full streak before backing off again
            }
        }
    }

    // Track the running RMS of returns so value targets stay ~unit scale (the critic can't diverge).
    // Calibrated directly on the first healthy batch, then EMA; a NaN/degenerate batch keeps the prior.
    private void UpdateReturnScale()
    {
        float rms = PpoUpdate.ReturnRms(_ret);
        if (!float.IsFinite(rms) || rms <= 0f) return;
        _retStd = _retStdInit ? (1f - RetStdBeta) * _retStd + RetStdBeta * rms : rms;
        _retStd = MathF.Max(_retStd, 1e-4f);
        _retStdInit = true;
    }

    private void EmitStats()
    {
        float meanReturn = _segEpisodes > 0 ? _segReturnSum / _segEpisodes : SegPartialMean();
        Iteration++;
        LastMeanReturn = meanReturn;
        OnIterationComplete?.Invoke(new TrainingStats(Iteration, meanReturn, _segEpisodes, TotalAgentSteps, LastLoss));
    }

    // Fallback when a segment completed with no finished episode: mean of in-flight running returns,
    // mirroring Ppo's `running.Average()`.
    private float SegPartialMean()
    {
        float sum = 0f;
        for (int e = 0; e < _running.Length; e++) sum += _running[e];
        return sum / _running.Length;
    }

    // Start a fresh rollout segment: re-read the (now-updated) exploration noise, reset per-segment
    // counters. Agent world state is the host's; it carries across the boundary (episodes are not cut).
    private void BeginSegment()
    {
        _logStd = _ac.LogStd.ToArray();
        ActionLayout.ClampLogStd(_logStd);                  // sample with the same σ band the update enforces
        _std = new float[_logStd.Length];
        for (int k = 0; k < _logStd.Length; k++) _std[k] = MathF.Exp(_logStd[k]);
        _filled = 0;
        _hasPending = false;
        System.Array.Clear(_running, 0, _running.Length);
        _segReturnSum = 0f;
        _segEpisodes = 0;
    }
}
