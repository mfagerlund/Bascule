using System;
using System.Collections.Generic;
using Godot;
using Bascule.RL;

namespace Bascule.Godot;

/// <summary>
/// The Godot face of the trainer: drop it above a set of identical <em>training areas</em>, point it
/// at observation/control/reward source nodes, press play, and it trains a PPO policy in-process — no
/// Python, no native runtime. It is the adapter between Godot's frame loop and the Godot-free
/// <see cref="BatchedPpoTrainer"/>.
///
/// Each direct child (or each node in <see cref="TrainingAreas"/>) is one arena; every descendant that
/// implements <see cref="IObservationSource"/>/<see cref="IControlSurface"/>/<see cref="IRewardSource"/>
/// (and optionally <see cref="IEpisodeReset"/>) is discovered and composed into that arena's
/// <see cref="CompositeAgent"/>. All arenas must share one observation/control layout — they are meant
/// to be instances of the same scene — so the policy sees a homogeneous <c>[N, obs]</c> batch.
///
/// The frame loop is inverted relative to <see cref="Ppo"/>: Godot integrates physics between
/// <see cref="_PhysicsProcess"/> ticks, so each tick reads the reward/done produced by last tick's
/// action, auto-resets any finished episode, gathers the batch, runs <b>one</b> policy forward for all
/// arenas, and scatters the actions back — exactly the batched-not-threaded shape the README mandates.
/// </summary>
[Tool]
[GlobalClass]
public partial class LearningAgent : Node
{
    public enum AgentMode { Idle, Train, Inference }

    /// <summary>Idle does nothing; Train learns a policy; Inference runs a saved <see cref="Model"/>.</summary>
    [Export] public AgentMode Mode { get; set; } = AgentMode.Idle;

    /// <summary>If true, the run starts automatically in <see cref="_Ready"/>. Set false when a controller
    /// must configure the agent first (e.g. assign a loaded <see cref="Model"/>) and then call
    /// <see cref="StartRun"/> itself.</summary>
    [Export] public bool AutoStart { get; set; } = true;

    /// <summary>The arena roots to batch across. If empty, every direct child of this node is one arena.</summary>
    [Export] public global::Godot.Collections.Array<NodePath> TrainingAreas { get; set; } = new();

    /// <summary>Inference: the model to load and run. Train: the resource <see cref="CaptureModel"/> writes into.</summary>
    [Export] public ModelResource? Model { get; set; }

    [ExportGroup("Arenas")]
    /// <summary>Optional: a single arena scene to replicate. When set, <see cref="ArenaCount"/> copies are
    /// instantiated as children before discovery — the batched "many arenas in one tick" design from one
    /// scene + a count. Leave null to use whatever arenas are already in the tree.</summary>
    [Export] public PackedScene? ArenaScene { get; set; }

    /// <summary>How many copies of <see cref="ArenaScene"/> to spawn (ignored if it is null).</summary>
    [Export] public int ArenaCount { get; set; } = 1;

    [ExportGroup("Network")]
    [Export] public int Hidden { get; set; } = 64;

    [ExportGroup("Training")]
    [Export] public int Horizon { get; set; } = 256;
    [Export] public int Seed { get; set; } = 0;
    [Export] public float LearningRate { get; set; } = 2e-3f;
    [Export] public int Epochs { get; set; } = 10;
    [Export] public int MinibatchSize { get; set; } = 512;
    [Export] public float Gamma { get; set; } = 0.99f;
    [Export] public float Lambda { get; set; } = 0.95f;
    [Export] public float ClipEps { get; set; } = 0.2f;
    [Export] public float EntropyCoef { get; set; } = 0.0f;

    [ExportGroup("Stability")]
    /// <summary>Trust-region KL early-stop target. &gt; 0 bounds how far one update moves the policy (the
    /// preventive cure for peak-then-collapse); 0 disables it. Default 0.02 is the standard PPO range.</summary>
    [Export] public float TargetKl { get; set; } = 0.02f;
    /// <summary>When true, halve the learning rate after any iteration whose crash guard skipped a
    /// minibatch — the reactive backstop for instability KL early-stop doesn't catch. The best-checkpoint
    /// keeps the peak; this keeps the run from degrading further.</summary>
    [Export] public bool LrBackoffOnInstability { get; set; } = true;
    /// <summary>Factor the LR is multiplied by on an unstable iteration (0..1).</summary>
    [Export] public float LrBackoffFactor { get; set; } = 0.5f;
    /// <summary>Floor the reactive LR backoff will not go below.</summary>
    [Export] public float LrBackoffMinLr { get; set; } = 1e-5f;
    /// <summary>Back off the LR after this many <em>consecutive</em> iterations that hit KL early-stop —
    /// the slow-drift trigger (skips alone don't fire once KL early-stop suppresses them). ≤ 0 disables it.</summary>
    [Export] public int LrBackoffKlStreak { get; set; } = 3;

    /// <summary>Emitted after each completed rollout segment + PPO update. The training dock plots this.</summary>
    [Signal] public delegate void IterationCompletedEventHandler(int iteration, float meanReturn);

    private CompositeAgent[] _agents = Array.Empty<CompositeAgent>();
    private BatchedPpoTrainer? _trainer;
    private InferencePolicy? _policy;
    private float[] _obs = Array.Empty<float>();
    private float[] _rew = Array.Empty<float>();
    private float[] _done = Array.Empty<float>();
    private float[] _act = Array.Empty<float>();
    private int _obsSize, _actSize;
    private bool _firstTick;
    private bool _running;
    private int[] _epLen = Array.Empty<int>();  // per-arena steps in the current episode
    private long _epCount;
    private double _epLenSum;

    /// <summary>The live trainer once a Train run has started (null otherwise).</summary>
    public BatchedPpoTrainer? Trainer => _trainer;
    /// <summary>Completed PPO iterations so far.</summary>
    public int Iteration => _trainer?.Iteration ?? 0;
    /// <summary>Mean episode return of the most recently completed iteration.</summary>
    public float LastMeanReturn => _trainer?.LastMeanReturn ?? 0f;
    /// <summary>Mean total PPO loss of the most recently completed iteration (set before
    /// <see cref="IterationCompleted"/> fires, so handlers can read it).</summary>
    public float LastLoss => _trainer?.LastLoss ?? 0f;
    /// <summary>Minibatch updates the crash guard skipped last iteration (0 = healthy). &gt; 0 means a
    /// diverged/NaN update was caught and discarded instead of corrupting the policy.</summary>
    public int LastSkippedUpdates => _trainer?.LastSkippedUpdates ?? 0;
    /// <summary>Total minibatch updates the crash guard has skipped since the run started.</summary>
    public long TotalSkippedUpdates => _trainer?.TotalSkippedUpdates ?? 0;
    /// <summary>Mean approx-KL of the most recent update — the trust-region health read-out (NaN before
    /// the first update; a value much above <see cref="TargetKl"/> means KL early-stop is engaging).</summary>
    public float LastApproxKl => _trainer?.LastApproxKl ?? float.NaN;
    /// <summary>The trainer's current learning rate, which drops below the configured
    /// <see cref="LearningRate"/> as the reactive backoff engages.</summary>
    public float CurrentLearningRate => _trainer?.LearningRate ?? LearningRate;
    /// <summary>Episodes that have ended across all arenas since the run started (Train or Inference).</summary>
    public long EpisodesCompleted => _epCount;
    /// <summary>Mean episode length (steps survived) over all completed episodes — the natural
    /// performance read-out, and equal to mean return for unit-per-step rewards like cart-pole.</summary>
    public float MeanEpisodeLength => _epCount > 0 ? (float)(_epLenSum / _epCount) : 0f;

    public override void _Ready()
    {
        // Don't drive anything from the editor viewport; training runs when the scene plays.
        if (Engine.IsEditorHint()) return;
        if (Mode == AgentMode.Idle || !AutoStart) return;
        StartRun();
    }

    /// <summary>Discover arenas and set up the trainer (Train) or load the policy (Inference). Called
    /// automatically from <see cref="_Ready"/> at runtime; exposed so headless harnesses can start it.</summary>
    public void StartRun()
    {
        if (ArenaScene != null) SpawnArenas();
        _agents = DiscoverAgents();
        if (_agents.Length == 0)
            throw new InvalidOperationException(
                "LearningAgent found no training areas exposing discovery sources " +
                "(IObservationSource / IControlSurface / IRewardSource).");

        _obsSize = _agents[0].ObservationSize;
        _actSize = _agents[0].ActionSize;
        ValidateHomogeneous();

        int n = _agents.Length;
        _obs = new float[n * _obsSize];
        _rew = new float[n];
        _done = new float[n];
        _act = new float[n * _actSize];
        _epLen = new int[n];
        _epCount = 0;
        _epLenSum = 0;

        if (Mode == AgentMode.Train)
            SetupTrainer(n);
        else if (Mode == AgentMode.Inference)
            SetupInference();

        // Prime observations for the first tick.
        for (int e = 0; e < n; e++)
            _agents[e].WriteObservation(_obs.AsSpan(e * _obsSize, _obsSize));
        _firstTick = true;
        _running = true;
    }

    private void SetupTrainer(int n)
    {
        ValidateTrainingConfig();
        var rng = new Random(Seed);
        var ac = new ActorCritic(_obsSize, _agents[0].Controls, hidden: Hidden);
        _trainer = new BatchedPpoTrainer(ac, _agents[0].Controls, n, Horizon, rng)
        {
            LearningRate = LearningRate,
            Epochs = Epochs,
            MinibatchSize = MinibatchSize,
            Gamma = Gamma,
            Lambda = Lambda,
            ClipEps = ClipEps,
            EntropyCoef = EntropyCoef,
            TargetKl = TargetKl,
            LrBackoffOnInstability = LrBackoffOnInstability,
            LrBackoffFactor = LrBackoffFactor,
            LrBackoffMinLr = LrBackoffMinLr,
            LrBackoffKlStreak = LrBackoffKlStreak,
        };
        _trainer.OnIterationComplete += s =>
        {
            EmitSignal(SignalName.IterationCompleted, s.Iteration, s.MeanEpisodeReturn);
            EmitDebuggerStats(s);
        };
    }

    /// <summary>When the scene is played from the editor (remote debugger attached), stream this
    /// iteration's telemetry — and a fresh serialized model snapshot — to the editor's training dock
    /// over the <c>tensotron:stats</c> capture channel. A no-op for headless/shipped runs
    /// (<see cref="EngineDebugger.IsActive"/> is false), so it costs nothing off the editor.</summary>
    private void EmitDebuggerStats(TrainingStats s)
    {
        if (!EngineDebugger.IsActive() || _trainer == null) return;

        // The dock saves the model editor-side from these bytes, so no editor->game round-trip is
        // needed; serializing only happens while the debugger is attached.
        byte[] model;
        try { model = ModelSerializer.Save(_trainer.Network, _trainer.Controls); }
        catch { model = Array.Empty<byte>(); }

        EngineDebugger.SendMessage("tensotron:stats", new global::Godot.Collections.Array
        {
            s.Iteration, s.MeanEpisodeReturn, s.Loss,
            EpisodesCompleted, MeanEpisodeLength, TotalSkippedUpdates, model,
        });
    }

    private void SetupInference()
    {
        if (Model == null || !Model.HasModel)
            throw new InvalidOperationException("Inference mode requires a Model with saved data.");
        _policy = Model.LoadPolicy();
        if (_policy.ObservationSize != _obsSize)
            throw new InvalidOperationException(
                $"Model expects observation size {_policy.ObservationSize}, but the agents provide {_obsSize}.");
        // Matching action SIZE isn't enough — a saved continuous policy must not be driven onto a discrete
        // (or reordered) control surface of the same width. Compare the full channel layout.
        if (!_policy.Controls.ChannelsMatch(_agents[0].Controls, out string? reason))
            throw new InvalidOperationException(
                $"Model control layout does not match the agent ({reason}).");
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!_running || Engine.IsEditorHint()) return;
        float dt = (float)delta;

        // Read the consequence of last tick's action (now integrated), auto-reset finished episodes,
        // and gather the current batch. Skipped on the very first tick (no prior action).
        if (!_firstTick) ReadOutcomesAndObserve();
        _firstTick = false;

        if (Mode == AgentMode.Train)
            _trainer!.Tick(_obs, _rew, _done, _act);
        else
            for (int e = 0; e < _agents.Length; e++)
                _policy!.Act(_obs.AsSpan(e * _obsSize, _obsSize), _act.AsSpan(e * _actSize, _actSize));

        // Apply this tick's actions; Godot integrates physics before the next tick.
        for (int e = 0; e < _agents.Length; e++)
            _agents[e].ApplyAction(_act.AsSpan(e * _actSize, _actSize), dt);
    }

    private void ReadOutcomesAndObserve()
    {
        for (int e = 0; e < _agents.Length; e++)
        {
            _rew[e] = _agents[e].CollectReward();
            bool done = _agents[e].IsDone();
            _epLen[e]++;
            _done[e] = done ? 1f : 0f;
            if (done)
            {
                _epLenSum += _epLen[e];
                _epCount++;
                _epLen[e] = 0;
                _agents[e].ResetEpisode();         // world reset (IEpisodeReset) + reward bookkeeping
            }
            _agents[e].WriteObservation(_obs.AsSpan(e * _obsSize, _obsSize));
        }
    }

    /// <summary>Snapshot the trained policy into <see cref="Model"/> (creating it if needed). This is
    /// what the dock's Save Model button calls; persist it with <c>ResourceSaver.Save</c>.</summary>
    public void CaptureModel()
    {
        if (_trainer == null)
            throw new InvalidOperationException("No trainer to capture — start a Train run first.");
        Model ??= new ModelResource();
        Model.Data = ModelSerializer.Save(_trainer.Network, _trainer.Controls);
    }

    /// <summary>Snapshot the trained policy and write it to <paramref name="path"/> (e.g.
    /// <c>res://models/turret.tres</c>). Returns the <see cref="Error"/> from <c>ResourceSaver</c>.</summary>
    public Error SaveModel(string path)
    {
        CaptureModel();
        return ResourceSaver.Save(Model!, path);
    }

    /// <summary>Freeze the run: stop ticking the trainer/policy so the world holds still. Used by demos to
    /// lock in a good policy and let a replay overlay loop indefinitely instead of training into collapse.</summary>
    public void Stop() => _running = false;

    // Catch inspector misconfiguration before it becomes a hang or a cryptic deep-stack crash. A
    // MinibatchSize of 0, for example, would spin forever in the PPO epoch loop.
    private void ValidateTrainingConfig()
    {
        void Require(bool ok, string msg)
        {
            if (!ok) throw new InvalidOperationException($"LearningAgent training config: {msg}");
        }
        Require(Horizon > 0, $"Horizon must be > 0 (got {Horizon}).");
        Require(Hidden > 0, $"Hidden must be > 0 (got {Hidden}).");
        Require(Epochs > 0, $"Epochs must be > 0 (got {Epochs}).");
        Require(MinibatchSize > 0, $"MinibatchSize must be > 0 (got {MinibatchSize}).");
        Require(float.IsFinite(LearningRate) && LearningRate > 0, $"LearningRate must be finite and > 0 (got {LearningRate}).");
        Require(float.IsFinite(Gamma) && Gamma is >= 0f and <= 1f, $"Gamma must be in [0,1] (got {Gamma}).");
        Require(float.IsFinite(Lambda) && Lambda is >= 0f and <= 1f, $"Lambda must be in [0,1] (got {Lambda}).");
        Require(float.IsFinite(ClipEps) && ClipEps > 0, $"ClipEps must be finite and > 0 (got {ClipEps}).");
        Require(float.IsFinite(EntropyCoef) && EntropyCoef >= 0, $"EntropyCoef must be finite and >= 0 (got {EntropyCoef}).");
        Require(float.IsFinite(TargetKl), $"TargetKl must be finite (got {TargetKl}); use 0 or less to disable KL early-stop.");
        if (LrBackoffOnInstability)
        {
            Require(float.IsFinite(LrBackoffFactor) && LrBackoffFactor is > 0f and < 1f,
                $"LrBackoffFactor must be in (0,1) (got {LrBackoffFactor}).");
            Require(float.IsFinite(LrBackoffMinLr) && LrBackoffMinLr > 0,
                $"LrBackoffMinLr must be finite and > 0 (got {LrBackoffMinLr}).");
        }
    }

    private void SpawnArenas()
    {
        int count = Math.Max(1, ArenaCount);
        for (int i = 0; i < count; i++)
        {
            var arena = ArenaScene!.Instantiate();
            AddChild(arena);          // parent is in the tree, so the arena's _Ready runs synchronously here
        }
    }

    private CompositeAgent[] DiscoverAgents()
    {
        var roots = new List<Node>();
        if (TrainingAreas != null && TrainingAreas.Count > 0)
        {
            foreach (var path in TrainingAreas)
            {
                var node = GetNodeOrNull(path);
                if (node != null) roots.Add(node);
            }
        }
        else
        {
            foreach (var child in GetChildren()) roots.Add(child);
        }

        var agents = new List<CompositeAgent>();
        foreach (var root in roots)
        {
            var obs = new List<IObservationSource>();
            var ctrl = new List<IControlSurface>();
            var rew = new List<IRewardSource>();
            var reset = new List<IEpisodeReset>();
            CollectSources(root, obs, ctrl, rew, reset);
            if (obs.Count == 0 && ctrl.Count == 0 && rew.Count == 0)
                continue; // not a training area — no discovery sources under it

            // It HAS some sources, so it's meant to be a training area — but a partial set is a silent
            // footgun: a control-less arena crashes in the policy update, a reward-less one learns nothing.
            var missing = new List<string>();
            if (obs.Count == 0) missing.Add("IObservationSource");
            if (ctrl.Count == 0) missing.Add("IControlSurface");
            if (rew.Count == 0) missing.Add("IRewardSource");
            if (missing.Count > 0)
                throw new InvalidOperationException(
                    $"Training area '{root.Name}' is missing required source(s): {string.Join(", ", missing)}. " +
                    "Each arena needs at least one observation source, one control surface, and one reward source.");

            var agent = new CompositeAgent(obs, ctrl, rew, reset);
            if (agent.ActionSize == 0)
                throw new InvalidOperationException(
                    $"Training area '{root.Name}' has control surface(s) but zero action channels.");
            agents.Add(agent);
        }
        return agents.ToArray();
    }

    private static void CollectSources(
        Node node,
        List<IObservationSource> obs,
        List<IControlSurface> ctrl,
        List<IRewardSource> rew,
        List<IEpisodeReset> reset)
    {
        if (node is IObservationSource o) obs.Add(o);
        if (node is IControlSurface c) ctrl.Add(c);
        if (node is IRewardSource r) rew.Add(r);
        if (node is IEpisodeReset er) reset.Add(er);
        foreach (var child in node.GetChildren())
            CollectSources(child, obs, ctrl, rew, reset);
    }

    private void ValidateHomogeneous()
    {
        ControlSpec spec0 = _agents[0].Controls;
        for (int e = 1; e < _agents.Length; e++)
        {
            if (_agents[e].ObservationSize != _obsSize)
                throw new InvalidOperationException(
                    $"Training area {e} exposes observation size {_agents[e].ObservationSize}, but area 0 is " +
                    $"{_obsSize}. All areas must share one layout (same scene).");
            // Same action SIZE is not enough: a continuous and a discrete channel both occupy one action
            // slot but drive different policy heads, so the batched policy would be silently wrong.
            if (!spec0.ChannelsMatch(_agents[e].Controls, out string? reason))
                throw new InvalidOperationException(
                    $"Training area {e} control layout differs from area 0 ({reason}). " +
                    "All areas must share one control layout (same scene).");
        }
    }
}
