using System.Collections.Generic;
using Godot;

namespace Tensotron.Godot.Examples;

/// <summary>
/// Root of the turret example: one <see cref="LearningAgent"/> (Train mode) replicates the multi-node
/// <c>Turret.tscn</c> arena <c>ArenaCount</c> times and trains a mixed continuous+discrete policy across
/// all of them in one batched tick. A shared <see cref="TrainingHud"/> sibling shows the generic
/// iteration / mean-return / graph readout; this script only adds the turret-specific live hit-rate via
/// <see cref="TrainingHud.ExtraText"/>. Headless, it raises the physics rate and quits once
/// <see cref="TargetIterations"/> is reached, saving the model the inference demo loads.
/// </summary>
[GlobalClass]
public partial class TurretDemo : Node2D
{
    /// <summary>The agent to drive. If unset, the first <see cref="LearningAgent"/> child is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>The overlay to feed the live hit-rate into. If unset, the first <see cref="TrainingHud"/>
    /// child is used (windowed runs only).</summary>
    [Export] public TrainingHud? Hud { get; set; }

    /// <summary>Headless run quits after this many completed PPO iterations.</summary>
    [Export] public int TargetIterations { get; set; } = 30;

    /// <summary>Where to save the trained model; the inference demo loads from the same path.</summary>
    [Export] public string SaveModelPath { get; set; } = "user://turret_model.tres";

    /// <summary>Physics rate used only in headless runs, to train faster than real time.</summary>
    [Export] public int HeadlessPhysicsTicksPerSecond { get; set; } = 600;

    private bool _saved;
    private readonly List<TurretArena> _arenas = new();
    private double _sampleTimer;
    private int _prevShots, _prevHits;
    private float _recentHitRate;

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushError("TurretDemo: no LearningAgent child found.");
            return;
        }
        Agent.IterationCompleted += OnIteration;

        if (IsHeadless())
        {
            Engine.PhysicsTicksPerSecond = HeadlessPhysicsTicksPerSecond;
            Engine.MaxFps = 0;
            GD.Print($"[TurretDemo] headless training: target {TargetIterations} iterations, " +
                     $"{Agent.ArenaCount} arenas, horizon {Agent.Horizon}.");
        }
        else
        {
            // The agent spawned its arenas in its own _Ready (it runs before this parent's), so they're
            // here to read for the live hit-rate the HUD shows.
            foreach (var child in Agent.GetChildren())
                if (child is TurretArena ta) _arenas.Add(ta);
            Hud ??= FindHud();
        }
    }

    public override void _Process(double delta)
    {
        if (Engine.IsEditorHint() || Hud == null) return;

        _sampleTimer += delta;
        if (_sampleTimer >= 0.4)
        {
            var (shots, hits) = Tally();
            int dShots = shots - _prevShots, dHits = hits - _prevHits;
            if (dShots > 0)
            {
                float sample = 100f * dHits / dShots;                  // exploration-policy hit-rate this window
                _recentHitRate += 0.35f * (sample - _recentHitRate);   // EMA-smoothed so it doesn't jump
            }
            _prevShots = shots;
            _prevHits = hits;
            _sampleTimer = 0;
        }

        Hud.ExtraText =
            $"hit rate    {_recentHitRate:0}%  (training policy — noisy; greedy ~90%)\n" +
            "barrel flash:  green = hit,  red = miss";
    }

    private (int shots, int hits) Tally()
    {
        int s = 0, h = 0;
        foreach (var a in _arenas) { s += a.Shots; h += a.Hits; }
        return (s, h);
    }

    private LearningAgent? FindAgent()
    {
        foreach (var child in GetChildren())
            if (child is LearningAgent la) return la;
        return null;
    }

    private TrainingHud? FindHud()
    {
        foreach (var child in GetChildren())
            if (child is TrainingHud hud) return hud;
        return null;
    }

    private void OnIteration(int iteration, float meanReturn)
    {
        GD.Print($"[TurretDemo] iter {iteration}: meanReturn={meanReturn:0.00}");
        if (iteration < TargetIterations || _saved) return;
        _saved = true;

        if (!string.IsNullOrEmpty(SaveModelPath) && Agent != null)
        {
            Error err = Agent.SaveModel(SaveModelPath);
            GD.Print($"[TurretDemo] saved model to {SaveModelPath} ({err}).");
        }

        if (IsHeadless())
        {
            GD.Print($"[TurretDemo] reached {TargetIterations} iterations " +
                     $"(last meanReturn={meanReturn:0.00}). Quitting.");
            GetTree().Quit();
        }
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
