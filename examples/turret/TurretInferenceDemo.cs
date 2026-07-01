using System.Collections.Generic;
using Godot;

namespace Bascule.Godot.Examples;

/// <summary>
/// The shipping half of the turret loop: load the model <see cref="TurretDemo"/> saved and run it in
/// Inference mode — greedy, GPU-free. Because the policy is mixed, this also proves the discrete head
/// round-trips: it tallies each arena's cumulative shots and hits and reports the accuracy, which for a
/// trained policy sits well above the ~0% a random controller manages. Loads the model, assigns it to a
/// <c>AutoStart=false</c> agent, then starts the run — the pattern for configuring before init.
/// </summary>
[GlobalClass]
public partial class TurretInferenceDemo : Node2D
{
    /// <summary>The Inference-mode agent to drive. If unset, the first <see cref="LearningAgent"/> child is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Where to load the trained model from (matches the training demo's save path).</summary>
    [Export] public string ModelPath { get; set; } = "user://turret_model.tres";

    /// <summary>Headless run quits after this many physics ticks.</summary>
    [Export] public int HeadlessRunTicks { get; set; } = 1200;

    /// <summary>Physics rate used only in headless runs.</summary>
    [Export] public int HeadlessPhysicsTicksPerSecond { get; set; } = 600;

    private int _ticks;
    private bool _ok;
    private readonly List<TurretArena> _arenas = new();

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushError("TurretInferenceDemo: no LearningAgent child found.");
            return;
        }

        if (!ResourceLoader.Exists(ModelPath))
        {
            GD.PushError($"[TurretInferenceDemo] no model at '{ModelPath}'. Run the training demo (TurretDemo) first.");
            return;
        }

        var model = ResourceLoader.Load<ModelResource>(ModelPath);
        if (model == null || !model.HasModel)
        {
            GD.PushError($"[TurretInferenceDemo] '{ModelPath}' did not load as a usable ModelResource.");
            return;
        }

        Agent.Model = model;
        Agent.StartRun();   // the agent is authored AutoStart=false, so we start it after assigning the model
        CollectArenas();
        _ok = true;
        GD.Print($"[TurretInferenceDemo] loaded {ModelPath}; running greedy inference on {Agent.ArenaCount} arenas.");

        if (IsHeadless())
        {
            Engine.PhysicsTicksPerSecond = HeadlessPhysicsTicksPerSecond;
            Engine.MaxFps = 0;
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!_ok || Engine.IsEditorHint()) return;
        _ticks++;

        if (_ticks % 300 == 0)
        {
            var (shots, hits) = Tally();
            float acc = shots > 0 ? (float)hits / shots : 0f;
            GD.Print($"[TurretInferenceDemo] tick {_ticks}: {hits}/{shots} shots hit (accuracy {acc:0.00}).");
        }

        if (IsHeadless() && _ticks >= HeadlessRunTicks)
        {
            var (shots, hits) = Tally();
            float acc = shots > 0 ? (float)hits / shots : 0f;
            GD.Print($"[TurretInferenceDemo] done: {hits}/{shots} shots hit (accuracy {acc:0.00}) over {_ticks} ticks. Quitting.");
            GetTree().Quit();
        }
    }

    private (int shots, int hits) Tally()
    {
        int s = 0, h = 0;
        foreach (var a in _arenas) { s += a.Shots; h += a.Hits; }
        return (s, h);
    }

    private void CollectArenas()
    {
        _arenas.Clear();
        if (Agent == null) return;
        // The agent spawns each ArenaScene as its own child; collect the turret arenas to read their tallies.
        foreach (var child in Agent.GetChildren())
            if (child is TurretArena ta) _arenas.Add(ta);
    }

    private LearningAgent? FindAgent()
    {
        foreach (var child in GetChildren())
            if (child is LearningAgent la) return la;
        return null;
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
