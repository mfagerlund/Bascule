using Godot;

namespace Bascule.Godot.Examples;

/// <summary>
/// Root of the drift-racer example: one <see cref="LearningAgent"/> (Train mode) replicates a
/// <c>RaceCar</c> arena <c>ArenaCount</c> times and trains across all of them in one batched tick.
/// Windowed, you watch a fleet of cars go from spinning off every corner to flowing — and sliding —
/// around the circuit; the body tints orange as each car drifts. Headless, it cranks
/// <c>PhysicsTicksPerSecond</c> for throughput and quits once <see cref="TargetIterations"/> is
/// reached, so the scene doubles as a smoke test of the whole stack.
/// </summary>
[GlobalClass]
public partial class CarRaceDemo : Node2D
{
    /// <summary>The agent to drive. If unset, the first <see cref="LearningAgent"/> child is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Headless run quits after this many completed PPO iterations.</summary>
    [Export] public int TargetIterations { get; set; } = 60;

    /// <summary>Where to save the trained model once <see cref="TargetIterations"/> is reached. An
    /// inference demo can load from this same path. Leave empty to skip saving.</summary>
    [Export] public string SaveModelPath { get; set; } = "user://drift_racer_model.tres";

    /// <summary>Optional: also snapshot the policy at each of these iterations to
    /// <c>{CheckpointDir}gen-{iter}.tres</c>. Used to capture the learning arc (one model per generation)
    /// for the README's per-generation SVGs. Empty = off; does not change the shipped run.</summary>
    [Export] public int[] CheckpointIters { get; set; } = System.Array.Empty<int>();

    /// <summary>Directory (Godot path) the <see cref="CheckpointIters"/> snapshots are written to.</summary>
    [Export] public string CheckpointDir { get; set; } = "user://";

    /// <summary>Physics rate used only in headless runs, to train faster than real time. The car uses a
    /// fixed internal step, so the dynamics are unchanged by this — only throughput.</summary>
    [Export] public int HeadlessPhysicsTicksPerSecond { get; set; } = 600;

    private bool _saved;
    private float _emaReturn;        // smoothed meanReturn, so a single noisy spike can't claim "best"
    private float _bestReturn = float.NegativeInfinity;
    private bool _emaInit;

    public override void _EnterTree()
        => RaceCar.Overlay = true;   // render the whole fleet on one shared track (set before arenas spawn)

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushError("CarRaceDemo: no LearningAgent child found.");
            return;
        }
        Agent.IterationCompleted += OnIteration;

        if (IsHeadless())
        {
            Engine.PhysicsTicksPerSecond = HeadlessPhysicsTicksPerSecond;
            Engine.MaxFps = 0;
            GD.Print($"[CarRaceDemo] headless training: target {TargetIterations} iterations, " +
                     $"{Agent.ArenaCount} arenas, horizon {Agent.Horizon}.");
        }
    }

    private LearningAgent? FindAgent()
    {
        foreach (var child in GetChildren())
            if (child is LearningAgent la) return la;
        return null;
    }

    private void OnIteration(int iteration, float meanReturn)
    {
        // Smooth the (noisy) per-iteration return and checkpoint whenever it sets a new high. PPO here can
        // climb to a peak and then wobble down as the critic overshoots (the guarded "unstable updates"),
        // so the LAST model is often worse than the best — keep the peak instead of the tail.
        _emaReturn = _emaInit ? 0.8f * _emaReturn + 0.2f * meanReturn : meanReturn;
        _emaInit = true;

        bool newBest = iteration >= 5 && _emaReturn > _bestReturn && !string.IsNullOrEmpty(SaveModelPath) && Agent != null;
        if (newBest)
        {
            _bestReturn = _emaReturn;
            Agent!.SaveModel(SaveModelPath);
        }

        if (Agent != null && System.Array.IndexOf(CheckpointIters, iteration) >= 0)
        {
            string path = $"{CheckpointDir}gen-{iteration}.tres";
            Agent.SaveModel(path);
            GD.Print($"[CarRaceDemo] checkpoint saved: {path}");
        }

        GD.Print($"[CarRaceDemo] iter {iteration}: meanEpisodeReturn={meanReturn:0.0}  ema={_emaReturn:0.0}  " +
                 $"meanEpLen={Agent?.MeanEpisodeLength:0}  episodes={Agent?.EpisodesCompleted}  " +
                 $"kl={Agent?.LastApproxKl:0.0000}  lr={Agent?.CurrentLearningRate:0.######}  " +
                 $"skipped={Agent?.TotalSkippedUpdates}{(newBest ? "  <- saved best" : "")}");

        if (iteration < TargetIterations || _saved) return;
        _saved = true;

        if (IsHeadless())
        {
            GD.Print($"[CarRaceDemo] reached {TargetIterations} iterations " +
                     $"(best ema meanReturn={_bestReturn:0.0}). Quitting.");
            GetTree().Quit();
            return;
        }

        // Windowed: freeze the policy at its peak instead of training on into collapse. The replay overlay
        // keeps each car's best run ever, so the recorded race loops on the good runs from here.
        Agent?.Stop();
        GD.Print($"[CarRaceDemo] reached {TargetIterations} iterations (best ema meanReturn={_bestReturn:0.0}). " +
                 "Training frozen — looping the recorded race.");
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
