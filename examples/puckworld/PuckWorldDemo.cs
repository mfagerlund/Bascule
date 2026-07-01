using Godot;

namespace Bascule.Godot.Examples;

/// <summary>
/// Root of the PuckWorld example: one <see cref="LearningAgent"/> (Train mode) replicates a
/// <see cref="PuckArena"/> <c>ArenaCount</c> times and trains across all of them in one batched tick.
/// Windowed, you watch a grid of pucks go from wandering into the red enemy to confidently hugging the
/// green target and peeling away from danger. Headless, it cranks <c>PhysicsTicksPerSecond</c> for
/// throughput and quits once <see cref="TargetIterations"/> is reached, so the scene doubles as a smoke
/// test of the whole stack.
/// </summary>
[GlobalClass]
public partial class PuckWorldDemo : Node2D
{
    /// <summary>The agent to drive. If unset, the first <see cref="LearningAgent"/> child is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Headless run quits after this many completed PPO iterations.</summary>
    [Export] public int TargetIterations { get; set; } = 60;

    /// <summary>Where to save the trained model. The watch demo loads from here. Empty = skip saving.</summary>
    [Export] public string SaveModelPath { get; set; } = "user://puckworld_model.tres";

    /// <summary>Optional: also snapshot the policy at each of these iterations to
    /// <c>{CheckpointDir}puck-gen-{iter}.tres</c>, to capture the learning arc for per-generation SVGs.
    /// Empty = off; does not change the shipped run.</summary>
    [Export] public int[] CheckpointIters { get; set; } = System.Array.Empty<int>();

    /// <summary>Directory (Godot path) the <see cref="CheckpointIters"/> snapshots are written to.</summary>
    [Export] public string CheckpointDir { get; set; } = "user://";

    /// <summary>Physics rate used only in headless runs, to train faster than real time.</summary>
    [Export] public int HeadlessPhysicsTicksPerSecond { get; set; } = 600;

    private bool _saved;
    private float _emaReturn;
    private float _bestReturn = float.NegativeInfinity;
    private bool _emaInit;

    public override void _EnterTree()
    {
        // Headless config sweep: ++ --save=<path> --iters=<N> plus the arena-shaping flags. Must run here
        // (before the LearningAgent child's _Ready spawns the arenas) so the overrides are in place.
        PuckArena.ApplyCmdlineOverrides();
        bool sawSave = false;
        foreach (string arg in OS.GetCmdlineUserArgs())
        {
            if (arg.StartsWith("--save=")) { SaveModelPath = arg["--save=".Length..]; sawSave = true; }
            else if (arg.StartsWith("--iters="))
                TargetIterations = int.Parse(arg["--iters=".Length..], System.Globalization.CultureInfo.InvariantCulture);
        }
        if (sawSave) CheckpointIters = System.Array.Empty<int>();   // sweep runs only need the best-EMA model
    }

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushError("PuckWorldDemo: no LearningAgent child found.");
            return;
        }
        Agent.IterationCompleted += OnIteration;

        if (IsHeadless())
        {
            Engine.PhysicsTicksPerSecond = HeadlessPhysicsTicksPerSecond;
            Engine.MaxFps = 0;
            GD.Print($"[PuckWorldDemo] headless training: target {TargetIterations} iterations, " +
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
        // Smooth the noisy per-iteration return and checkpoint the best, not the last — the keep-away
        // reward is dense and negative, so the climb is "less negative over time".
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
            string path = $"{CheckpointDir}puck-gen-{iteration}.tres";
            Agent.SaveModel(path);
            GD.Print($"[PuckWorldDemo] checkpoint saved: {path}");
        }

        GD.Print($"[PuckWorldDemo] iter {iteration}: meanEpisodeReturn={meanReturn:0.0}  ema={_emaReturn:0.0}  " +
                 $"episodes={Agent?.EpisodesCompleted}  kl={Agent?.LastApproxKl:0.0000}  " +
                 $"lr={Agent?.CurrentLearningRate:0.######}  skipped={Agent?.TotalSkippedUpdates}" +
                 $"{(newBest ? "  <- saved best" : "")}");

        if (iteration < TargetIterations || _saved) return;
        _saved = true;

        if (IsHeadless())
        {
            GD.Print($"[PuckWorldDemo] reached {TargetIterations} iterations " +
                     $"(best ema meanReturn={_bestReturn:0.0}). Quitting.");
            GetTree().Quit();
            return;
        }

        Agent?.Stop();
        GD.Print($"[PuckWorldDemo] reached {TargetIterations} iterations (best ema meanReturn={_bestReturn:0.0}). " +
                 "Training frozen.");
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
