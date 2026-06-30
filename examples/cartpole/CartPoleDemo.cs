using Godot;

namespace Tensotron.Godot.Examples;

/// <summary>
/// Root of the pole-cart example: holds one <see cref="LearningAgent"/> (Train mode) that replicates a
/// <c>CartPole2D</c> arena <c>ArenaCount</c> times and trains across all of them in one batched tick.
/// It just listens to the agent's <c>IterationCompleted</c> signal and logs progress.
///
/// Under a windowed run you watch the fleet of poles go from flailing to balancing. Headless, it cranks
/// <c>PhysicsTicksPerSecond</c> for throughput and quits once <see cref="TargetIterations"/> is reached
/// — which is exactly how this scene doubles as a runtime smoke test of the whole stack.
/// </summary>
[GlobalClass]
public partial class CartPoleDemo : Node2D
{
    /// <summary>The agent to drive. If unset, the first <see cref="LearningAgent"/> child is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Headless run quits after this many completed PPO iterations.</summary>
    [Export] public int TargetIterations { get; set; } = 12;

    /// <summary>Where to save the trained model once <see cref="TargetIterations"/> is reached. The
    /// inference demo loads from this same path. Leave empty to skip saving.</summary>
    [Export] public string SaveModelPath { get; set; } = "user://cartpole_model.tres";

    /// <summary>Physics rate used only in headless runs, to train faster than real time.</summary>
    [Export] public int HeadlessPhysicsTicksPerSecond { get; set; } = 600;

    private bool _saved;

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushError("CartPoleDemo: no LearningAgent child found.");
            return;
        }
        Agent.IterationCompleted += OnIteration;

        if (IsHeadless())
        {
            Engine.PhysicsTicksPerSecond = HeadlessPhysicsTicksPerSecond;
            Engine.MaxFps = 0;
            GD.Print($"[CartPoleDemo] headless training: target {TargetIterations} iterations, " +
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
        GD.Print($"[CartPoleDemo] iter {iteration}: meanEpisodeReturn={meanReturn:0.0}");
        if (iteration < TargetIterations || _saved) return;
        _saved = true;

        if (!string.IsNullOrEmpty(SaveModelPath) && Agent != null)
        {
            Error err = Agent.SaveModel(SaveModelPath);
            GD.Print($"[CartPoleDemo] saved model to {SaveModelPath} ({err}).");
        }

        // Headless: this scene doubles as a smoke test, so finish and exit. Windowed: keep the trained
        // poles balancing on screen.
        if (IsHeadless())
        {
            GD.Print($"[CartPoleDemo] reached {TargetIterations} iterations " +
                     $"(last meanReturn={meanReturn:0.0}). Quitting.");
            GetTree().Quit();
        }
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
