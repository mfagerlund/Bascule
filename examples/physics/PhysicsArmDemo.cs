using Godot;

namespace Tensotron.Godot.Examples;

/// <summary>
/// Root of the physics-control example: one <see cref="LearningAgent"/> (Train mode) replicates a
/// <c>PhysicsArm</c> arena <c>ArenaCount</c> times and trains across all of them in one batched tick.
/// Windowed, you watch a fleet of torque-actuated arms learn to swing onto their targets and brake;
/// headless, it quits once <see cref="TargetIterations"/> is reached, doubling as a smoke test of the
/// physics-control path.
///
/// Note this demo does <b>not</b> crank <c>PhysicsTicksPerSecond</c> (unlike the direct-control
/// cart-pole): real physics dynamics depend on the tick rate, so training and viewing run at the same
/// 60 Hz for a transferable policy.
/// </summary>
[GlobalClass]
public partial class PhysicsArmDemo : Node2D
{
    /// <summary>The agent to drive. If unset, the first <see cref="LearningAgent"/> child is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Headless run quits after this many completed PPO iterations.</summary>
    [Export] public int TargetIterations { get; set; } = 30;

    /// <summary>Where to save the trained model once <see cref="TargetIterations"/> is reached. Leave
    /// empty to skip saving.</summary>
    [Export] public string SaveModelPath { get; set; } = "user://physics_arm_model.tres";

    private bool _saved;

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushError("PhysicsArmDemo: no LearningAgent child found.");
            return;
        }
        Agent.IterationCompleted += OnIteration;

        if (IsHeadless())
        {
            Engine.MaxFps = 0;   // spin the loop; physics still steps at 60 Hz (rate-locked dynamics)
            GD.Print($"[PhysicsArmDemo] headless training: target {TargetIterations} iterations, " +
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
        GD.Print($"[PhysicsArmDemo] iter {iteration}: meanEpisodeReturn={meanReturn:0.0}");
        if (iteration < TargetIterations || _saved) return;
        _saved = true;

        if (!string.IsNullOrEmpty(SaveModelPath) && Agent != null)
        {
            Error err = Agent.SaveModel(SaveModelPath);
            GD.Print($"[PhysicsArmDemo] saved model to {SaveModelPath} ({err}).");
        }

        if (IsHeadless())
        {
            GD.Print($"[PhysicsArmDemo] reached {TargetIterations} iterations " +
                     $"(last meanReturn={meanReturn:0.0}). Quitting.");
            GetTree().Quit();
        }
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
