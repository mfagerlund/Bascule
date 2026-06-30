namespace Tensotron.Rl;

/// <summary>
/// Engine-agnostic continuous-control environment — the contract PPO trains against. Multi-channel:
/// <see cref="Step"/> takes one value per <see cref="ControlSpec"/> channel, each normalized to
/// [-1,1] (a control surface owns de-normalization to real units). Godot-free by design — the
/// addon adapts Godot nodes to this, so the RL core stays reusable and unit-testable without an editor.
/// </summary>
public interface IEnvironment
{
    int ObservationSize { get; }
    ControlSpec Controls { get; }

    /// <summary>Reset to a fresh start state and return its observation.</summary>
    float[] Reset();

    /// <summary>The current observation (length <see cref="ObservationSize"/>).</summary>
    float[] GetState();

    /// <summary>Advance one step with one normalized action per channel; returns reward and whether the episode terminated.</summary>
    (float reward, bool done) Step(ReadOnlySpan<float> action);
}
