namespace Bascule.RL;

/// <summary>
/// The three interfaces a component advertises so the trainer can compose an environment from it
/// without knowing what it is. This is the project's load-bearing idea — a gun, a car, a leg joint,
/// and a shader parameter are all just an observation, a control surface, and a reward. They are
/// deliberately Godot-free (only <see cref="System.Span{T}"/> and <see cref="ControlSpec"/>), so the
/// same composition works for console sims and unit tests as for Godot nodes.
/// </summary>
public interface IObservationSource
{
    /// <summary>Number of floats this source contributes to the observation vector.</summary>
    int Size { get; }

    /// <summary>Write exactly <see cref="Size"/> floats into <paramref name="dst"/> (its length is Size).</summary>
    void Write(Span<float> dst);
}

/// <summary>Something the policy can drive: it declares its channels via <see cref="Spec"/> and
/// receives the matching slice of the action vector each step.</summary>
public interface IControlSurface
{
    /// <summary>The channels this surface exposes (names, ranges, discrete flags).</summary>
    ControlSpec Spec { get; }

    /// <summary>Apply the action slice for this surface's channels (length == Spec.Count) over <paramref name="dt"/>.</summary>
    void Apply(ReadOnlySpan<float> action, float dt);
}

/// <summary>A reward signal and episode-termination condition. Multiple sources are summed (reward)
/// and OR-ed (done) by the composition.</summary>
public interface IRewardSource
{
    /// <summary>Reward accrued for the step just taken.</summary>
    float Reward { get; }

    /// <summary>True when the episode should terminate (failure or success).</summary>
    bool Done { get; }

    /// <summary>Reset any per-episode accumulators at the start of a new episode.</summary>
    void ResetEpisode();
}

/// <summary>
/// Optional fourth discovery role: restoring an arena's <em>world</em> to a fresh start-of-episode
/// state — teleport bodies, zero velocities, re-randomize. Distinct from <see cref="IRewardSource"/>'s
/// per-episode bookkeeping reset: a node implements this when ending an episode means physically
/// resetting the scene. In the synchronous/console path the caller owns world reset (via
/// <see cref="CompositeEnvironment"/>'s reset delegate); in Godot, where the world is the scene, the
/// arena node advertises this and the composition restores it when the episode terminates.
/// </summary>
public interface IEpisodeReset
{
    /// <summary>Restore this part of the world to a fresh episode start.</summary>
    void ResetEpisode();
}
