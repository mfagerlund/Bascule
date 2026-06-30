namespace Tensotron.Rl;

/// <summary>
/// Adapts a <see cref="CompositeAgent"/> to the synchronous <see cref="IEnvironment"/> contract the
/// in-process <see cref="Ppo"/> trainer consumes, for the self-contained console/test path where the
/// caller owns the simulation. Each <see cref="Step"/> applies the action, advances the world one
/// tick via the supplied delegate, then reads the resulting reward and done flag.
///
/// This is NOT how Godot trains — there the frame loop is Godot's and observations are batched across
/// arenas per <c>_physics_process</c> tick; that path drives the same <see cref="CompositeAgent"/>
/// directly. This adapter exists so the discovery abstraction is trainable and testable without an
/// editor.
/// </summary>
public sealed class CompositeEnvironment : IEnvironment
{
    private readonly CompositeAgent _agent;
    private readonly Action _advanceWorld;
    private readonly Action _resetWorld;
    private readonly float _dt;
    private readonly float[] _obs;

    /// <param name="agent">The composed observation/control/reward routing.</param>
    /// <param name="advanceWorld">Steps the underlying simulation exactly one tick.</param>
    /// <param name="resetWorld">Returns the underlying simulation to a fresh start-of-episode state.</param>
    /// <param name="dt">Timestep passed to control surfaces' <see cref="IControlSurface.Apply"/>.</param>
    public CompositeEnvironment(CompositeAgent agent, Action advanceWorld, Action resetWorld, float dt = 0.02f)
    {
        _agent = agent;
        _advanceWorld = advanceWorld;
        _resetWorld = resetWorld;
        _dt = dt;
        _obs = new float[agent.ObservationSize];
    }

    public int ObservationSize => _agent.ObservationSize;
    public ControlSpec Controls => _agent.Controls;

    public float[] Reset()
    {
        _resetWorld();
        _agent.ResetEpisode();
        return GetState();
    }

    public float[] GetState()
    {
        _agent.WriteObservation(_obs);
        return (float[])_obs.Clone();
    }

    public (float reward, bool done) Step(ReadOnlySpan<float> action)
    {
        _agent.ApplyAction(action, _dt);
        _advanceWorld();
        return (_agent.CollectReward(), _agent.IsDone());
    }
}
