namespace Tensotron.Rl;

/// <summary>
/// A loaded, inference-ready policy: the shippable, GPU-free product of training. Wraps a host-side
/// <see cref="CpuActorCritic"/> (launch-free scalar forward) plus the <see cref="ControlSpec"/> it
/// was trained against, and produces a greedy (deterministic) action for one observation.
///
/// This is what Inference mode runs: no autograd, no optimizer, no value head in the hot path, no
/// accelerator round-trip — a single small-net forward per tick. Built by
/// <see cref="ModelSerializer.Load(byte[])"/>; never constructed directly by callers.
/// </summary>
public sealed class InferencePolicy
{
    private readonly CpuActorCritic _cpu;
    private readonly float[] _mean;

    /// <summary>The control layout this policy was trained for. The Godot layer matches it against
    /// the agent's live <see cref="IControlSurface"/> to catch model/agent mismatches before use.</summary>
    public ControlSpec Controls { get; }

    public int ObservationSize => _cpu.StateSize;
    public int ActionSize => _cpu.ActionSize;

    internal InferencePolicy(CpuActorCritic cpu, ControlSpec controls)
    {
        _cpu = cpu;
        Controls = controls;
        _mean = new float[cpu.ActionSize];
    }

    /// <summary>
    /// Greedy action for one observation: the policy mean clamped to the normalized [-1,1] range the
    /// policy was trained to emit (mapping to a channel's physical range is the control surface's job).
    /// Allocation-free — writes exactly <see cref="ActionSize"/> values into <paramref name="action"/>.
    /// Matches <see cref="Ppo.EvaluateMeanSteps"/>'s per-step action computation exactly.
    /// </summary>
    public void Act(ReadOnlySpan<float> observation, Span<float> action)
    {
        _cpu.Forward(observation, _mean, out _);
        for (int k = 0; k < _mean.Length; k++)
            action[k] = Math.Clamp(_mean[k], -1f, 1f);
    }

    /// <summary>Convenience overload returning a fresh action array; prefer the span overload on hot paths.</summary>
    public float[] Act(ReadOnlySpan<float> observation)
    {
        var action = new float[_cpu.ActionSize];
        Act(observation, action);
        return action;
    }
}
