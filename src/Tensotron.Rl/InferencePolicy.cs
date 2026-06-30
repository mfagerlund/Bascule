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
    private readonly ActionLayout _layout;
    private readonly float[] _policyOut;

    /// <summary>The control layout this policy was trained for. The Godot layer matches it against
    /// the agent's live <see cref="IControlSurface"/> to catch model/agent mismatches before use.</summary>
    public ControlSpec Controls { get; }

    public int ObservationSize => _cpu.StateSize;
    /// <summary>Env-facing action-vector length (one float per channel).</summary>
    public int ActionSize => _layout.ChannelCount;

    internal InferencePolicy(CpuActorCritic cpu, ControlSpec controls)
    {
        _cpu = cpu;
        Controls = controls;
        _layout = new ActionLayout(controls);
        _policyOut = new float[cpu.PolicyOutSize];
    }

    /// <summary>
    /// Greedy action for one observation: continuous channels emit the policy mean clamped to [-1,1];
    /// discrete channels emit the argmax category index (mapping to a channel's physical range/effect is
    /// the control surface's job). Allocation-free — writes exactly <see cref="ActionSize"/> values into
    /// <paramref name="action"/>. Matches <see cref="ActionLayout.Greedy"/>.
    /// </summary>
    public void Act(ReadOnlySpan<float> observation, Span<float> action)
    {
        _cpu.Forward(observation, _policyOut, out _);
        _layout.Greedy(_policyOut, action);
    }

    /// <summary>Convenience overload returning a fresh action array; prefer the span overload on hot paths.</summary>
    public float[] Act(ReadOnlySpan<float> observation)
    {
        var action = new float[ActionSize];
        Act(observation, action);
        return action;
    }
}
