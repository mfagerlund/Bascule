namespace Tensotron.Rl;

/// <summary>
/// The composition core that realizes interface-driven discovery: given a node's
/// <see cref="IObservationSource"/>s, <see cref="IControlSurface"/>s and <see cref="IRewardSource"/>s,
/// it computes the merged observation layout and <see cref="ControlSpec"/>, routes each step's action
/// vector to the right surfaces, and aggregates rewards — with no loop of its own and no notion of how
/// the world advances. That makes it usable by both the synchronous console/test path (via
/// <see cref="CompositeEnvironment"/>) and the Godot frame-driven path, which calls these same methods
/// from <c>_physics_process</c>. Godot-free by construction.
/// </summary>
public sealed class CompositeAgent
{
    private readonly IObservationSource[] _observations;
    private readonly IControlSurface[] _controls;
    private readonly IRewardSource[] _rewards;
    private readonly IEpisodeReset[] _resets;
    private readonly int[] _obsOffsets;     // start index of each observation source in the vector
    private readonly int[] _channelCounts;  // action-channel count of each control surface

    /// <summary>Total observation vector length (sum of every source's <see cref="IObservationSource.Size"/>).</summary>
    public int ObservationSize { get; }

    /// <summary>The merged control spec — every surface's channels concatenated in source order.</summary>
    public ControlSpec Controls { get; }

    /// <summary>Total action-vector length (== <c>Controls.Count</c>).</summary>
    public int ActionSize => Controls.Count;

    public CompositeAgent(
        IReadOnlyList<IObservationSource> observations,
        IReadOnlyList<IControlSurface> controls,
        IReadOnlyList<IRewardSource> rewards,
        IReadOnlyList<IEpisodeReset>? resets = null)
    {
        _observations = observations.ToArray();
        _controls = controls.ToArray();
        _rewards = rewards.ToArray();
        _resets = resets?.ToArray() ?? System.Array.Empty<IEpisodeReset>();

        _obsOffsets = new int[_observations.Length];
        int obsTotal = 0;
        for (int i = 0; i < _observations.Length; i++)
        {
            _obsOffsets[i] = obsTotal;
            obsTotal += _observations[i].Size;
        }
        ObservationSize = obsTotal;

        var channels = new List<ControlChannel>();
        _channelCounts = new int[_controls.Length];
        for (int i = 0; i < _controls.Length; i++)
        {
            var spec = _controls[i].Spec;
            _channelCounts[i] = spec.Count;
            channels.AddRange(spec.Channels);
        }
        Controls = new ControlSpec(channels.ToArray());
    }

    /// <summary>Gather the full observation vector by letting each source write its own slice.
    /// <paramref name="dst"/> must be at least <see cref="ObservationSize"/> long.</summary>
    public void WriteObservation(Span<float> dst)
    {
        for (int i = 0; i < _observations.Length; i++)
            _observations[i].Write(dst.Slice(_obsOffsets[i], _observations[i].Size));
    }

    /// <summary>Route the action vector to the control surfaces — each gets the contiguous slice of
    /// its own channels, in source order. <paramref name="action"/> length must be <see cref="ActionSize"/>.</summary>
    public void ApplyAction(ReadOnlySpan<float> action, float dt)
    {
        int offset = 0;
        for (int i = 0; i < _controls.Length; i++)
        {
            int n = _channelCounts[i];
            _controls[i].Apply(action.Slice(offset, n), dt);
            offset += n;
        }
    }

    /// <summary>Sum the reward across all reward sources for the current step.</summary>
    public float CollectReward()
    {
        float sum = 0f;
        for (int i = 0; i < _rewards.Length; i++) sum += _rewards[i].Reward;
        return sum;
    }

    /// <summary>True if any reward source signals episode termination.</summary>
    public bool IsDone()
    {
        for (int i = 0; i < _rewards.Length; i++)
            if (_rewards[i].Done) return true;
        return false;
    }

    /// <summary>Start a new episode: restore the world via every <see cref="IEpisodeReset"/> source,
    /// then clear per-episode bookkeeping on every reward source. World reset runs first so reward
    /// sources that read post-reset state see the fresh world.</summary>
    public void ResetEpisode()
    {
        for (int i = 0; i < _resets.Length; i++) _resets[i].ResetEpisode();
        for (int i = 0; i < _rewards.Length; i++) _rewards[i].ResetEpisode();
    }
}
