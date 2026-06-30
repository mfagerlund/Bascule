namespace Tensotron.Rl;

/// <summary>
/// Actor + value critic for PPO. The actor is a small tanh MLP mapping state → a policy head of width
/// <see cref="ActionLayout.PolicyOutSize"/>: one mean column per continuous channel plus a
/// state-independent learnable log-σ (<see cref="LogStd"/>, one per continuous channel), and K logit
/// columns per discrete channel. The critic is a separate tanh MLP mapping state → a scalar value.
/// All knowledge of which columns are which lives in <see cref="Layout"/>. Adapted from the Tensotron
/// engine showcase (Rl/ActorCritic.cs), widened from a single continuous action to the mixed
/// continuous/discrete <see cref="ControlSpec"/> shape.
/// </summary>
public sealed class ActorCritic
{
    private readonly Sequential _policy;
    private readonly Sequential _value;
    private readonly Linear[] _policyLin;
    private readonly Linear[] _valueLin;

    /// <summary>Learnable log standard deviation, one per <em>continuous</em> action dim
    /// (state-independent). Length is <see cref="ActionLayout.ContinuousCount"/> (empty if all discrete).</summary>
    public Tensor LogStd { get; }

    /// <summary>The control layout: channel→column mapping plus the sample/greedy/score math.</summary>
    public ActionLayout Layout { get; }

    public int StateSize { get; }
    /// <summary>Env-facing action-vector length (one float per channel == <c>Controls.Count</c>).</summary>
    public int ActionSize => Layout.ChannelCount;
    /// <summary>Width of the policy network output (≥ <see cref="ActionSize"/> when discrete channels exist).</summary>
    public int PolicyOutSize => Layout.PolicyOutSize;
    public int Hidden { get; }

    /// <summary>Build from an explicit control layout (the path that supports discrete channels).</summary>
    public ActorCritic(int stateSize, ControlSpec controls, int hidden = 64, float initLogStd = -0.5f)
    {
        StateSize = stateSize;
        Hidden = hidden;
        Layout = new ActionLayout(controls);

        // Keep the Linear instances so their weights can be snapshotted to a CPU mirror for
        // launch-free rollout inference (see SnapshotCpu); the Sequentials reuse the same ones.
        _policyLin = new[] { new Linear(stateSize, hidden), new Linear(hidden, hidden), new Linear(hidden, PolicyOutSize) };
        _policy = new Sequential(_policyLin[0], Activation.Tanh(), _policyLin[1], Activation.Tanh(), _policyLin[2]);

        _valueLin = new[] { new Linear(stateSize, hidden), new Linear(hidden, hidden), new Linear(hidden, 1) };
        _value = new Sequential(_valueLin[0], Activation.Tanh(), _valueLin[1], Activation.Tanh(), _valueLin[2]);

        int c = Layout.ContinuousCount;
        if (c > 0)
        {
            var logStd = new float[c];
            Array.Fill(logStd, initLogStd);
            LogStd = Tensor.FromShaped(logStd, new[] { c }).RequireGrad();
        }
        else
        {
            LogStd = Tensor.FromShaped(System.Array.Empty<float>(), new[] { 0 });
        }
    }

    /// <summary>Build an all-continuous network with <paramref name="actionSize"/> channels in [-1,1]
    /// (the cart-pole path; preserves the original behaviour exactly).</summary>
    public ActorCritic(int stateSize, int actionSize, int hidden = 64, float initLogStd = -0.5f)
        : this(stateSize, AllContinuous(actionSize), hidden, initLogStd)
    {
    }

    private static ControlSpec AllContinuous(int actionSize)
    {
        var channels = new ControlChannel[actionSize];
        for (int k = 0; k < actionSize; k++) channels[k] = new ControlChannel($"c{k}", -1f, 1f);
        return new ControlSpec(channels);
    }

    /// <summary>Raw policy head for a batch of states: (B, stateSize) → (B, <see cref="PolicyOutSize"/>).
    /// Continuous means and discrete logits interleaved per <see cref="Layout"/>.</summary>
    public Tensor PolicyOutput(Tensor states) => _policy.Forward(states);

    /// <summary>Value estimate for a batch of states: (B, stateSize) → (B,).</summary>
    public Tensor Value(Tensor states) => _value.Forward(states).Reshape(states.Shape.Dims[0]);

    public IReadOnlyList<Tensor> Parameters()
    {
        var ps = new List<Tensor>();
        ps.AddRange(_policy.Parameters());
        ps.AddRange(_value.Parameters());
        if (Layout.ContinuousCount > 0) ps.Add(LogStd);
        return ps;
    }

    /// <summary>Named state_dict for serialization. Reuses the engine's <c>Module.StateDict</c>
    /// naming under stable prefixes (<c>policy.</c> / <c>value.</c>) and adds the standalone
    /// <c>logstd</c> tensor when there are continuous channels. Saving and loading both build the
    /// network from the same <see cref="ControlSpec"/>, so the names and shapes line up by construction.</summary>
    public IEnumerable<(string name, Tensor tensor)> NamedTensors()
    {
        foreach (var kv in _policy.StateDict("policy.")) yield return kv;
        foreach (var kv in _value.StateDict("value.")) yield return kv;
        if (Layout.ContinuousCount > 0) yield return ("logstd", LogStd);
    }

    /// <summary>Copy weights from a name → tensor map into this network's parameters in place,
    /// preserving leaf identity (so an optimizer already bound to the params keeps working).
    /// Mirrors the engine's <c>Serialization.Load</c>; throws on any missing or mismatched param.</summary>
    public void LoadState(IReadOnlyDictionary<string, Tensor> src)
    {
        foreach (var (name, p) in NamedTensors())
        {
            if (!src.TryGetValue(name, out var s))
                throw new InvalidOperationException($"Missing parameter '{name}' in model data.");
            if (!s.Shape.Equals(p.Shape))
                throw new InvalidOperationException($"Parameter '{name}' shape {s.Shape} != model {p.Shape}.");
            p.Copy_(s); // torch-style in-place copy; preserves leaf identity
        }
    }

    /// <summary>
    /// Snapshot the current policy + value weights into a host-side <see cref="CpuActorCritic"/>
    /// for launch-free rollout inference. Pulls each layer to host once (cheap, small nets);
    /// numerically matches the device forward. Take a fresh snapshot per PPO iteration so the
    /// rollout uses the just-updated weights.
    /// </summary>
    public CpuActorCritic SnapshotCpu()
        => new(MakeCpuMlp(_policyLin), MakeCpuMlp(_valueLin), StateSize, PolicyOutSize);

    private static CpuMlp MakeCpuMlp(Linear[] lins)
    {
        int n = lins.Length;
        var w = new float[n][];
        var b = new float[n][];
        var inD = new int[n];
        var outD = new int[n];
        for (int i = 0; i < n; i++)
        {
            w[i] = lins[i].Weight.ToArray();           // row-major (out, in)
            b[i] = lins[i].Bias!.ToArray();            // ActorCritic always uses biased Linears
            outD[i] = lins[i].Weight.Shape.Dims[0];
            inD[i] = lins[i].Weight.Shape.Dims[1];
        }
        return new CpuMlp(w, b, inD, outD);
    }
}
