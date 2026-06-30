namespace Tensotron.Rl;

/// <summary>
/// Gaussian actor + value critic for continuous PPO. The actor maps state → action mean
/// (a small tanh MLP) and carries a state-independent learnable log-σ per action dim
/// (standard PPO practice). The critic is a separate tanh MLP mapping state → scalar value.
/// Adapted verbatim from the Tensotron engine showcase (Rl/ActorCritic.cs).
/// </summary>
public sealed class ActorCritic
{
    private readonly Sequential _policy;
    private readonly Sequential _value;
    private readonly Linear[] _policyLin;
    private readonly Linear[] _valueLin;

    /// <summary>Learnable log standard deviation, one per action dim (state-independent).</summary>
    public Tensor LogStd { get; }

    public int StateSize { get; }
    public int ActionSize { get; }
    public int Hidden { get; }

    public ActorCritic(int stateSize, int actionSize, int hidden = 64, float initLogStd = -0.5f)
    {
        StateSize = stateSize;
        ActionSize = actionSize;
        Hidden = hidden;

        // Keep the Linear instances so their weights can be snapshotted to a CPU mirror for
        // launch-free rollout inference (see SnapshotCpu); the Sequentials reuse the same ones.
        _policyLin = new[] { new Linear(stateSize, hidden), new Linear(hidden, hidden), new Linear(hidden, actionSize) };
        _policy = new Sequential(_policyLin[0], Activation.Tanh(), _policyLin[1], Activation.Tanh(), _policyLin[2]);

        _valueLin = new[] { new Linear(stateSize, hidden), new Linear(hidden, hidden), new Linear(hidden, 1) };
        _value = new Sequential(_valueLin[0], Activation.Tanh(), _valueLin[1], Activation.Tanh(), _valueLin[2]);

        var logStd = new float[actionSize];
        Array.Fill(logStd, initLogStd);
        LogStd = Tensor.FromShaped(logStd, new[] { actionSize }).RequireGrad();
    }

    /// <summary>Action mean for a batch of states: (B, stateSize) → (B, actionSize).</summary>
    public Tensor PolicyMean(Tensor states) => _policy.Forward(states);

    /// <summary>Value estimate for a batch of states: (B, stateSize) → (B,).</summary>
    public Tensor Value(Tensor states) => _value.Forward(states).Reshape(states.Shape.Dims[0]);

    public IReadOnlyList<Tensor> Parameters()
    {
        var ps = new List<Tensor>();
        ps.AddRange(_policy.Parameters());
        ps.AddRange(_value.Parameters());
        ps.Add(LogStd);
        return ps;
    }

    /// <summary>Named state_dict for serialization. Reuses the engine's <c>Module.StateDict</c>
    /// naming under stable prefixes (<c>policy.</c> / <c>value.</c>) and adds the standalone
    /// <c>logstd</c> tensor. Saving and loading both build the network the same way, so the names
    /// line up by construction.</summary>
    public IEnumerable<(string name, Tensor tensor)> NamedTensors()
    {
        foreach (var kv in _policy.StateDict("policy.")) yield return kv;
        foreach (var kv in _value.StateDict("value.")) yield return kv;
        yield return ("logstd", LogStd);
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
        => new(MakeCpuMlp(_policyLin), MakeCpuMlp(_valueLin), StateSize, ActionSize);

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
