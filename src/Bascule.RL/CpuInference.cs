namespace Bascule.RL;

/// <summary>
/// Plain-CPU forward pass for a tanh MLP (tanh on every layer except the last). Mirrors
/// Tensotron's <c>Linear</c> exactly (y = x·Wᵀ + b, Weight row-major (out,in)) but runs as
/// ordinary scalar C# — no accelerator launch or Synchronize per call. PPO rollouts are
/// hundreds of tiny sequential forwards; doing them on the device means hundreds of
/// launch+sync round-trips (the measured bottleneck). Weights are snapshotted once per PPO
/// iteration and this evaluator drives every rollout step from host memory.
/// Adapted verbatim from the Tensotron engine showcase (Rl/CpuInference.cs).
/// </summary>
internal sealed class CpuMlp
{
    private readonly float[][] _w;   // [layer] row-major (out, in)
    private readonly float[][] _b;   // [layer] (out,)
    private readonly int[] _in;
    private readonly int[] _out;
    private float[] _bufA;
    private float[] _bufB;

    public int InSize => _in[0];
    public int OutSize => _out[^1];

    public CpuMlp(float[][] w, float[][] b, int[] inDims, int[] outDims)
    {
        _w = w; _b = b; _in = inDims; _out = outDims;
        int wide = inDims[0];
        foreach (var o in outDims) wide = Math.Max(wide, o);
        _bufA = new float[wide];
        _bufB = new float[wide];
    }

    /// <summary>Forward one sample. <paramref name="result"/> must hold at least <see cref="OutSize"/>.</summary>
    public void Forward(ReadOnlySpan<float> x, Span<float> result)
    {
        int layers = _w.Length;
        x.Slice(0, _in[0]).CopyTo(_bufA);
        float[] cur = _bufA, nxt = _bufB;
        for (int l = 0; l < layers; l++)
        {
            int inD = _in[l], outD = _out[l];
            float[] w = _w[l], b = _b[l];
            bool activate = l < layers - 1; // tanh between layers, identity on the last
            for (int o = 0; o < outD; o++)
            {
                float s = b[o];
                int baseIdx = o * inD;
                for (int i = 0; i < inD; i++) s += cur[i] * w[baseIdx + i];
                nxt[o] = activate ? MathF.Tanh(s) : s;
            }
            (cur, nxt) = (nxt, cur);
        }
        cur.AsSpan(0, OutSize).CopyTo(result);
    }
}

/// <summary>
/// Host-side snapshot of an <see cref="ActorCritic"/>'s weights for launch-free PPO rollout
/// inference. Produced by <see cref="ActorCritic.SnapshotCpu"/> once per iteration; numerically
/// matches the device forward to float tolerance (same Linear math), so the rollout policy and
/// the policy re-evaluated on-device during the update agree (ratio starts at ~1).
/// </summary>
public sealed class CpuActorCritic
{
    private readonly CpuMlp _policy;
    private readonly CpuMlp _value;
    private readonly float[] _valueBuf = new float[1];

    public int StateSize { get; }
    /// <summary>Width of the raw policy head output (continuous means + discrete logits).</summary>
    public int PolicyOutSize { get; }

    internal CpuActorCritic(CpuMlp policy, CpuMlp value, int stateSize, int policyOutSize)
    {
        _policy = policy; _value = value;
        StateSize = stateSize; PolicyOutSize = policyOutSize;
    }

    /// <summary>Raw policy head (into <paramref name="policyOut"/>, length >= PolicyOutSize) and value
    /// for one state. Turning the head into an env-facing action is <see cref="ActionLayout"/>'s job.</summary>
    public void Forward(ReadOnlySpan<float> state, Span<float> policyOut, out float value)
    {
        _policy.Forward(state, policyOut);
        _value.Forward(state, _valueBuf);
        value = _valueBuf[0];
    }
}
