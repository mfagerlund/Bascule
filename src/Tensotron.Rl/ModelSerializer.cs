namespace Tensotron.Rl;

/// <summary>
/// Saves a trained <see cref="ActorCritic"/> to a single self-describing byte blob and loads it back
/// as an inference-ready <see cref="InferencePolicy"/>. The blob is everything needed to reconstruct
/// the policy with no external context: format header, network sizes, the <see cref="ControlSpec"/>,
/// and the weight state_dict.
///
/// The tensor-block layout matches the Tensotron engine's <c>Serialization</c> format, but this adds
/// the RL metadata the engine's tensor-only format can't carry. Lives in <c>Tensotron.Rl</c> and is
/// Godot-free: the editor's <c>ModelResource</c> (.tres) simply stores the returned <c>byte[]</c>.
/// </summary>
public static class ModelSerializer
{
    private const int Magic = 0x4D4C5254; // 'TRLM' — Tensotron RL Model
    private const int Version = 1;

    /// <summary>Serialize a trained network plus its control layout into a portable blob.</summary>
    public static byte[] Save(ActorCritic ac, ControlSpec controls)
    {
        if (controls.Count != ac.ActionSize)
            throw new ArgumentException(
                $"ControlSpec has {controls.Count} channels but the policy has {ac.ActionSize} action dims.",
                nameof(controls));

        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            w.Write(Magic);
            w.Write(Version);
            w.Write(ac.StateSize);
            w.Write(ac.ActionSize);
            w.Write(ac.Hidden);

            w.Write(controls.Count);
            foreach (var ch in controls.Channels)
            {
                w.Write(ch.Name);
                w.Write(ch.Min);
                w.Write(ch.Max);
                w.Write(ch.IsDiscrete);
            }

            var tensors = ac.NamedTensors().ToList();
            w.Write(tensors.Count);
            foreach (var (name, t) in tensors)
            {
                w.Write(name);
                w.Write(t.Rank);
                foreach (var d in t.Shape.Dims) w.Write(d);
                var data = t.ToArray();
                w.Write(data.Length);
                foreach (var f in data) w.Write(f);
            }
        }
        return ms.ToArray();
    }

    /// <summary>Reconstruct an inference-ready policy from a blob written by <see cref="Save"/>.</summary>
    public static InferencePolicy Load(byte[] data)
    {
        using var ms = new MemoryStream(data, writable: false);
        using var r = new BinaryReader(ms, System.Text.Encoding.UTF8);

        if (data.Length < 8 || r.ReadInt32() != Magic)
            throw new InvalidOperationException("Not a Tensotron.Rl model blob (bad magic).");
        int version = r.ReadInt32();
        if (version != Version)
            throw new InvalidOperationException($"Unsupported model version {version} (expected {Version}).");

        int stateSize = r.ReadInt32();
        int actionSize = r.ReadInt32();
        int hidden = r.ReadInt32();

        int channelCount = r.ReadInt32();
        var channels = new ControlChannel[channelCount];
        for (int i = 0; i < channelCount; i++)
        {
            string name = r.ReadString();
            float min = r.ReadSingle();
            float max = r.ReadSingle();
            bool isDiscrete = r.ReadBoolean();
            channels[i] = new ControlChannel(name, min, max, isDiscrete);
        }
        var controls = new ControlSpec(channels);

        int tensorCount = r.ReadInt32();
        var dict = new Dictionary<string, Tensor>(tensorCount);
        for (int i = 0; i < tensorCount; i++)
        {
            string name = r.ReadString();
            int rank = r.ReadInt32();
            var dims = new int[rank];
            for (int j = 0; j < rank; j++) dims[j] = r.ReadInt32();
            int len = r.ReadInt32();
            var arr = new float[len];
            for (int j = 0; j < len; j++) arr[j] = r.ReadSingle();
            dict[name] = Tensor.FromShaped(arr, dims);
        }

        // Rebuild the same architecture from the saved control layout (so discrete channels restore the
        // right policy-head width), copy the saved weights in by name, then take a host snapshot for
        // launch-free inference. Reusing ActorCritic keeps the layer wiring in one place.
        if (controls.Count != actionSize)
            throw new InvalidOperationException(
                $"Model action size {actionSize} disagrees with its {controls.Count}-channel control spec.");
        var ac = new ActorCritic(stateSize, controls, hidden);
        ac.LoadState(dict);
        return new InferencePolicy(ac.SnapshotCpu(), controls);
    }

    /// <summary>Convenience: save directly to a file (e.g. a headless trainer dumping a <c>.trlm</c>).</summary>
    public static void SaveToFile(ActorCritic ac, ControlSpec controls, string path)
        => File.WriteAllBytes(path, Save(ac, controls));

    /// <summary>Convenience: load a policy from a file written by <see cref="SaveToFile"/>.</summary>
    public static InferencePolicy LoadFromFile(string path)
        => Load(File.ReadAllBytes(path));
}
