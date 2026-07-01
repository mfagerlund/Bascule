namespace Bascule.RL.Tests;

/// <summary>
/// Proves the train → save → load → infer round-trip that makes Inference mode shippable: a saved
/// model reconstructs a byte-identical greedy policy from its blob alone, carries its
/// <see cref="ControlSpec"/> and sizes with it, and round-trips through a file. All Godot-free.
/// </summary>
public class ModelSerializerTests
{
    private static ActorCritic MakeTrained(out ControlSpec controls, int seed = 7)
    {
        Init.Seed(seed);
        var rng = new Random(seed);
        var probe = new SinglePoleCart(rng, 200);
        controls = probe.Controls;
        var ac = new ActorCritic(probe.ObservationSize, probe.Controls.Count, hidden: 16);
        var ppo = new Ppo(ac, () => new SinglePoleCart(rng, 200), rng)
        {
            NumEnvs = 8, Horizon = 64, Epochs = 3, MinibatchSize = 256, LearningRate = 3e-3f,
        };
        ppo.Train(1);
        return ac;
    }

    [Fact]
    public void Roundtrip_reproduces_greedy_actions_exactly()
    {
        var ac = MakeTrained(out var controls);
        var reference = ac.SnapshotCpu(); // the same host policy the trainer rolls out with

        var policy = ModelSerializer.Load(ModelSerializer.Save(ac, controls));

        var rng = new Random(123);
        var expected = new float[ac.ActionSize];
        var actual = new float[ac.ActionSize];
        for (int trial = 0; trial < 50; trial++)
        {
            var obs = new float[ac.StateSize];
            for (int i = 0; i < obs.Length; i++) obs[i] = (float)(rng.NextDouble() * 2 - 1);

            reference.Forward(obs, expected, out _);
            for (int k = 0; k < expected.Length; k++) expected[k] = MathF.Tanh(expected[k]);   // greedy = tanh-bounded mean

            policy.Act(obs, actual);
            for (int k = 0; k < actual.Length; k++)
                Assert.Equal(expected[k], actual[k], 5);
        }
    }

    [Fact]
    public void Roundtrip_reproduces_a_mixed_continuous_discrete_policy()
    {
        // A mixed head (discrete Fire first, then continuous Aim) reconstructs from the saved control
        // spec alone — wider policy layer, logstd sized to the continuous count — and reproduces greedy
        // actions exactly: continuous clamped, discrete argmax.
        Init.Seed(3);
        var rng = new Random(3);
        var probe = new AimAndFire(rng);
        var controls = probe.Controls;
        var ac = new ActorCritic(probe.ObservationSize, controls, hidden: 16);
        new Ppo(ac, () => new AimAndFire(rng), rng) { NumEnvs = 8, Horizon = 16, Epochs = 2 }.Train(1);

        var reference = ac.SnapshotCpu();
        var refLayout = ac.Layout;
        var policy = ModelSerializer.Load(ModelSerializer.Save(ac, controls));
        Assert.Equal(ac.ActionSize, policy.ActionSize);   // env-facing channel count (2), not head width (3)

        var po = new float[ac.PolicyOutSize];
        var expected = new float[ac.ActionSize];
        var actual = new float[ac.ActionSize];
        var evalRng = new Random(456);
        for (int trial = 0; trial < 50; trial++)
        {
            var obs = new[] { (float)(evalRng.NextDouble() * 2 - 1), evalRng.NextDouble() < 0.5 ? 0f : 1f };
            reference.Forward(obs, po, out _);
            refLayout.Greedy(po, expected);

            policy.Act(obs, actual);
            Assert.Equal(expected[0], actual[0], 5);  // discrete category index
            Assert.Equal(expected[1], actual[1], 5);  // continuous clamped mean
        }
    }

    [Fact]
    public void Roundtrip_preserves_metadata_and_controlspec()
    {
        var ac = MakeTrained(out var controls);
        var policy = ModelSerializer.Load(ModelSerializer.Save(ac, controls));

        Assert.Equal(ac.StateSize, policy.ObservationSize);
        Assert.Equal(ac.ActionSize, policy.ActionSize);
        Assert.Equal(controls.Count, policy.Controls.Count);
        for (int i = 0; i < controls.Count; i++)
        {
            var src = controls.Channels[i];
            var dst = policy.Controls.Channels[i];
            Assert.Equal(src.Name, dst.Name);
            Assert.Equal(src.Min, dst.Min);
            Assert.Equal(src.Max, dst.Max);
            Assert.Equal(src.IsDiscrete, dst.IsDiscrete);
        }
    }

    [Fact]
    public void File_roundtrip_loads_a_working_policy()
    {
        var ac = MakeTrained(out var controls);
        string path = Path.Combine(Path.GetTempPath(), $"trlm_test_{Guid.NewGuid():N}.trlm");
        try
        {
            ModelSerializer.SaveToFile(ac, controls, path);
            Assert.True(File.Exists(path));

            var policy = ModelSerializer.LoadFromFile(path);
            Assert.Equal(ac.ActionSize, policy.ActionSize);

            var action = policy.Act(new float[ac.StateSize]); // zero obs is a valid input
            Assert.Equal(ac.ActionSize, action.Length);
            Assert.All(action, a => Assert.InRange(a, -1f, 1f));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Save_rejects_controlspec_that_does_not_match_the_policy()
    {
        var ac = MakeTrained(out _);
        var wrong = new ControlSpec(new[]
        {
            new ControlChannel("A", -1f, 1f),
            new ControlChannel("B", -1f, 1f),
        });
        Assert.Throws<ArgumentException>(() => ModelSerializer.Save(ac, wrong));
    }

    [Fact]
    public void Load_rejects_non_model_bytes()
        => Assert.Throws<InvalidOperationException>(
            () => ModelSerializer.Load(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }));
}
