using Xunit.Abstractions;

namespace Tensotron.Rl.Tests;

/// <summary>
/// Verifies interface-driven discovery: that <see cref="CompositeAgent"/> stitches independent
/// observation/control/reward components into one coherent environment surface (correct layout,
/// channel routing, reward aggregation) and that the existing <see cref="Ppo"/> trainer actually
/// learns through the composed <see cref="CompositeEnvironment"/>.
/// </summary>
public class CompositeAgentTests
{
    private readonly ITestOutputHelper _out;
    public CompositeAgentTests(ITestOutputHelper output) => _out = output;

    private sealed class FakeObs : IObservationSource
    {
        private readonly float[] _vals;
        public FakeObs(params float[] vals) => _vals = vals;
        public int Size => _vals.Length;
        public void Write(Span<float> dst) => _vals.CopyTo(dst);
    }

    private sealed class FakeControl : IControlSurface
    {
        public ControlSpec Spec { get; }
        public float[]? LastAction;
        public float LastDt;
        public FakeControl(params string[] channels)
            => Spec = new ControlSpec(channels.Select(n => new ControlChannel(n, -1f, 1f)).ToArray());
        public void Apply(ReadOnlySpan<float> action, float dt)
        {
            LastAction = action.ToArray();
            LastDt = dt;
        }
    }

    private sealed class FakeReward : IRewardSource
    {
        public float Reward { get; set; }
        public bool Done { get; set; }
        public int ResetCount;
        public void ResetEpisode() => ResetCount++;
    }

    [Fact]
    public void Concatenates_observations_in_source_order()
    {
        var agent = new CompositeAgent(
            new IObservationSource[] { new FakeObs(1f, 2f), new FakeObs(3f, 4f, 5f) },
            Array.Empty<IControlSurface>(),
            Array.Empty<IRewardSource>());

        Assert.Equal(5, agent.ObservationSize);

        var buf = new float[5];
        agent.WriteObservation(buf);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f }, buf);
    }

    [Fact]
    public void Merges_control_specs_preserving_channel_order()
    {
        var agent = new CompositeAgent(
            Array.Empty<IObservationSource>(),
            new IControlSurface[] { new FakeControl("Yaw"), new FakeControl("Pitch", "Fire") },
            Array.Empty<IRewardSource>());

        Assert.Equal(3, agent.ActionSize);
        Assert.Equal(new[] { "Yaw", "Pitch", "Fire" }, agent.Controls.Channels.Select(c => c.Name));
    }

    [Fact]
    public void Routes_each_control_surface_its_own_action_slice()
    {
        var a = new FakeControl("Yaw");
        var b = new FakeControl("Pitch", "Fire");
        var agent = new CompositeAgent(
            Array.Empty<IObservationSource>(),
            new IControlSurface[] { a, b },
            Array.Empty<IRewardSource>());

        agent.ApplyAction(new[] { 0.1f, 0.2f, 0.3f }, dt: 0.5f);

        Assert.Equal(new[] { 0.1f }, a.LastAction);
        Assert.Equal(new[] { 0.2f, 0.3f }, b.LastAction);
        Assert.Equal(0.5f, a.LastDt);
        Assert.Equal(0.5f, b.LastDt);
    }

    [Fact]
    public void Sums_reward_and_ORs_done_across_sources()
    {
        var r1 = new FakeReward { Reward = 0.5f, Done = false };
        var r2 = new FakeReward { Reward = 0.25f, Done = true };
        var agent = new CompositeAgent(
            Array.Empty<IObservationSource>(),
            Array.Empty<IControlSurface>(),
            new IRewardSource[] { r1, r2 });

        Assert.Equal(0.75f, agent.CollectReward());
        Assert.True(agent.IsDone());

        r2.Done = false;
        Assert.False(agent.IsDone());
    }

    [Fact]
    public void ResetEpisode_propagates_to_every_reward_source()
    {
        var r1 = new FakeReward();
        var r2 = new FakeReward();
        var agent = new CompositeAgent(
            Array.Empty<IObservationSource>(),
            Array.Empty<IControlSurface>(),
            new IRewardSource[] { r1, r2 });

        agent.ResetEpisode();
        agent.ResetEpisode();

        Assert.Equal(2, r1.ResetCount);
        Assert.Equal(2, r2.ResetCount);
    }

    [Fact]
    public void Ppo_learns_through_a_composed_environment()
    {
        const int maxSteps = 150;
        Init.Seed(1);
        var rng = new Random(1);

        var probe = CompositeCartPole.Create(rng, maxSteps);
        var ac = new ActorCritic(probe.ObservationSize, probe.Controls.Count, hidden: 32);
        var ppo = new Ppo(ac, () => CompositeCartPole.Create(rng, maxSteps), rng)
        {
            NumEnvs = 8, Horizon = 64, Epochs = 3, MinibatchSize = 256, LearningRate = 3e-3f,
        };

        float before = ppo.EvaluateMeanSteps(episodes: 3, maxSteps: maxSteps);
        ppo.Train(5, (i, ret) => _out.WriteLine($"iter {i}: meanReturn={ret:0.0}"));
        float after = ppo.EvaluateMeanSteps(episodes: 3, maxSteps: maxSteps);

        _out.WriteLine($"composed meanSteps before={before:0.0}, after={after:0.0} (/{maxSteps})");
        Assert.True(after > before + 10f,
            $"PPO showed no learning signal through composition (before={before:0.0}, after={after:0.0}).");
    }
}
