using Xunit.Abstractions;

namespace Tensotron.Rl.Tests;

/// <summary>
/// The mixed continuous+discrete task end to end: a turret learns to <em>track</em> an orbiting target
/// (the continuous Aim channel) and to <em>shoot</em> it only when lined up (the discrete Fire channel),
/// through a single PPO policy over <see cref="TurretEnv"/>. This is the Godot turret example's learning
/// problem, proven Godot-free so the slow headless run is an integration check, not a debugging loop.
/// Asserts measurable learning on both channels, not full convergence.
/// </summary>
public class TurretLearningTests
{
    private readonly ITestOutputHelper _out;
    public TurretLearningTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Turret_learns_to_track_and_shoot()
    {
        const int maxSteps = 200;
        Init.Seed(4);
        var rng = new Random(4);

        var probe = new TurretEnv(rng, maxSteps);
        var ac = new ActorCritic(probe.ObservationSize, probe.Controls, hidden: 48);
        var ppo = new Ppo(ac, () => new TurretEnv(rng, maxSteps), rng)
        {
            NumEnvs = 16, Horizon = 64, Epochs = 4, MinibatchSize = 256, LearningRate = 3e-3f,
            EntropyCoef = 0.005f,
        };

        var before = Eval(ac, new Random(555), maxSteps, episodes: 20);
        ppo.Train(40, (i, r) => _out.WriteLine($"iter {i}: meanReward={r:0.00}"));
        var after = Eval(ac, new Random(555), maxSteps, episodes: 20);

        _out.WriteLine($"track cos {before.track:0.00}->{after.track:0.00}, " +
                       $"hits/ep {before.hits:0.0}->{after.hits:0.0}, hitRate {before.hitRate:0.00}->{after.hitRate:0.00}");

        Assert.True(after.track > 0.85f, $"turret did not learn to track (cos={after.track:0.00}).");
        Assert.True(after.hits > before.hits + 5f,
            $"turret did not learn to shoot (hits/ep {before.hits:0.0}->{after.hits:0.0}).");
        Assert.True(after.hitRate > 0.6f, $"turret wastes most of its shots (hitRate={after.hitRate:0.00}).");
    }

    // Greedy rollout metrics: mean cos(bearing error) over all steps (tracking), hits per episode, and
    // the fraction of shots that hit (shot discipline).
    private static (float track, float hits, float hitRate) Eval(
        ActorCritic ac, Random rng, int maxSteps, int episodes)
    {
        var cpu = ac.SnapshotCpu();
        var po = new float[ac.PolicyOutSize];
        var act = new float[ac.ActionSize];
        double cosSum = 0;
        int steps = 0, fires = 0, hits = 0;
        for (int ep = 0; ep < episodes; ep++)
        {
            var env = new TurretEnv(rng, maxSteps);
            var s = env.GetState();
            for (int t = 0; t < maxSteps; t++)
            {
                cpu.Forward(s, po, out _);
                ac.Layout.Greedy(po, act);
                var (_, done) = env.Step(act);
                cosSum += MathF.Cos(env.BearingError);
                steps++;
                if (env.LastFired) { fires++; if (env.LastHit) hits++; }
                s = env.GetState();
                if (done) break;
            }
        }
        return ((float)(cosSum / steps), (float)hits / episodes, fires > 0 ? (float)hits / fires : 0f);
    }
}
