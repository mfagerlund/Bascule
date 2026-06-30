using Xunit.Abstractions;

namespace Tensotron.Rl.Tests;

/// <summary>
/// Proves PPO learns through the <em>discrete</em> (categorical) action head added with
/// <see cref="ActionLayout"/> — the turret's Fire channel depends on it. Three angles: a pure-discrete
/// bandit through the synchronous <see cref="Ppo"/>, a mixed continuous+discrete bandit (discrete first,
/// to catch column-offset bugs), and a pure-discrete bandit through the inverted
/// <see cref="BatchedPpoTrainer"/> tick loop (Godot's path). All use a non-zero entropy coefficient so
/// the categorical-entropy term in the update is exercised, and assert measurable learning, not full
/// convergence, to stay fast and stable in the default suite.
/// </summary>
public class DiscretePpoTests
{
    private readonly ITestOutputHelper _out;
    public DiscretePpoTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Discrete_ppo_learns_a_categorical_decision()
    {
        Init.Seed(7);
        var rng = new Random(7);

        var probe = new SignMatchBandit(rng);
        var ac = new ActorCritic(probe.ObservationSize, probe.Controls, hidden: 32);
        var ppo = new Ppo(ac, () => new SignMatchBandit(rng), rng)
        {
            NumEnvs = 16, Horizon = 16, Epochs = 4, MinibatchSize = 128, LearningRate = 5e-3f,
            EntropyCoef = 0.01f,
        };

        float before = Accuracy(ac, new Random(999), 1000);
        ppo.Train(15, (i, r) => _out.WriteLine($"iter {i}: meanReward={r:0.00}"));
        float after = Accuracy(ac, new Random(999), 1000);

        _out.WriteLine($"discrete accuracy before={before:0.00}, after={after:0.00}");
        Assert.True(after > 0.85f, $"discrete policy did not learn (accuracy after={after:0.00}).");
        Assert.True(after > before + 0.2f, $"no learning signal (before={before:0.00}, after={after:0.00}).");
    }

    [Fact]
    public void Mixed_continuous_and_discrete_learn_together()
    {
        Init.Seed(5);
        var rng = new Random(5);

        var probe = new AimAndFire(rng);
        var ac = new ActorCritic(probe.ObservationSize, probe.Controls, hidden: 32);
        var ppo = new Ppo(ac, () => new AimAndFire(rng), rng)
        {
            NumEnvs = 16, Horizon = 16, Epochs = 4, MinibatchSize = 128, LearningRate = 5e-3f,
            EntropyCoef = 0.01f,
        };

        var (fireBefore, aimErrBefore) = EvalMixed(ac, new Random(321), 1000);
        ppo.Train(25, (i, r) => _out.WriteLine($"iter {i}: meanReward={r:0.00}"));
        var (fireAfter, aimErrAfter) = EvalMixed(ac, new Random(321), 1000);

        _out.WriteLine($"fire acc {fireBefore:0.00}->{fireAfter:0.00}, aim |err| {aimErrBefore:0.00}->{aimErrAfter:0.00}");
        Assert.True(fireAfter > 0.8f, $"discrete channel did not learn (fire acc={fireAfter:0.00}).");
        Assert.True(aimErrAfter < aimErrBefore - 0.1f,
            $"continuous channel did not improve (|err| {aimErrBefore:0.00}->{aimErrAfter:0.00}).");
    }

    [Fact]
    public void Batched_trainer_learns_a_discrete_decision()
    {
        const int n = 16, horizon = 16, targetIterations = 15;
        Init.Seed(11);
        var rng = new Random(11);

        var controls = new ControlSpec(new[] { new ControlChannel("Fire", 0f, 1f, IsDiscrete: true) });
        var ac = new ActorCritic(SignMatchBandit.Obs, controls, hidden: 32);
        var trainer = new BatchedPpoTrainer(ac, controls, agentCount: n, horizon: horizon, rng)
        {
            Epochs = 4, MinibatchSize = 128, LearningRate = 5e-3f, EntropyCoef = 0.01f,
        };

        int S = SignMatchBandit.Obs, A = 1;
        var ctx = new float[n][];
        var obs = new float[n * S];
        var rew = new float[n];
        var done = new float[n];
        var act = new float[n * A];

        var worldRng = new Random(123);
        void Roll(int e)
        {
            ctx[e] = new float[S];
            for (int j = 0; j < S; j++) ctx[e][j] = (float)(worldRng.NextDouble() * 2.0 - 1.0);
        }
        for (int e = 0; e < n; e++) { Roll(e); Array.Copy(ctx[e], 0, obs, e * S, S); }

        float before = Accuracy(ac, new Random(999), 1000);

        int safety = 0, cap = targetIterations * horizon * 3;
        while (trainer.Iteration < targetIterations && safety++ < cap)
        {
            trainer.Tick(obs, rew, done, act);
            for (int e = 0; e < n; e++)
            {
                int choice = (int)MathF.Round(act[e * A]);
                rew[e] = choice == SignMatchBandit.Target(ctx[e]) ? 1f : 0f;
                done[e] = 1f;                 // one-step bandit: every transition is terminal
                Roll(e);                      // auto-reset to a fresh context
                Array.Copy(ctx[e], 0, obs, e * S, S);
            }
        }

        float after = Accuracy(ac, new Random(999), 1000);

        _out.WriteLine($"batched discrete accuracy before={before:0.00}, after={after:0.00}");
        Assert.True(trainer.Iteration >= targetIterations,
            $"trainer completed only {trainer.Iteration}/{targetIterations} iterations.");
        Assert.True(after > 0.8f && after > before + 0.2f,
            $"batched discrete policy did not learn (before={before:0.00}, after={after:0.00}).");
    }

    // Greedy categorical accuracy on fresh random contexts (SignMatchBandit rule).
    private static float Accuracy(ActorCritic ac, Random rng, int samples)
    {
        var cpu = ac.SnapshotCpu();
        var policyOut = new float[ac.PolicyOutSize];
        var act = new float[ac.ActionSize];
        var ctx = new float[SignMatchBandit.Obs];
        int correct = 0;
        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < ctx.Length; j++) ctx[j] = (float)(rng.NextDouble() * 2.0 - 1.0);
            cpu.Forward(ctx, policyOut, out _);
            ac.Layout.Greedy(policyOut, act);
            if ((int)act[0] == SignMatchBandit.Target(ctx)) correct++;
        }
        return (float)correct / samples;
    }

    // Greedy evaluation of the mixed AimAndFire policy: fire accuracy and mean |aim - target|.
    private static (float fireAcc, float aimErr) EvalMixed(ActorCritic ac, Random rng, int samples)
    {
        var cpu = ac.SnapshotCpu();
        var policyOut = new float[ac.PolicyOutSize];
        var act = new float[ac.ActionSize];
        int fireCorrect = 0;
        float errSum = 0f;
        for (int i = 0; i < samples; i++)
        {
            float t = (float)(rng.NextDouble() * 2.0 - 1.0);
            int g = rng.NextDouble() < 0.5 ? 0 : 1;
            cpu.Forward(new[] { t, (float)g }, policyOut, out _);
            ac.Layout.Greedy(policyOut, act);
            if ((int)act[0] == g) fireCorrect++;
            errSum += MathF.Abs(act[1] - t);
        }
        return ((float)fireCorrect / samples, errSum / samples);
    }
}
