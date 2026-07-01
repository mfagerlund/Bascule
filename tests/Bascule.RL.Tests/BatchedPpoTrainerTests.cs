using Xunit.Abstractions;

namespace Bascule.RL.Tests;

/// <summary>
/// The inverted-control counterpart to <see cref="PpoLearningTests"/>: the same cart-pole learns, but
/// through <see cref="BatchedPpoTrainer"/>'s tick API — the host owns the loop, advances every arena
/// per tick, and feeds observations plus the previous tick's reward/done back in. This is the exact
/// shape Godot's <c>_physics_process</c> drives, exercised without an editor. Asserts measurable
/// improvement (not full convergence) so it stays fast and stable in the default suite.
/// </summary>
public class BatchedPpoTrainerTests
{
    private readonly ITestOutputHelper _out;
    public BatchedPpoTrainerTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Batched_trainer_learns_through_the_tick_loop()
    {
        const int maxSteps = 150;
        const int n = 8;
        const int horizon = 64;
        const int targetIterations = 6;
        Init.Seed(2);
        var rng = new Random(2);

        // Size the network/controls from one probe arena.
        var (_, probe) = CompositeCartPole.CreateRaw(rng, maxSteps);
        var ac = new ActorCritic(probe.ObservationSize, probe.ActionSize, hidden: 32);
        var trainer = new BatchedPpoTrainer(ac, probe.Controls, agentCount: n, horizon: horizon, rng)
        {
            Epochs = 3,
            MinibatchSize = 256,
            LearningRate = 3e-3f,
        };
        trainer.OnIterationComplete += s =>
            _out.WriteLine($"iter {s.Iteration}: meanReturn={s.MeanEpisodeReturn:0.0} " +
                           $"(eps={s.EpisodesCompleted}, steps={s.TotalAgentSteps})");

        // The N arenas the host advances.
        var worlds = new CartPoleWorld[n];
        var agents = new CompositeAgent[n];
        for (int e = 0; e < n; e++)
        {
            var (w, ag) = CompositeCartPole.CreateRaw(rng, maxSteps);
            worlds[e] = w;
            agents[e] = ag;
            w.Reset();
            ag.ResetEpisode();
        }

        int s = trainer.ObservationSize, a = trainer.ActionSize;
        var obs = new float[n * s];
        var rew = new float[n];
        var done = new float[n];
        var act = new float[n * a];
        for (int e = 0; e < n; e++) agents[e].WriteObservation(obs.AsSpan(e * s, s));

        float before = EvaluateGreedy(ac, maxSteps, episodes: 5);

        // Drive the inverted loop exactly as the Godot LearningAgent will: act -> advance -> observe
        // reward/done -> auto-reset -> re-observe. reward/done passed to the next Tick describe the
        // action applied this tick.
        int safety = 0, safetyCap = targetIterations * horizon * 3;
        while (trainer.Iteration < targetIterations && safety++ < safetyCap)
        {
            trainer.Tick(obs, rew, done, act);
            for (int e = 0; e < n; e++)
            {
                agents[e].ApplyAction(act.AsSpan(e * a, a), CartPoleWorld.Tau);
                worlds[e].Integrate();
                rew[e] = agents[e].CollectReward();
                done[e] = agents[e].IsDone() ? 1f : 0f;
                if (done[e] != 0f)
                {
                    worlds[e].Reset();
                    agents[e].ResetEpisode();
                }
                agents[e].WriteObservation(obs.AsSpan(e * s, s));
            }
        }

        float after = EvaluateGreedy(ac, maxSteps, episodes: 5);

        _out.WriteLine($"batched meanSteps before={before:0.0}, after={after:0.0} (/{maxSteps})");
        Assert.True(trainer.Iteration >= targetIterations,
            $"trainer completed only {trainer.Iteration}/{targetIterations} iterations.");
        Assert.True(after > before + 10f,
            $"Batched PPO showed no learning signal (before={before:0.0}, after={after:0.0}).");
    }

    [Fact]
    public void Loss_telemetry_is_emitted_and_finite()
    {
        const int maxSteps = 150;
        const int n = 8;
        const int horizon = 64;
        const int targetIterations = 4;
        Init.Seed(5);
        var rng = new Random(5);

        var (_, probe) = CompositeCartPole.CreateRaw(rng, maxSteps);
        var ac = new ActorCritic(probe.ObservationSize, probe.ActionSize, hidden: 32);
        var trainer = new BatchedPpoTrainer(ac, probe.Controls, agentCount: n, horizon: horizon, rng)
        {
            Epochs = 3,
            MinibatchSize = 256,
            LearningRate = 3e-3f,
        };

        var losses = new List<float>();
        trainer.OnIterationComplete += s =>
        {
            losses.Add(s.Loss);
            // The stat and the live property agree at emit time (set before the event fires).
            Assert.Equal(trainer.LastLoss, s.Loss);
        };

        var worlds = new CartPoleWorld[n];
        var agents = new CompositeAgent[n];
        for (int e = 0; e < n; e++)
        {
            var (w, ag) = CompositeCartPole.CreateRaw(rng, maxSteps);
            worlds[e] = w;
            agents[e] = ag;
            w.Reset();
            ag.ResetEpisode();
        }

        int s = trainer.ObservationSize, a = trainer.ActionSize;
        var obs = new float[n * s];
        var rew = new float[n];
        var done = new float[n];
        var act = new float[n * a];
        for (int e = 0; e < n; e++) agents[e].WriteObservation(obs.AsSpan(e * s, s));

        int safety = 0, safetyCap = targetIterations * horizon * 3;
        while (trainer.Iteration < targetIterations && safety++ < safetyCap)
        {
            trainer.Tick(obs, rew, done, act);
            for (int e = 0; e < n; e++)
            {
                agents[e].ApplyAction(act.AsSpan(e * a, a), CartPoleWorld.Tau);
                worlds[e].Integrate();
                rew[e] = agents[e].CollectReward();
                done[e] = agents[e].IsDone() ? 1f : 0f;
                if (done[e] != 0f)
                {
                    worlds[e].Reset();
                    agents[e].ResetEpisode();
                }
                agents[e].WriteObservation(obs.AsSpan(e * s, s));
            }
        }

        Assert.Equal(targetIterations, losses.Count);
        Assert.All(losses, l => Assert.True(float.IsFinite(l), $"loss was not finite: {l}"));
        Assert.Contains(losses, l => l != 0f);   // telemetry actually populated, not a stuck default
    }

    [Fact]
    public void Crash_guard_survives_a_nan_reward_storm_and_recovers()
    {
        const int maxSteps = 150;
        const int n = 8;
        const int horizon = 64;
        const int targetIterations = 4;
        Init.Seed(7);
        var rng = new Random(7);

        var (_, probe) = CompositeCartPole.CreateRaw(rng, maxSteps);
        var ac = new ActorCritic(probe.ObservationSize, probe.ActionSize, hidden: 32);
        var trainer = new BatchedPpoTrainer(ac, probe.Controls, agentCount: n, horizon: horizon, rng)
        {
            Epochs = 3,
            MinibatchSize = 256,
            LearningRate = 3e-3f,
        };

        var worlds = new CartPoleWorld[n];
        var agents = new CompositeAgent[n];
        for (int e = 0; e < n; e++)
        {
            var (w, ag) = CompositeCartPole.CreateRaw(rng, maxSteps);
            worlds[e] = w;
            agents[e] = ag;
            w.Reset();
            ag.ResetEpisode();
        }

        int s = trainer.ObservationSize, a = trainer.ActionSize;
        var obs = new float[n * s];
        var rew = new float[n];
        var done = new float[n];
        var act = new float[n * a];
        for (int e = 0; e < n; e++) agents[e].WriteObservation(obs.AsSpan(e * s, s));

        int safety = 0, safetyCap = targetIterations * horizon * 3;
        while (trainer.Iteration < targetIterations && safety++ < safetyCap)
        {
            trainer.Tick(obs, rew, done, act);
            for (int e = 0; e < n; e++)
            {
                agents[e].ApplyAction(act.AsSpan(e * a, a), CartPoleWorld.Tau);
                worlds[e].Integrate();
                // Poison every reward feeding the FIRST iteration's update; feed clean rewards after.
                rew[e] = trainer.Iteration == 0 ? float.NaN : agents[e].CollectReward();
                done[e] = agents[e].IsDone() ? 1f : 0f;
                if (done[e] != 0f) { worlds[e].Reset(); agents[e].ResetEpisode(); }
                agents[e].WriteObservation(obs.AsSpan(e * s, s));
            }
            // The weights must stay finite at every step, even mid-NaN-storm.
            AssertAllParamsFinite(ac);
        }

        // The guard engaged (the all-NaN iteration was discarded, not applied)...
        Assert.True(trainer.TotalSkippedUpdates > 0,
            "crash guard never engaged despite a NaN reward storm.");
        // ...the policy is still finite and usable (a NaN policy would draw the gun off-screen)...
        AssertAllParamsFinite(ac);
        Assert.True(float.IsFinite(EvaluateGreedy(ac, maxSteps, episodes: 3)));
        // ...and once clean rewards return, training recovers: the last iteration applied real updates.
        Assert.Equal(0, trainer.LastSkippedUpdates);
        Assert.True(float.IsFinite(trainer.LastLoss), $"loss did not recover: {trainer.LastLoss}");
    }

    [Fact]
    public void Value_target_normalization_keeps_the_critic_bounded()
    {
        const int maxSteps = 200;
        const int n = 8;
        const int horizon = 64;
        const int targetIterations = 8;
        Init.Seed(11);
        var rng = new Random(11);

        var (_, probe) = CompositeCartPole.CreateRaw(rng, maxSteps);
        var ac = new ActorCritic(probe.ObservationSize, probe.ActionSize, hidden: 32);
        var trainer = new BatchedPpoTrainer(ac, probe.Controls, agentCount: n, horizon: horizon, rng)
        {
            Epochs = 3,
            MinibatchSize = 256,
            LearningRate = 3e-3f,
        };

        float maxLoss = 0f;
        trainer.OnIterationComplete += s =>
        {
            if (float.IsFinite(s.Loss)) maxLoss = MathF.Max(maxLoss, MathF.Abs(s.Loss));
        };

        var worlds = new CartPoleWorld[n];
        var agents = new CompositeAgent[n];
        for (int e = 0; e < n; e++)
        {
            var (w, ag) = CompositeCartPole.CreateRaw(rng, maxSteps);
            worlds[e] = w;
            agents[e] = ag;
            w.Reset();
            ag.ResetEpisode();
        }

        int s = trainer.ObservationSize, a = trainer.ActionSize;
        var obs = new float[n * s];
        var rew = new float[n];
        var done = new float[n];
        var act = new float[n * a];
        for (int e = 0; e < n; e++) agents[e].WriteObservation(obs.AsSpan(e * s, s));

        int safety = 0, safetyCap = targetIterations * horizon * 3;
        while (trainer.Iteration < targetIterations && safety++ < safetyCap)
        {
            trainer.Tick(obs, rew, done, act);
            for (int e = 0; e < n; e++)
            {
                agents[e].ApplyAction(act.AsSpan(e * a, a), CartPoleWorld.Tau);
                worlds[e].Integrate();
                rew[e] = agents[e].CollectReward();
                done[e] = agents[e].IsDone() ? 1f : 0f;
                if (done[e] != 0f) { worlds[e].Reset(); agents[e].ResetEpisode(); }
                agents[e].WriteObservation(obs.AsSpan(e * s, s));
            }
        }

        // Returns reach well past unit scale (so the test actually exercises normalization)...
        Assert.True(trainer.ReturnScale > 5f,
            $"return scale {trainer.ReturnScale:0.0} too small — test can't distinguish normalized from raw.");
        // ...yet the value loss stays O(1), not ~ReturnScale² (the un-normalized critic's blow-up path).
        Assert.True(float.IsFinite(maxLoss) && maxLoss < 25f,
            $"value loss not bounded by normalization (max {maxLoss:0.0}, return scale {trainer.ReturnScale:0.0}).");
    }

    private static void AssertAllParamsFinite(ActorCritic ac)
    {
        foreach (var p in ac.Parameters())
            Assert.All(p.ToArray(), v => Assert.True(float.IsFinite(v), $"non-finite weight: {v}"));
    }

    // Greedy (mean-action) evaluation of the shared network, on its own RNG so it doesn't perturb the
    // training stream. Reuses the synchronous trainer's proven greedy rollout.
    private static float EvaluateGreedy(ActorCritic ac, int maxSteps, int episodes)
    {
        var evalRng = new Random(777);
        var ppo = new Ppo(ac, () => CompositeCartPole.Create(evalRng, maxSteps), evalRng);
        return ppo.EvaluateMeanSteps(episodes, maxSteps);
    }
}
