using Xunit.Abstractions;

namespace Tensotron.Rl.Tests;

/// <summary>
/// The load-bearing proof for the RL core: a continuous PPO controller — trained entirely on
/// Tensotron tensors, through the generalized multi-channel <see cref="IEnvironment"/>/<see
/// cref="ControlSpec"/> abstraction — actually learns to balance a pole-cart. Sized to run fast on
/// any backend (CPU SIMD included) and asserts measurable *improvement* rather than full
/// convergence, so it stays in the default suite as a regression guard. Full-strength,
/// GPU-only convergence is a separate (deferred) showcase test.
/// </summary>
public class PpoLearningTests
{
    private readonly ITestOutputHelper _out;
    public PpoLearningTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Ppo_learns_to_balance_single_pole()
    {
        const int maxSteps = 150;
        Init.Seed(1);
        var rng = new Random(1);

        var probe = new SinglePoleCart(rng, maxSteps);
        var ac = new ActorCritic(probe.ObservationSize, probe.Controls.Count, hidden: 32);
        var ppo = new Ppo(ac, () => new SinglePoleCart(rng, maxSteps), rng)
        {
            NumEnvs = 8, Horizon = 64, Epochs = 3, MinibatchSize = 256, LearningRate = 3e-3f,
        };

        float before = ppo.EvaluateMeanSteps(episodes: 3, maxSteps: maxSteps);
        ppo.Train(5, (i, ret) => _out.WriteLine($"iter {i}: meanReturn={ret:0.0}"));
        float after = ppo.EvaluateMeanSteps(episodes: 3, maxSteps: maxSteps);

        _out.WriteLine($"meanSteps before={before:0.0}, after={after:0.0} (/{maxSteps})");
        Assert.True(after > before + 10f,
            $"PPO showed no learning signal (before={before:0.0}, after={after:0.0}).");
    }
}
