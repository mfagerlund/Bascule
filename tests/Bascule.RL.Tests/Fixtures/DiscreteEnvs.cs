namespace Bascule.RL.Tests;

/// <summary>
/// A one-step contextual bandit with a single <em>discrete</em> control: given a random context, the
/// policy must pick the category that matches a fixed rule (here: whether the context sums positive).
/// It isolates the categorical head — sampling, log-prob, and entropy — from any sequential dynamics,
/// so a learning failure points squarely at the discrete action plumbing.
/// </summary>
public sealed class SignMatchBandit : IEnvironment
{
    public const int Obs = 4;

    private readonly Random _rng;
    private readonly float[] _ctx = new float[Obs];

    public SignMatchBandit(Random rng) { _rng = rng; Roll(); }

    public int ObservationSize => Obs;
    public ControlSpec Controls { get; } = new(new[] { new ControlChannel("Fire", 0f, 1f, IsDiscrete: true) });

    /// <summary>The correct category for a context: 1 if its components sum positive, else 0.</summary>
    public static int Target(ReadOnlySpan<float> ctx)
    {
        float s = 0f;
        for (int i = 0; i < ctx.Length; i++) s += ctx[i];
        return s > 0f ? 1 : 0;
    }

    private void Roll() { for (int i = 0; i < Obs; i++) _ctx[i] = (float)(_rng.NextDouble() * 2.0 - 1.0); }

    public float[] Reset() { Roll(); return GetState(); }
    public float[] GetState() => (float[])_ctx.Clone();

    public (float reward, bool done) Step(ReadOnlySpan<float> action)
    {
        int choice = (int)MathF.Round(action[0]);
        float reward = choice == Target(_ctx) ? 1f : 0f;
        return (reward, true); // one-step bandit: every step is terminal
    }
}

/// <summary>
/// A one-step bandit with <em>both</em> kinds of control at once: a discrete "Fire" channel and a
/// continuous "Aim" channel — and the discrete channel is declared first, so the continuous mean does
/// <b>not</b> sit at policy-output column 0. That deliberately exercises <see cref="ActionLayout"/>'s
/// column offsetting: a layout bug that confused logit columns with the mean column would fail here
/// even though the pure-discrete and pure-continuous tests pass. The optimum is aim = target and
/// fire = gate, both readable from the observation.
/// </summary>
public sealed class AimAndFire : IEnvironment
{
    private readonly Random _rng;
    private float _target;
    private int _gate;

    public AimAndFire(Random rng) { _rng = rng; Roll(); }

    public int ObservationSize => 2;
    public ControlSpec Controls { get; } = new(new[]
    {
        new ControlChannel("Fire", 0f, 1f, IsDiscrete: true),   // discrete first (column 0..1)
        new ControlChannel("Aim", -1f, 1f),                      // continuous mean at column 2
    });

    public float Target => _target;
    public int Gate => _gate;

    private void Roll()
    {
        _target = (float)(_rng.NextDouble() * 2.0 - 1.0);
        _gate = _rng.NextDouble() < 0.5 ? 0 : 1;
    }

    public float[] Reset() { Roll(); return GetState(); }
    public float[] GetState() => new[] { _target, _gate };

    public (float reward, bool done) Step(ReadOnlySpan<float> action)
    {
        int fire = (int)MathF.Round(action[0]);
        float aim = action[1];
        float aimErr = aim - _target;
        float reward = -(aimErr * aimErr) + (fire == _gate ? 1f : 0f);
        return (reward, true);
    }
}
