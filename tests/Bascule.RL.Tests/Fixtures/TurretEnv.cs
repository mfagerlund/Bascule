namespace Bascule.RL.Tests;

/// <summary>
/// A turret that tracks an orbiting target and shoots it — the multi-channel, <em>mixed</em> task the
/// Godot turret example mirrors. One continuous "Aim" channel (turn rate) and one discrete "Fire"
/// channel (shoot / hold). Reward shapes aiming (cos of the bearing error) and pays off the discrete
/// decision (a hit while firing scores, a wasted shot is penalized), so a good policy both tracks the
/// target and fires only when lined up. Godot-free and deterministic; the Godot nodes use these exact
/// constants so windowed, headless, and unit runs share one learning problem.
/// </summary>
public sealed class TurretEnv : IEnvironment
{
    public const float Tau = 1f / 60f;
    public const float MaxTurnRate = 4.0f;        // rad/s at |aim|=1
    public const float HitThreshold = 0.12f;      // rad; a shot within this of the target hits
    public const float AimShaping = 0.05f;        // per-step reward * cos(bearing error)
    public const float HitReward = 1.0f;
    public const float MissPenalty = -0.25f;
    public const float MaxOmega = 1.2f;           // rad/s target orbital speed range
    public const int DefaultMaxSteps = 200;

    private readonly Random _rng;
    private readonly int _maxSteps;

    private float _turret;     // turret angle (rad)
    private float _target;     // target angle (rad)
    private float _omega;      // target angular speed (rad/s)
    private float _lastAim;    // last applied aim in [-1,1]
    private int _steps;

    public bool LastFired { get; private set; }
    public bool LastHit { get; private set; }
    public float TurretAngle => _turret;
    public float TargetAngle => _target;

    public TurretEnv(Random rng, int maxSteps = DefaultMaxSteps)
    {
        _rng = rng;
        _maxSteps = maxSteps;
        Roll();
    }

    public int ObservationSize => 4;
    public ControlSpec Controls { get; } = new(new[]
    {
        new ControlChannel("Aim", -1f, 1f),                     // continuous turn rate
        new ControlChannel("Fire", 0f, 1f, IsDiscrete: true),   // discrete shoot / hold
    });

    public float BearingError => Wrap(_target - _turret);

    private float Uniform(float lo, float hi) => lo + (float)_rng.NextDouble() * (hi - lo);

    private void Roll()
    {
        _turret = Uniform(-MathF.PI, MathF.PI);
        _target = Uniform(-MathF.PI, MathF.PI);
        _omega = Uniform(-MaxOmega, MaxOmega);
        _lastAim = 0f;
        _steps = 0;
        LastFired = false;
        LastHit = false;
    }

    public float[] Reset() { Roll(); return GetState(); }

    public float[] GetState()
    {
        float err = BearingError;
        return new[] { MathF.Sin(err), MathF.Cos(err), _lastAim, _omega / MaxOmega };
    }

    public (float reward, bool done) Step(ReadOnlySpan<float> action)
    {
        float aim = Math.Clamp(action[0], -1f, 1f);
        bool fire = (int)MathF.Round(action[1]) >= 1;

        _turret = Wrap(_turret + aim * MaxTurnRate * Tau);
        _target = Wrap(_target + _omega * Tau);
        _lastAim = aim;

        float err = BearingError;
        float reward = AimShaping * MathF.Cos(err);
        bool hit = false;
        if (fire)
        {
            hit = MathF.Abs(err) <= HitThreshold;
            reward += hit ? HitReward : MissPenalty;
        }
        LastFired = fire;
        LastHit = hit;

        _steps++;
        return (reward, _steps >= _maxSteps);
    }

    /// <summary>Wrap an angle to [-π, π).</summary>
    public static float Wrap(float a)
    {
        const float TwoPi = 2f * MathF.PI;
        return a - TwoPi * MathF.Floor((a + MathF.PI) / TwoPi);
    }
}
