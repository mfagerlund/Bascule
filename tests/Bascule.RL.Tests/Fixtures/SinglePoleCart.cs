namespace Bascule.RL.Tests;

/// <summary>
/// Classic single pole-cart (cart-pole), ported from the Tensotron engine showcase to the
/// multi-channel <see cref="IEnvironment"/> contract. Physics follows the standard model
/// (www.igi.tugraz.at/lehre/MLA/WS99/pole.c). Observation = [cartX, cartV, poleAngle, poleAngV];
/// one continuous control "Force" in [-1,1]. Self-contained — no engine or Godot dependency.
/// </summary>
public sealed class SinglePoleCart : IEnvironment
{
    private const float Gravity = 9.8f;
    private const float MassCart = 1.0f;
    private const float MassPole = 0.1f;
    private const float TotalMass = MassPole + MassCart;
    private const float Length = 0.5f; // half the pole's length
    private const float PoleMassLength = MassPole * Length;
    private const float ForceMag = 10.0f;
    private const float FourThirds = 4f / 3f;
    private const float Tau = 0.02f;
    private const float RailLengthHalf = 2.4f;
    private const float MaxAngleRad = 12f * MathF.PI / 180f;

    private readonly Random _rng;
    private readonly int _maxSteps;

    public SinglePoleCart(Random rng, int maxSteps = 500)
    {
        _rng = rng;
        _maxSteps = maxSteps;
    }

    public int ObservationSize => 4;
    public ControlSpec Controls { get; } = new(new[] { new ControlChannel("Force", -1f, 1f) });

    public float CartPosition { get; private set; }
    public float CartSpeed { get; private set; }
    public float PoleAngle { get; private set; }
    public float PoleAngleSpeed { get; private set; }
    public int Steps { get; private set; }

    private float Uniform(float lo, float hi) => lo + (float)_rng.NextDouble() * (hi - lo);

    public bool IsOutOfBounds =>
        MathF.Abs(CartPosition) > RailLengthHalf || MathF.Abs(PoleAngle) > MaxAngleRad;

    public float[] Reset()
    {
        CartPosition = Uniform(-0.05f, 0.05f);
        CartSpeed = Uniform(-0.05f, 0.05f);
        PoleAngle = Uniform(-0.05f, 0.05f);
        PoleAngleSpeed = Uniform(-0.05f, 0.05f);
        Steps = 0;
        return GetState();
    }

    public float[] GetState() => new[] { CartPosition, CartSpeed, PoleAngle, PoleAngleSpeed };

    public (float reward, bool done) Step(ReadOnlySpan<float> action)
    {
        float a = Math.Clamp(action[0], -1f, 1f);
        Steps++;
        float force = a * ForceMag;
        float cos = MathF.Cos(PoleAngle);
        float sin = MathF.Sin(PoleAngle);

        float temp = (force + PoleMassLength * PoleAngleSpeed * PoleAngleSpeed * sin) / TotalMass;
        float thetaAcc = (Gravity * sin - cos * temp) /
                         (Length * (FourThirds - MassPole * cos * cos / TotalMass));
        float xAcc = temp - PoleMassLength * thetaAcc * cos / TotalMass;

        CartPosition += Tau * CartSpeed;
        CartSpeed += Tau * xAcc;
        PoleAngle += Tau * PoleAngleSpeed;
        PoleAngleSpeed += Tau * thetaAcc;

        bool done = IsOutOfBounds || Steps >= _maxSteps;
        return (1f, done); // +1 per surviving step
    }
}
