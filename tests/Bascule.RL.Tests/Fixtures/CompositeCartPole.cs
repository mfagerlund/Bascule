namespace Bascule.RL.Tests;

/// <summary>
/// The cart-pole re-expressed as the three discovery components over a shared world, instead of one
/// monolithic <see cref="IEnvironment"/>. Proves interface-driven discovery composes into something
/// the trainer can learn through. Physics matches <see cref="SinglePoleCart"/> exactly.
/// </summary>
internal sealed class CartPoleWorld
{
    private const float Gravity = 9.8f;
    private const float MassCart = 1.0f;
    private const float MassPole = 0.1f;
    private const float TotalMass = MassPole + MassCart;
    private const float Length = 0.5f;
    private const float PoleMassLength = MassPole * Length;
    public const float ForceMag = 10.0f;
    private const float FourThirds = 4f / 3f;
    public const float Tau = 0.02f;
    private const float RailLengthHalf = 2.4f;
    private const float MaxAngleRad = 12f * MathF.PI / 180f;

    private readonly Random _rng;
    private readonly int _maxSteps;

    public float CartPosition, CartSpeed, PoleAngle, PoleAngleSpeed;
    public int Steps;
    public float PendingForce; // normalized [-1,1], set by the control surface

    public CartPoleWorld(Random rng, int maxSteps = 500)
    {
        _rng = rng;
        _maxSteps = maxSteps;
    }

    private float Uniform(float lo, float hi) => lo + (float)_rng.NextDouble() * (hi - lo);

    public bool OutOfBounds =>
        MathF.Abs(CartPosition) > RailLengthHalf || MathF.Abs(PoleAngle) > MaxAngleRad;

    public bool Done => OutOfBounds || Steps >= _maxSteps;

    public void Reset()
    {
        CartPosition = Uniform(-0.05f, 0.05f);
        CartSpeed = Uniform(-0.05f, 0.05f);
        PoleAngle = Uniform(-0.05f, 0.05f);
        PoleAngleSpeed = Uniform(-0.05f, 0.05f);
        Steps = 0;
        PendingForce = 0f;
    }

    public void Integrate()
    {
        Steps++;
        float force = Math.Clamp(PendingForce, -1f, 1f) * ForceMag;
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
    }
}

internal sealed class CartPoleObservation : IObservationSource
{
    private readonly CartPoleWorld _w;
    public CartPoleObservation(CartPoleWorld w) => _w = w;
    public int Size => 4;
    public void Write(Span<float> dst)
    {
        dst[0] = _w.CartPosition;
        dst[1] = _w.CartSpeed;
        dst[2] = _w.PoleAngle;
        dst[3] = _w.PoleAngleSpeed;
    }
}

internal sealed class CartPoleForce : IControlSurface
{
    private readonly CartPoleWorld _w;
    public CartPoleForce(CartPoleWorld w) => _w = w;
    public ControlSpec Spec { get; } = new(new[] { new ControlChannel("Force", -1f, 1f) });
    public void Apply(ReadOnlySpan<float> action, float dt) => _w.PendingForce = action[0];
}

internal sealed class CartPoleReward : IRewardSource
{
    private readonly CartPoleWorld _w;
    public CartPoleReward(CartPoleWorld w) => _w = w;
    public float Reward => 1f;          // +1 per surviving step, like SinglePoleCart
    public bool Done => _w.Done;
    public void ResetEpisode() { }      // no per-episode accumulator
}

internal static class CompositeCartPole
{
    /// <summary>The composed cart-pole as an <see cref="IEnvironment"/> for the synchronous trainer.</summary>
    public static CompositeEnvironment Create(Random rng, int maxSteps)
    {
        var (world, agent) = CreateRaw(rng, maxSteps);
        return new CompositeEnvironment(agent, world.Integrate, world.Reset, dt: CartPoleWorld.Tau);
    }

    /// <summary>The raw world + its <see cref="CompositeAgent"/>, for driving the inverted/batched tick
    /// loop directly (the host owns advancing and resetting the world, as Godot does).</summary>
    public static (CartPoleWorld world, CompositeAgent agent) CreateRaw(Random rng, int maxSteps)
    {
        var world = new CartPoleWorld(rng, maxSteps);
        var agent = new CompositeAgent(
            new IObservationSource[] { new CartPoleObservation(world) },
            new IControlSurface[] { new CartPoleForce(world) },
            new IRewardSource[] { new CartPoleReward(world) });
        return (world, agent);
    }
}
