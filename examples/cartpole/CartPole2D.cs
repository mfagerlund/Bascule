using System;
using Godot;
using Tensotron.Rl;

namespace Tensotron.Godot.Examples;

/// <summary>
/// A self-contained cart-pole arena that advertises all four discovery roles at once: it is its own
/// observation, control surface, reward, and episode-reset. Drop it (or replicate it via
/// <see cref="LearningAgent"/>'s ArenaScene) and the trainer composes it with no further wiring — the
/// whole point of interface-driven discovery.
///
/// Control is <b>direct</b> (the v1 MVP): <see cref="IControlSurface.Apply"/> sets the force and
/// integrates one fixed simulation step itself, rather than handing a torque to Godot's physics server.
/// The step uses a fixed <see cref="Tau"/> (not the frame delta) so the dynamics — and thus the
/// learning problem — are identical whether the game runs at 60 fps or a headless trainer cranks
/// <c>PhysicsTicksPerSecond</c> for throughput. The physics matches the unit-test cart-pole exactly.
/// </summary>
[Tool]
[GlobalClass]
public partial class CartPole2D : Node2D, IObservationSource, IControlSurface, IRewardSource, IEpisodeReset
{
    // Dynamics — identical to the SinglePoleCart / CartPoleWorld used in the RL unit tests.
    private const float Gravity = 9.8f;
    private const float MassCart = 1.0f;
    private const float MassPole = 0.1f;
    private const float TotalMass = MassPole + MassCart;
    private const float Length = 0.5f;                 // half the pole's length
    private const float PoleMassLength = MassPole * Length;
    private const float ForceMag = 10.0f;
    private const float FourThirds = 4f / 3f;
    private const float Tau = 0.02f;                    // fixed simulation timestep (50 Hz)
    private const float RailLengthHalf = 2.4f;
    private const float MaxAngleRad = 12f * MathF.PI / 180f;

    /// <summary>Episode length cap, in simulation steps (matches the test's 200-step ceiling).</summary>
    [Export] public int MaxSteps { get; set; } = 200;

    private float _cartPosition, _cartSpeed, _poleAngle, _poleAngleSpeed;
    private int _steps;
    private float _pendingForce;
    private Random _rng = new(0);

    private bool OutOfBounds =>
        MathF.Abs(_cartPosition) > RailLengthHalf || MathF.Abs(_poleAngle) > MaxAngleRad;

    public override void _Ready()
    {
        // Diverse-but-reproducible start per arena, and a grid layout so a fleet of them is watchable.
        _rng = new Random(1000 + GetIndex());
        ResetWorld();
        // Defer: LearningAgent adds the arenas one at a time, so GetChildCount() is still partial during
        // this synchronous _Ready. Deferring runs PlaceInGrid once every sibling exists (final count).
        Callable.From(PlaceInGrid).CallDeferred();
    }

    // ---- IObservationSource ----
    public int Size => 4;
    public void Write(Span<float> dst)
    {
        dst[0] = _cartPosition;
        dst[1] = _cartSpeed;
        dst[2] = _poleAngle;
        dst[3] = _poleAngleSpeed;
    }

    // ---- IControlSurface ----
    public ControlSpec Spec { get; } = new(new[] { new ControlChannel("Force", -1f, 1f) });

    /// <summary>Direct control: apply the (normalized) force and advance the world one fixed step. The
    /// frame delta is intentionally ignored — see the class summary.</summary>
    public void Apply(ReadOnlySpan<float> action, float dt)
    {
        _pendingForce = Math.Clamp(action[0], -1f, 1f);
        Integrate();
        QueueRedraw();
    }

    // ---- IRewardSource ----
    public float Reward => 1f;                          // +1 per surviving step
    public bool Done => OutOfBounds || _steps >= MaxSteps;
    void IRewardSource.ResetEpisode() { }               // no per-episode reward bookkeeping

    // ---- IEpisodeReset ----
    void IEpisodeReset.ResetEpisode() => ResetWorld();  // the world reset (teleport + re-randomize)

    private void ResetWorld()
    {
        _cartPosition = Uniform(-0.05f, 0.05f);
        _cartSpeed = Uniform(-0.05f, 0.05f);
        _poleAngle = Uniform(-0.05f, 0.05f);
        _poleAngleSpeed = Uniform(-0.05f, 0.05f);
        _steps = 0;
        _pendingForce = 0f;
        QueueRedraw();
    }

    private float Uniform(float lo, float hi) => lo + (float)_rng.NextDouble() * (hi - lo);

    private void Integrate()
    {
        _steps++;
        float force = _pendingForce * ForceMag;
        float cos = MathF.Cos(_poleAngle);
        float sin = MathF.Sin(_poleAngle);

        float temp = (force + PoleMassLength * _poleAngleSpeed * _poleAngleSpeed * sin) / TotalMass;
        float thetaAcc = (Gravity * sin - cos * temp) /
                         (Length * (FourThirds - MassPole * cos * cos / TotalMass));
        float xAcc = temp - PoleMassLength * thetaAcc * cos / TotalMass;

        _cartPosition += Tau * _cartSpeed;
        _cartSpeed += Tau * xAcc;
        _poleAngle += Tau * _poleAngleSpeed;
        _poleAngleSpeed += Tau * thetaAcc;
    }

    // ---- Visual ----
    private const float RailHalfPx = 110f;
    private const float PolePx = 70f;

    public override void _Draw()
    {
        float scale = RailHalfPx / RailLengthHalf;
        var rail = new Color(0.4f, 0.4f, 0.45f);
        var cartColor = OutOfBounds ? new Color(0.85f, 0.3f, 0.3f) : new Color(0.3f, 0.6f, 0.9f);
        var poleColor = new Color(0.9f, 0.8f, 0.3f);

        DrawLine(new Vector2(-RailHalfPx, 0), new Vector2(RailHalfPx, 0), rail, 2f);

        float cartX = _cartPosition * scale;
        var cart = new Rect2(cartX - 18f, -9f, 36f, 18f);
        DrawRect(cart, cartColor);

        var pivot = new Vector2(cartX, -9f);
        var tip = pivot + new Vector2(MathF.Sin(_poleAngle) * PolePx, -MathF.Cos(_poleAngle) * PolePx);
        DrawLine(pivot, tip, poleColor, 4f);
    }

    private void PlaceInGrid()
    {
        int n = GetParent()?.GetChildCount() ?? 1;
        int cols = Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(n)));
        int idx = GetIndex();
        // Origin leaves a top band clear for the training HUD overlay (see TrainingHud).
        const float cellW = 260f, cellH = 150f;
        Position = new Vector2(140f + (idx % cols) * cellW, 320f + (idx / cols) * cellH);
    }
}
