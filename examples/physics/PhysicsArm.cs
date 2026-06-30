using System;
using Godot;
using Tensotron.Rl;

namespace Tensotron.Godot.Examples;

/// <summary>
/// A self-contained <b>physics-control</b> arena (the second of v1's three training modes). A
/// torque-actuated <see cref="RigidBody2D"/> "rotor" must rotate to point at a target heading and hold
/// it. Unlike the direct-control demos (<see cref="CartPole2D"/>), <see cref="IControlSurface.Apply"/>
/// hands a torque to Godot's physics server via <see cref="RigidBody2D.ApplyTorque"/> and lets the
/// engine integrate it between ticks — so the agent must <em>manage angular momentum</em> (torque
/// toward the target, then counter-torque/brake to settle) rather than snap to the angle. It still
/// advertises all four discovery roles, so the trainer composes it with no extra wiring.
///
/// The body simulates continuously; only the target re-randomizes on episode reset (no
/// <see cref="RigidBody2D"/> teleporting — the discouraged, glitchy path is avoided entirely).
/// </summary>
[Tool]
[GlobalClass]
public partial class PhysicsArm : Node2D, IObservationSource, IControlSurface, IRewardSource, IEpisodeReset
{
    private const float MaxTorque = 16f;         // peak torque at action ±1
    private const float Inertia = 1f;            // overridden so angular accel is simply τ/I = τ
    private const float AngularDampValue = 4f;   // terminal ω ≈ τ/(I·damp) ≈ 4 rad/s — reaches ±π well inside an episode
    private const float MaxOmega = 6f;           // observation normalization for angular velocity
    private const float AlignedThreshold = 0.15f; // ~8.6°, drives the "on target" colour only

    /// <summary>Episode length cap, in physics ticks. A new random target is drawn each episode. Kept
    /// short (≈ rollout horizon) so each rollout completes ~1 episode per arena — a far smoother
    /// learning curve than long episodes that rarely finish inside a rollout.</summary>
    [Export] public int MaxSteps { get; set; } = 80;

    private RigidBody2D _rotor = null!;
    private float _targetAngle;
    private float _lastAction;
    private int _steps;
    private Random _rng = new(0);

    public override void _Ready()
    {
        _rotor = GetNode<RigidBody2D>("Rotor");
        _rotor.GravityScale = 0f;        // top-down: torque is the only thing that moves it
        _rotor.CanSleep = false;         // a sleeping body ignores ApplyTorque — never let it sleep
        _rotor.Inertia = Inertia;
        _rotor.AngularDamp = AngularDampValue;

        _rng = new Random(3000 + GetIndex());
        _targetAngle = _rotor.Rotation + RandPi();
        // Defer placement: LearningAgent adds arenas one at a time, so GetChildCount() is partial in
        // this synchronous _Ready (see CartPole2D for the same trap).
        Callable.From(PlaceInGrid).CallDeferred();
    }

    // ---- IObservationSource ----
    public int Size => 3;
    public void Write(Span<float> dst)
    {
        float err = WrapToPi(_targetAngle - _rotor.Rotation);
        dst[0] = MathF.Sin(err);
        dst[1] = MathF.Cos(err);
        dst[2] = Math.Clamp(_rotor.AngularVelocity / MaxOmega, -1f, 1f);
    }

    // ---- IControlSurface ----
    public ControlSpec Spec { get; } = new(new[] { new ControlChannel("Torque", -1f, 1f) });

    /// <summary>Physics control: hand the (normalized) torque to Godot and let the physics step integrate
    /// it. The frame delta is unused — the physics server owns the timestep.</summary>
    public void Apply(ReadOnlySpan<float> action, float dt)
    {
        _lastAction = Math.Clamp(action[0], -1f, 1f);
        _rotor.ApplyTorque(_lastAction * MaxTorque);
        _steps++;
        QueueRedraw();
    }

    // ---- IRewardSource ----
    public float Reward
    {
        get
        {
            float err = WrapToPi(_targetAngle - _rotor.Rotation);
            // dense pointing reward (peaks at the target) minus a tiny control cost. No explicit ω
            // penalty: holding cos(err)=1 already requires settling, and penalizing ω just teaches the
            // arm not to swing.
            return MathF.Cos(err) - 0.001f * _lastAction * _lastAction;
        }
    }
    public bool Done => _steps >= MaxSteps;
    void IRewardSource.ResetEpisode() { }

    // ---- IEpisodeReset ----
    void IEpisodeReset.ResetEpisode()
    {
        _rotor.AngularVelocity = 0f;                      // start each episode at rest (velocity sets are
                                                          // safe on RigidBody2D — unlike teleporting position)
        _targetAngle = _rotor.Rotation + RandPi();        // a fresh target at a random offset to swing to
        _steps = 0;
        QueueRedraw();
    }

    private float RandPi() => (float)(_rng.NextDouble() * 2.0 - 1.0) * MathF.PI;

    private static float WrapToPi(float a)
    {
        a %= 2f * MathF.PI;
        if (a > MathF.PI) a -= 2f * MathF.PI;
        else if (a < -MathF.PI) a += 2f * MathF.PI;
        return a;
    }

    // ---- Visual ----
    private const float ArmPx = 60f;
    public override void _Draw()
    {
        DrawArc(Vector2.Zero, ArmPx + 12f, 0f, MathF.Tau, 48, new Color(0.32f, 0.32f, 0.38f), 1.5f);

        var tdir = new Vector2(MathF.Cos(_targetAngle), MathF.Sin(_targetAngle));
        DrawCircle(tdir * (ArmPx + 12f), 5f, new Color(0.95f, 0.5f, 0.2f));   // target marker

        float rot = _rotor?.Rotation ?? 0f;
        bool aligned = MathF.Abs(WrapToPi(_targetAngle - rot)) < AlignedThreshold;
        var armColor = aligned ? new Color(0.4f, 0.95f, 0.5f) : new Color(0.55f, 0.7f, 0.95f);
        var dir = new Vector2(MathF.Cos(rot), MathF.Sin(rot));
        DrawLine(Vector2.Zero, dir * ArmPx, armColor, 4f);
        DrawCircle(Vector2.Zero, 6f, new Color(0.8f, 0.8f, 0.85f));
    }

    private void PlaceInGrid()
    {
        int n = GetParent()?.GetChildCount() ?? 1;
        int cols = Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(n)));
        int idx = GetIndex();
        const float cellW = 200f, cellH = 180f;
        Position = new Vector2(170f + (idx % cols) * cellW, 300f + (idx / cols) * cellH);
    }
}
