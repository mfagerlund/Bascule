using System;
using Godot;
using Bascule.RL;

namespace Bascule.Godot.Examples;

/// <summary>
/// One turret-vs-target arena, and the example that exercises <em>multi-node</em> interface-driven
/// discovery: unlike the self-contained cart-pole node, this arena splits its three roles across
/// separate child nodes — <see cref="TurretGun"/> (control), <see cref="TargetSensor"/> (observation),
/// <see cref="HitReward"/> (reward) — which the trainer discovers and composes with no extra wiring.
/// This node is just the shared world they read and write, plus <see cref="IEpisodeReset"/> and the
/// drawing. The dynamics and constants are identical to the Godot-free <c>TurretEnv</c> the unit tests
/// train on, so windowed, headless, and test runs are the same learning problem.
///
/// Control is direct (the v1 MVP): <see cref="Advance"/> integrates one fixed step itself, so the
/// dynamics don't change when a headless trainer raises the physics tick rate.
/// </summary>
[Tool]
[GlobalClass]
public partial class TurretArena : Node2D, IEpisodeReset
{
    // Identical to TurretEnv (the unit-test fixture).
    private const float Tau = 1f / 60f;
    private const float MaxTurnRate = 4.0f;
    private const float HitThreshold = 0.12f;
    private const float AimShaping = 0.05f;
    private const float HitRewardValue = 1.0f;
    private const float MissPenalty = -0.25f;
    private const float MaxOmega = 1.2f;

    /// <summary>Fixed episode length (steps). Episodes never fail early — they just end.</summary>
    [Export] public int MaxSteps { get; set; } = 200;

    private float _turret, _target, _omega, _lastAim;
    private int _steps;
    private Random _rng = new(0);

    public float LastStepReward { get; private set; }
    public bool LastFired { get; private set; }
    public bool LastHit { get; private set; }
    public int Steps => _steps;

    /// <summary>Cumulative shots / hits across the whole run (not reset per episode) — the inference
    /// demo reads these to report shot accuracy.</summary>
    public int Shots { get; private set; }
    public int Hits { get; private set; }

    // ---- what the sensor reads ----
    public float BearingError => Wrap(_target - _turret);
    public float NormAim => _lastAim;
    public float NormOmega => _omega / MaxOmega;

    public override void _Ready()
    {
        _rng = new Random(2000 + GetIndex());
        ResetWorld();
        // Defer: LearningAgent adds the arenas one at a time, so GetChildCount() is still partial during
        // this synchronous _Ready. Deferring runs PlaceInGrid once every sibling exists (final count).
        Callable.From(PlaceInGrid).CallDeferred();
    }

    /// <summary>Direct-control step: rotate the turret, orbit the target, resolve a shot, and bank the
    /// reward — called by <see cref="TurretGun.Apply"/> once per tick. The frame delta is ignored in
    /// favour of the fixed <see cref="Tau"/>, exactly like the unit env.</summary>
    public void Advance(float aim, bool fire, float dt)
    {
        aim = Math.Clamp(aim, -1f, 1f);
        _turret = Wrap(_turret + aim * MaxTurnRate * Tau);
        _target = Wrap(_target + _omega * Tau);
        _lastAim = aim;

        float err = BearingError;
        float reward = AimShaping * MathF.Cos(err);
        bool hit = false;
        if (fire)
        {
            hit = MathF.Abs(err) <= HitThreshold;
            reward += hit ? HitRewardValue : MissPenalty;
            Shots++;
            if (hit) Hits++;
        }
        LastStepReward = reward;
        LastFired = fire;
        LastHit = hit;

        _steps++;
        QueueRedraw();
    }

    void IEpisodeReset.ResetEpisode() => ResetWorld();

    private void ResetWorld()
    {
        _turret = Uniform(-MathF.PI, MathF.PI);
        _target = Uniform(-MathF.PI, MathF.PI);
        _omega = Uniform(-MaxOmega, MaxOmega);
        _lastAim = 0f;
        _steps = 0;
        LastStepReward = 0f;
        LastFired = false;
        LastHit = false;
        QueueRedraw();
    }

    private float Uniform(float lo, float hi) => lo + (float)_rng.NextDouble() * (hi - lo);

    /// <summary>Wrap an angle to [-π, π).</summary>
    public static float Wrap(float a)
    {
        const float TwoPi = 2f * MathF.PI;
        return a - TwoPi * MathF.Floor((a + MathF.PI) / TwoPi);
    }

    // ---- Visual ----
    private const float TargetRadius = 60f;
    private const float BarrelLen = 46f;
    private const float BaseRadius = 13f;

    public override void _Draw()
    {
        var baseColor = new Color(0.35f, 0.4f, 0.5f);
        var barrelColor = LastFired
            ? (LastHit ? new Color(0.4f, 0.9f, 0.4f) : new Color(0.9f, 0.4f, 0.35f))
            : new Color(0.75f, 0.78f, 0.85f);
        var targetColor = new Color(0.95f, 0.7f, 0.25f);

        DrawCircle(Vector2.Zero, TargetRadius, new Color(0.16f, 0.17f, 0.2f));   // arena disc
        DrawCircle(Vector2.Zero, BaseRadius, baseColor);                          // turret base

        var aim = new Vector2(MathF.Cos(_turret), MathF.Sin(_turret));
        DrawLine(Vector2.Zero, aim * BarrelLen, barrelColor, 3f);

        if (LastFired)
            DrawLine(aim * BarrelLen, aim * (TargetRadius + 6f), barrelColor, 1.5f);  // tracer

        var tgt = new Vector2(MathF.Cos(_target), MathF.Sin(_target)) * TargetRadius;
        DrawCircle(tgt, 6f, targetColor);
    }

    private void PlaceInGrid()
    {
        int n = GetParent()?.GetChildCount() ?? 1;
        int cols = Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(n)));
        int idx = GetIndex();
        // Origin leaves a left band clear for the training HUD (see TurretDemo.BuildHud).
        const float originX = 390f, originY = 100f, cellW = 230f, cellH = 150f;
        Position = new Vector2(originX + (idx % cols) * cellW, originY + (idx / cols) * cellH);
    }
}
