using System;
using System.Collections.Generic;
using System.Globalization;
using Godot;
using Bascule.RL;
using FluentSvg;
using NVec = System.Numerics.Vector2;

namespace Bascule.Godot.Examples;

/// <summary>
/// PuckWorld — the keep-away toy problem (Karpathy's reinforcejs classic). A puck must stay <b>close to a
/// green target</b> that teleports to a new spot every <see cref="TargetMoveFrequency"/> steps, while
/// <b>fleeing a red enemy</b> that slowly homes in on it. Two opposing drives in one reward, and the
/// learned behaviour reads at a glance: orbit the target, peel away whenever the enemy's danger ring
/// closes in, drift back the moment it's safe.
///
/// Like the cart-pole arena it advertises all four discovery roles at once (observation, control surface,
/// reward, episode reset), so <see cref="LearningAgent"/> composes it with no wiring. Control is
/// <b>direct and continuous</b> (the v1 MVP): the policy outputs a 2-D thrust vector in [-1,1]², and
/// <see cref="Apply"/> integrates one fixed step itself — frame delta ignored — so the dynamics are
/// identical windowed, headless, or under a cranked physics tick. The dynamics are a faithful port of the
/// original <c>PuckWorld</c> (unit box, damping, wall bounce, slow-homing enemy).
/// </summary>
[Tool]
[GlobalClass]
public partial class PuckArena : Node2D, IObservationSource, IControlSurface, IRewardSource, IEpisodeReset
{
    // ---- dynamics (faithful to the original PuckWorld; world lives in the unit box [0,1]²) ----
    public const float Radius = 0.05f;          // the puck's radius
    private const float Damping = 0.95f;        // per-step velocity decay
    private const float Accel = 0.002f / 2f;    // thrust applied per step
    private const float MaxSpeed = 0.03722872f; // speed cap (the original's tuned value)
    private const float Bounciness = 0.5f;      // wall-bounce restitution
    private const float EnemyHoming = 0.001f;   // how fast the enemy creeps toward the puck each step
    private const float SlowdownZone = 0.05f;   // |thrust| below this brakes instead of accelerating

    /// <summary>The enemy's danger-ring radius. Karpathy's original is 0.25 (a large zone covering ~28% of
    /// the arena); a smaller value makes the enemy a localized, dodgeable threat.</summary>
    [Export] public float BadRadius { get; set; } = 0.25f;

    /// <summary>The target jumps to a fresh random spot every this many steps — the moving goal.</summary>
    [Export] public int TargetMoveFrequency { get; set; } = 100;

    /// <summary>Reward weight for hugging the target (closer = higher).</summary>
    [Export] public float TargetBenefitWeight { get; set; } = 1f;

    /// <summary>Optional dense "progress" reward (off by default): credit the per-step <em>reduction</em> in
    /// distance to the target. The raw <c>-distance</c> reward is dominated by where the target/enemy happen
    /// to be (exogenous noise that swamps a single thrust's tiny effect), so on-policy PPO can't read the
    /// action's contribution. Rewarding distance <em>closed</em> is directly action-coupled — far higher
    /// signal-to-noise — and the baseline is reset on a teleport so the jump doesn't register as progress.</summary>
    [Export] public float ProgressCoef { get; set; } = 0f;

    /// <summary>Reward weight for the enemy-proximity penalty (inside the danger ring).</summary>
    [Export] public float EnemyPunishWeight { get; set; } = 1f;

    /// <summary>Optional dense "flee" reward (off by default), the enemy-side counterpart to
    /// <see cref="ProgressCoef"/>: while inside the danger ring, credit the per-step <em>increase</em> in
    /// distance to the enemy. Without it the only enemy signal is the always-on proximity penalty, which the
    /// puck escapes by fleeing to a wall (losing the target); a dense dodge gradient lets it learn the
    /// efficient "step out of the ring, then drift back to the target" dance.</summary>
    [Export] public float EnemyFleeCoef { get; set; } = 0f;

    /// <summary>Optional penalty on speed (off by default, as in the original).</summary>
    [Export] public float MovementPunishWeight { get; set; } = 0f;

    /// <summary>Fixed episode length (steps). The task is continuing — episodes never fail early, they
    /// just end and re-randomize, so the policy learns the steady-state keep-away behaviour.</summary>
    [Export] public int MaxSteps { get; set; } = 400;

    /// <summary>Control mode. <b>true</b> = the original PuckWorld's five discrete actions
    /// (left/right/up/down/coast) as a single 5-way discrete channel — the shape Karpathy's DQN solved.
    /// <b>false</b> = a 2-D continuous thrust. Discrete gives a crisper, less noisy learning signal on the
    /// two-drive keep-away problem; continuous shows the mixed/continuous control surface.</summary>
    [Export] public bool DiscreteControl { get; set; } = true;

    /// <summary>Static opt-in (set before arenas spawn, like <c>RaceCar.Overlay</c>): when true, each arena
    /// records its render trajectory so a completed episode can be exported as an animated SVG. Off during
    /// training so it costs nothing there.</summary>
    public static bool Recording;

    // ---- headless sweep overrides (set from the root demo's _EnterTree via cmdline, before arenas spawn) ----
    public static bool? DiscreteOverride;
    public static float? RingOverride, EnemyWeightOverride, TargetWeightOverride, ProgressOverride, FleeOverride;

    /// <summary>Parse arena-shaping flags from <c>++ --ring= --discrete= --enemyw= --targetw= --progress= --fleew=</c>
    /// into the static overrides above, so one scene can sweep configs headlessly without a scene per cell.
    /// Call from the root demo's <c>_EnterTree</c> (before <see cref="LearningAgent"/> spawns the arenas).</summary>
    public static void ApplyCmdlineOverrides()
    {
        foreach (string arg in OS.GetCmdlineUserArgs())
        {
            if      (arg.StartsWith("--ring="))     RingOverride         = ParseF(arg);
            else if (arg.StartsWith("--enemyw="))   EnemyWeightOverride  = ParseF(arg);
            else if (arg.StartsWith("--targetw="))  TargetWeightOverride = ParseF(arg);
            else if (arg.StartsWith("--progress=")) ProgressOverride     = ParseF(arg);
            else if (arg.StartsWith("--fleew="))    FleeOverride         = ParseF(arg);
            else if (arg.StartsWith("--discrete=")) DiscreteOverride     = arg["--discrete=".Length..] is "1" or "true" or "True";
        }
    }

    private static float ParseF(string arg) => float.Parse(arg[(arg.IndexOf('=') + 1)..], CultureInfo.InvariantCulture);

    private Vector2 _puck, _vel, _target, _enemy;
    private int _time;
    private float _lastReward;
    private bool _inDanger;
    private float _prevTargetDist;   // distance baseline for the optional target-progress reward
    private float _prevEnemyDist;    // distance baseline for the optional flee reward
    private Random _rng = new(0);

    private readonly List<float[]> _epFrames = new();
    /// <summary>The most recently completed episode's render frames (each [px,py,tx,ty,ex,ey]); null until
    /// one full episode has finished. The SVG export reads this so the loop is a single clean episode.</summary>
    public float[][]? LastEpisode { get; private set; }
    /// <summary>True once a full episode has been recorded and is ready to export.</summary>
    public bool HasEpisode => LastEpisode != null;

    public override void _Ready()
    {
        // Headless sweep overrides win over the scene's exports (applied before Spec/ResetWorld read them).
        if (DiscreteOverride is { } d) DiscreteControl = d;
        if (RingOverride is { } r) BadRadius = r;
        if (EnemyWeightOverride is { } ew) EnemyPunishWeight = ew;
        if (TargetWeightOverride is { } tw) TargetBenefitWeight = tw;
        if (ProgressOverride is { } pc) ProgressCoef = pc;
        if (FleeOverride is { } fc) EnemyFleeCoef = fc;

        _rng = new Random(3000 + GetIndex());
        ResetWorld();
        // Defer: LearningAgent adds arenas one at a time, so GetChildCount() is partial during _Ready.
        Callable.From(PlaceInGrid).CallDeferred();
    }

    // ---- IObservationSource (8 floats — the original PuckWorld state) ----
    public int Size => 8;
    public void Write(Span<float> dst)
    {
        dst[0] = _puck.X - 0.5f;            // puck position, centred
        dst[1] = _puck.Y - 0.5f;
        dst[2] = _vel.X * 10f;              // velocity, scaled into a useful range
        dst[3] = _vel.Y * 10f;
        dst[4] = _target.X - _puck.X;       // vector to the target
        dst[5] = _target.Y - _puck.Y;
        dst[6] = _enemy.X - _puck.X;        // vector to the enemy
        dst[7] = _enemy.Y - _puck.Y;
    }

    // ---- IControlSurface (discrete 5-way or 2-D continuous; the trainer never knows it drives a puck) ----
    private ControlSpec? _spec;
    public ControlSpec Spec => _spec ??= DiscreteControl
        ? new(new[] { new ControlChannel("Move", 0f, 4f, IsDiscrete: true) })   // L / R / U / D / coast
        : new(new[] { new ControlChannel("ThrustX", -1f, 1f), new ControlChannel("ThrustY", -1f, 1f) });

    /// <summary>Direct control: apply the action and advance the world one fixed step. Frame delta is
    /// intentionally ignored so the learning problem is tick-rate independent.</summary>
    public void Apply(ReadOnlySpan<float> action, float dt)
    {
        _time++;
        if (_time % TargetMoveFrequency == 0) RandomizeTarget();

        // Integrate the puck: coast, damp, cap, thrust, cap again, then bounce off the walls.
        _puck += _vel;
        _vel *= Damping;
        CapSpeed();

        if (DiscreteControl)
        {
            // Category index 0..4 → a cardinal push (4 = coast). The original PuckWorld's action set.
            _vel += (int)MathF.Round(action[0]) switch
            {
                0 => new Vector2(-Accel, 0f),   // left
                1 => new Vector2(Accel, 0f),    // right
                2 => new Vector2(0f, -Accel),   // up
                3 => new Vector2(0f, Accel),    // down
                _ => Vector2.Zero,              // coast
            };
        }
        else
        {
            Vector2 thrust = new(action[0], action[1]);
            float mag = thrust.Length();
            if (mag > 1f) thrust /= mag;
            if (mag > SlowdownZone) _vel += thrust * Accel;
            else _vel *= 0f;        // a near-zero thrust is an explicit brake (the original's behaviour)
        }
        CapSpeed();

        Bounce();

        _lastReward = ComputeReward();

        if (Recording) _epFrames.Add(RenderFrame());
        QueueRedraw();
    }

    private void CapSpeed()
    {
        float s = _vel.Length();
        if (s > MaxSpeed) _vel = _vel / s * MaxSpeed;
    }

    private void Bounce()
    {
        if (_puck.X < Radius)       { _vel.X = -_vel.X * Bounciness; _puck.X = Radius; }
        if (_puck.X > 1f - Radius)  { _vel.X = -_vel.X * Bounciness; _puck.X = 1f - Radius; }
        if (_puck.Y < Radius)       { _vel.Y = -_vel.Y * Bounciness; _puck.Y = Radius; }
        if (_puck.Y > 1f - Radius)  { _vel.Y = -_vel.Y * Bounciness; _puck.Y = 1f - Radius; }
    }

    private float ComputeReward()
    {
        // The enemy slowly homes toward the puck (this is what makes "stay away" a moving problem).
        Vector2 toPuck = _puck - _enemy;
        float enemyDist = toPuck.Length();
        if (enemyDist > 0f) _enemy += toPuck / enemyDist * EnemyHoming;

        // Stay close to the target: reward is the negative distance to it.
        float targetDist = (_puck - _target).Length();
        float reward = -targetDist * TargetBenefitWeight;

        // Optional dense progress signal: credit distance closed since last step (action-coupled).
        if (ProgressCoef > 0f) reward += ProgressCoef * (_prevTargetDist - targetDist);
        _prevTargetDist = targetDist;

        // ...but pay a penalty that grows as the puck enters the enemy's danger ring.
        float aradius = BadRadius + Radius;
        _inDanger = enemyDist < aradius;
        if (_inDanger)
        {
            reward += EnemyPunishWeight * (enemyDist - aradius) / aradius;
            if (EnemyFleeCoef > 0f) reward += EnemyFleeCoef * (enemyDist - _prevEnemyDist);   // dense dodge signal
        }
        _prevEnemyDist = enemyDist;

        if (MovementPunishWeight > 0f) reward -= MovementPunishWeight * _vel.Length();
        return reward;
    }

    /// <summary>Current puck→target distance (unit-box units) — the headless eval reads this to grade a
    /// checkpoint with far lower variance than the per-episode return.</summary>
    public float TargetDistance => (_puck - _target).Length();
    /// <summary>True while the puck is inside the enemy's danger ring this step.</summary>
    public bool InDanger => _inDanger;

    // ---- IRewardSource ----
    public float Reward => _lastReward;
    public bool Done => _time >= MaxSteps;
    void IRewardSource.ResetEpisode() { }   // reward needs no per-episode bookkeeping

    // ---- IEpisodeReset ----
    void IEpisodeReset.ResetEpisode() => ResetWorld();

    private void ResetWorld()
    {
        // Snapshot the episode we just finished (for the SVG export), then start fresh.
        if (Recording && _epFrames.Count > 0)
        {
            LastEpisode = _epFrames.ToArray();
            _epFrames.Clear();
        }

        _time = 0;
        _puck = RandomPoint();
        _vel = Vector2.Zero;
        RandomizeTarget();                              // also seeds _prevTargetDist
        _enemy = RandomPoint();
        _prevEnemyDist = (_puck - _enemy).Length();     // seed the flee baseline
        _lastReward = 0f;
        _inDanger = false;
        QueueRedraw();
    }

    private void RandomizeTarget()
    {
        _target = RandomPoint() * (1f - Radius * 2f) + new Vector2(Radius, Radius);
        _prevTargetDist = (_puck - _target).Length();   // reset the progress baseline (no teleport "progress")
    }

    private Vector2 RandomPoint() => new((float)_rng.NextDouble(), (float)_rng.NextDouble());

    private float[] RenderFrame() => new[] { _puck.X, _puck.Y, _target.X, _target.Y, _enemy.X, _enemy.Y };

    // ---- Visual (windowed watch) ----
    private const float ViewPx = 180f;   // the unit box drawn at this pixel size

    public override void _Draw()
    {
        var frame = new Color(0.3f, 0.32f, 0.38f);
        var danger = new Color(0.9f, 0.3f, 0.28f, 0.16f);
        var enemyCol = new Color(0.93f, 0.33f, 0.3f);
        var targetCol = new Color(0.35f, 0.86f, 0.5f);
        var puckCol = _inDanger ? new Color(0.98f, 0.7f, 0.3f) : new Color(0.36f, 0.66f, 0.96f);

        DrawRect(new Rect2(0, 0, ViewPx, ViewPx), new Color(0.12f, 0.13f, 0.16f));
        DrawRect(new Rect2(0, 0, ViewPx, ViewPx), frame, false, 1.5f);

        DrawCircle(_enemy * ViewPx, BadRadius * ViewPx, danger);          // danger ring (to avoid)
        DrawCircle(_enemy * ViewPx, Radius * 0.6f * ViewPx, enemyCol);    // enemy core
        DrawCircle(_target * ViewPx, Radius * 0.7f * ViewPx, targetCol);  // target (to hug)
        DrawCircle(_puck * ViewPx, Radius * ViewPx, puckCol);            // the agent
    }

    private void PlaceInGrid()
    {
        int n = GetParent()?.GetChildCount() ?? 1;
        int cols = Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(n)));
        int idx = GetIndex();
        const float originX = 60f, originY = 90f, cell = ViewPx + 26f;
        Position = new Vector2(originX + (idx % cols) * cell, originY + (idx / cols) * cell);
    }

    // ---- SVG export (built with FluentSvg: github.com/mfagerlund/FluentSvg) --------------------

    private const float SvgScale = 600f;      // the unit box rendered at this size
    private const float SvgFps = 20f;         // playback rate the SMIL duration assumes (~3× slow-mo)
    private const int SvgTargetFrames = 160;  // motion frames after subsampling (keeps the file small)
    private const int SvgHoldFrames = 16;     // brief hold on the last pose before the loop restarts

    /// <summary>
    /// Write the last completed episode as a self-contained, looping <b>animated SVG</b> — the green
    /// target, the red enemy with its danger ring, and the puck, each a <c>&lt;circle&gt;</c> whose centre
    /// is driven by SMIL <c>&lt;animate&gt;</c> so it plays in a GitHub README. No screen capture: it is
    /// rebuilt from the recorded trajectory. Assembled with <b>FluentSvg</b>, which formats every number
    /// invariant-culture, so output is stable regardless of the machine locale.
    /// </summary>
    public void ExportSvg(string absolutePath)
    {
        if (LastEpisode == null) { GD.PushWarning("[PuckArena] ExportSvg called before an episode was recorded."); return; }

        float[][] frames = LastEpisode;
        int total = frames.Length;

        // Subsample to a manageable frame count, then hold the final pose briefly. Even spacing needs no
        // keyTimes — folding the hold frames into the value list makes the loop dwell on the last pose.
        int step = Math.Max(1, (int)MathF.Ceiling(total / (float)SvgTargetFrames));
        var idx = new List<int>();
        for (int g = 0; g < total; g += step) idx.Add(g);
        if (idx[^1] != total - 1) idx.Add(total - 1);
        int motionCount = idx.Count;
        for (int k = 0; k < SvgHoldFrames; k++) idx.Add(total - 1);
        string dur = Svg.Tos(idx.Count / SvgFps) + "s";

        var svg = new Svg(absolutePath, title: "PuckWorld — keep-away") { Margin = NVec.Zero };
        svg.AddRectangleFromTo(NVec.Zero, new NVec(SvgScale, SvgScale)).SetFill("#15151a").ClearStroke();
        svg.AddRectangleFromTo(new NVec(1, 1), new NVec(SvgScale - 1, SvgScale - 1))
           .SetFill("transparent").SetStroke("#3a3d47", 2);

        // Enemy danger ring (faint) + core, then the green target, then the puck on top.
        AnimateCircle(svg, frames, idx, dur, 4, 5, BadRadius * SvgScale,     "#ef4444", 0.16f, null);
        AnimateCircle(svg, frames, idx, dur, 4, 5, Radius * 0.6f * SvgScale, "#ef4444", 1f,    null);
        AnimateCircle(svg, frames, idx, dur, 2, 3, Radius * 0.7f * SvgScale, "#4ade80", 1f,    null);
        AnimateCircle(svg, frames, idx, dur, 0, 1, Radius * SvgScale,        "#52a3f2", 1f,    "#dbe9ff");

        svg.SaveToFile();
        GD.Print($"[PuckArena] wrote {absolutePath} ({motionCount} frames, {dur} loop).");
    }

    /// <summary>Add a circle and drive its cx/cy across the recorded path, looping forever.</summary>
    private static void AnimateCircle(Svg svg, float[][] frames, List<int> idx, string dur,
        int colX, int colY, float r, string fill, float opacity, string? stroke)
    {
        // Anchor the static circle at the box centre: FluentSvg computes the viewBox from geometry, and a
        // big danger ring parked near a wall would otherwise inflate it past the arena. The SMIL animation
        // below drives the real centre from t=0, so this anchor is never actually seen.
        var circle = svg.AddCircle(new NVec(SvgScale / 2f, SvgScale / 2f), r).SetFill(fill).ClearStroke();
        if (opacity < 1f) circle.SetFillOpacity(opacity);
        if (stroke != null) circle.SetStroke(stroke, 2);

        var centres = new List<NVec>(idx.Count);
        foreach (int i in idx) centres.Add(new NVec(frames[i][colX] * SvgScale, frames[i][colY] * SvgScale));
        var (ax, ay) = svg.AddAnimateXy(circle, dur, centres);
        ax.SetAttribute("repeatCount", "indefinite");
        ay.SetAttribute("repeatCount", "indefinite");
    }
}
