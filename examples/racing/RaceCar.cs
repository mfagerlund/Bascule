using System;
using Godot;
using Bascule.RL;

namespace Bascule.Godot.Examples;

/// <summary>
/// A drift-racing arena: a top-down car learns to drive (and power-slide) a closed circuit. It is the
/// third control flavour in the examples — <b>pure-code kinematic</b>, integrating its own bicycle
/// dynamics on a fixed timestep (like <see cref="CartPole2D"/>, unlike the physics-server
/// <c>PhysicsArm</c>), so the dynamics are identical at 60 fps or under a headless trainer cranking
/// <c>PhysicsTicksPerSecond</c>. It advertises all four discovery roles, so the trainer composes it
/// with no extra wiring.
///
/// <para><b>Drift model — credit.</b> The steering + traction model is ported from the
/// <i>kidscancode.org</i> Godot "Car Steering" recipe
/// (https://kidscancode.org/godot_recipes/4.x/2d/car_steering/), itself based on the classic
/// engineeringdotnet arcade-car algorithm. The power-slide is emergent: a rear/front-wheel bicycle
/// step produces a new heading, then the velocity is <em>lerped</em> toward that heading by a traction
/// factor that drops above <see cref="SlipSpeed"/> — so at speed the car keeps its momentum sideways
/// (slip angle = drift). A discrete handbrake channel forces the low-traction state on demand for a
/// deliberate Initial-D slide. Nothing in the reward asks for sliding; it emerges because carrying
/// speed through the corners is the fast line.</para>
/// </summary>
[Tool]
[GlobalClass]
public partial class RaceCar : Node2D, IObservationSource, IControlSurface, IRewardSource, IEpisodeReset
{
    // --- dynamics (px, seconds); tuned for the ~820px circuit ---
    private const float Dt = 1f / 60f;            // fixed step — fps-independent dynamics (see CartPole2D)
    private const float EnginePower = 620f;       // forward acceleration at full throttle, px/s²
    private const float Friction = -2.2f;         // linear rolling resistance (terminal speed ≈ 235 px/s)
    private const float Drag = -0.0015f;          // quadratic air drag
    private const float MaxSteer = 30f * MathF.PI / 180f;   // turn radius = WheelBase/tan(MaxSteer) ≈ 52px
    private const float WheelBase = 30f;          // front-to-rear axle distance, px
    private const float SlipSpeed = 150f;         // above this, traction drops and the car drifts
    private const float TractionSlow = 12f;       // grippy realign rate below SlipSpeed
    private const float TractionFast = 2.5f;      // loose realign rate above SlipSpeed (the drift)
    private const float HandbrakeTraction = 0.9f; // near-zero realign: a full slide while held
    private const float MaxReverseSpeed = 120f;
    private const float MaxSpeedRef = 270f;       // observation/visual normalization

    // --- reward shaping ---
    private const float ProgressScale = 0.07f;    // reward per px of centerline progress (≈ 56 / lap)
    private const float SpeedBonus = 0.015f;      // small carry-speed incentive (helps the slide pay off)
    private const float CrashPenalty = -1.5f;     // one-off on leaving the track (episode ends)

    private static readonly float[] Lookahead = { 45f, 95f, 170f, 270f };   // obs preview distances, px

    /// <summary>Episode length cap in steps. The start point is re-randomized each episode so the policy
    /// sees the whole circuit, not just the first corner.</summary>
    [Export] public int MaxSteps { get; set; } = 500;

    /// <summary>When true, the car renders nothing and is not placed in a grid — a single
    /// <see cref="RaceOverlay"/> draws one shared track and every car on top of it (leader opaque, the
    /// rest faded). Process-wide, set by the watch demo before the arenas spawn.</summary>
    public static bool Overlay;

    // --- pose + ranking, read by RaceOverlay in overlay mode ---
    public Vector2 CarPos => _pos;
    public float CarHeading => _heading;
    public float CarSlip => _slip;                 // signed lateral speed (drift tint)
    public float CarSteerAngle => _steerInput * MaxSteer;   // front-wheel angle, for drawing the wheels
    public float SlipFraction => Math.Clamp(MathF.Abs(_slip) / 150f, 0f, 1f);
    public float EpisodeDistance => _episodeDistance;   // px driven this episode — leader = the max

    private readonly RaceTrack _track = RaceTrack.Default;

    private Vector2 _pos;
    private float _heading;
    private Vector2 _velocity;
    private float _throttle, _steerInput;
    private bool _handbrake;
    private int _steps;
    private float _prevOffset;
    private float _lastReward;
    private bool _offTrack;
    private float _slip;            // signed lateral speed, for the drift tint
    private float _episodeDistance; // forward px accumulated this episode (overlay leader ranking)
    private Random _rng = new(0);

    private Line2D? _road;
    private Vector2[] _leftClosed = Array.Empty<Vector2>();
    private Vector2[] _rightClosed = Array.Empty<Vector2>();

    public override void _Ready()
    {
        _rng = new Random(7000 + GetIndex());
        ResetWorld();
        if (Overlay) return;            // RaceOverlay draws the shared track + every car; no per-car road/grid
        BuildRoad();
        // Defer: arenas are added one at a time, so GetChildCount() is partial in this synchronous _Ready.
        Callable.From(PlaceInGrid).CallDeferred();
    }

    // ---- IObservationSource ----
    public int Size => 5 + Lookahead.Length * 2;     // 13
    public void Write(Span<float> dst)
    {
        Vector2 fwd = Forward();
        Vector2 right = new(-fwd.Y, fwd.X);
        float offset = _track.Progress(_pos);
        float signed = _track.SignedOffset(_pos, offset);
        float tangentAngle = _track.TangentAt(offset).Angle();
        float headingErr = Wrap(tangentAngle - _heading);

        dst[0] = _velocity.Dot(fwd) / MaxSpeedRef;                       // forward speed
        dst[1] = _velocity.Dot(right) / MaxSpeedRef;                     // lateral speed (the slip)
        dst[2] = Math.Clamp(signed / _track.HalfWidth, -1.5f, 1.5f);     // where across the road
        dst[3] = MathF.Sin(headingErr);
        dst[4] = MathF.Cos(headingErr);

        for (int i = 0; i < Lookahead.Length; i++)
        {
            Vector2 ahead = _track.SampleAt(offset + Lookahead[i]);
            Vector2 local = (ahead - _pos).Rotated(-_heading);          // bend direction in the car's frame
            local = local.LengthSquared() > 1e-6f ? local.Normalized() : Vector2.Right;
            dst[5 + i * 2] = local.X;
            dst[6 + i * 2] = local.Y;
        }
    }

    // ---- IControlSurface ----
    // Throttle + steer are continuous; handbrake is a discrete on/off channel (forces the categorical
    // policy head, exactly like the turret's "Fire"). The slide is both emergent (speed) and on tap.
    public ControlSpec Spec { get; } = new(new[]
    {
        new ControlChannel("Throttle", -1f, 1f),
        new ControlChannel("Steer", -1f, 1f),
        new ControlChannel("Handbrake", 0f, 1f, IsDiscrete: true),
    });

    public void Apply(ReadOnlySpan<float> action, float dt)
    {
        _throttle = Math.Clamp(action[0], -1f, 1f);
        _steerInput = Math.Clamp(action[1], -1f, 1f);
        _handbrake = (int)MathF.Round(action[2]) >= 1;
        Integrate();
        QueueRedraw();
    }

    // ---- IRewardSource ----
    public float Reward => _lastReward;
    public bool Done => _offTrack || _steps >= MaxSteps;
    void IRewardSource.ResetEpisode() { }

    // ---- IEpisodeReset ----
    void IEpisodeReset.ResetEpisode() => ResetWorld();

    private void ResetWorld()
    {
        // Staggered grid start at the start/finish line (like real racing), not random around the lap:
        // two cars per row, each row a little further back, alternating sides of the centerline.
        int slot = GetIndex();
        float startOffset = _track.Length - (slot / 2 + 1) * 30f;            // rows behind the line (wraps)
        Vector2 center = _track.SampleAt(startOffset);
        Vector2 normal = _track.LeftNormalAt(startOffset);
        Vector2 tangent = _track.TangentAt(startOffset);

        float side = (slot % 2 == 0 ? 1f : -1f) * 0.45f * _track.HalfWidth;
        _pos = center + normal * (side + Uniform(-0.04f, 0.04f) * _track.HalfWidth);
        _heading = tangent.Angle() + Uniform(-0.05f, 0.05f);                  // tiny jitter for diversity
        _velocity = Forward() * 40f;                                          // gentle rolling start
        _prevOffset = _track.Progress(_pos);
        _steps = 0;
        _offTrack = false;
        _lastReward = 0f;
        _slip = 0f;
        _episodeDistance = 0f;
        QueueRedraw();
    }

    // The kidscancode bicycle + traction model (see class credit), plus reward/termination bookkeeping.
    private void Integrate()
    {
        _steps++;
        Vector2 fwd = Forward();

        // 1. longitudinal force, then rolling + air resistance, integrated into velocity
        Vector2 accel = fwd * (EnginePower * _throttle);
        accel += _velocity * Friction;
        accel += _velocity * (_velocity.Length() * Drag);
        _velocity += accel * Dt;

        // 2. bicycle step: rear wheel follows velocity, front wheel follows steered velocity
        float steer = _steerInput * MaxSteer;
        Vector2 rear = _pos - fwd * (WheelBase * 0.5f) + _velocity * Dt;
        Vector2 front = _pos + fwd * (WheelBase * 0.5f) + _velocity.Rotated(steer) * Dt;
        Vector2 newHeading = front - rear;
        newHeading = newHeading.LengthSquared() > 1e-6f ? newHeading.Normalized() : fwd;

        // 3. traction: realign velocity toward the new heading — slack above SlipSpeed / under handbrake
        float speed = _velocity.Length();
        float traction = speed > SlipSpeed ? TractionFast : TractionSlow;
        if (_handbrake) traction = HandbrakeTraction;
        if (speed > 1e-3f)
        {
            Vector2 velDir = _velocity / speed;
            if (newHeading.Dot(velDir) > 0f)
                _velocity = _velocity.Lerp(newHeading * speed, Math.Clamp(traction * Dt, 0f, 1f));
            else
                _velocity = -newHeading * MathF.Min(speed, MaxReverseSpeed);   // reversing
        }
        _heading = newHeading.Angle();
        _pos += _velocity * Dt;

        // 4. progress reward (centerline arc-length gained), small speed bonus, crash on leaving the road
        float offset = _track.Progress(_pos);
        float ds = _track.ForwardDelta(_prevOffset, offset);
        _prevOffset = offset;
        _episodeDistance += MathF.Max(0f, ds);
        _slip = _velocity.Dot(new Vector2(-fwd.Y, fwd.X));

        float fwdSpeed = _velocity.Dot(fwd);
        _lastReward = ds * ProgressScale + SpeedBonus * MathF.Max(0f, fwdSpeed) / MaxSpeedRef;

        if (MathF.Abs(_track.SignedOffset(_pos, offset)) > _track.HalfWidth)
        {
            _lastReward += CrashPenalty;
            _offTrack = true;
        }
    }

    private Vector2 Forward() => Vector2.Right.Rotated(_heading);
    private float Uniform(float lo, float hi) => lo + (float)_rng.NextDouble() * (hi - lo);

    private static float Wrap(float a)
    {
        a %= MathF.Tau;
        if (a > MathF.PI) a -= MathF.Tau;
        else if (a < -MathF.PI) a += MathF.Tau;
        return a;
    }

    // ---- Visual ----
    private void BuildRoad()
    {
        _road = new Line2D
        {
            Points = Close(_track.Centerline),
            Width = _track.HalfWidth * 2f,
            JointMode = Line2D.LineJointMode.Round,
            BeginCapMode = Line2D.LineCapMode.Round,
            EndCapMode = Line2D.LineCapMode.Round,
            DefaultColor = new Color(0.5f, 0.5f, 0.53f),    // clean light-grey asphalt (matches RaceOverlay)
            Antialiased = true,
            ZIndex = -1,                        // behind the car drawn in _Draw
        };
        AddChild(_road);
        _leftClosed = Close(_track.LeftEdge);
        _rightClosed = Close(_track.RightEdge);
    }

    public override void _Draw()
    {
        if (Overlay) return;            // the shared RaceOverlay draws everything in this mode

        // kerbs
        var kerb = new Color(0.9f, 0.9f, 0.95f, 0.8f);
        DrawPolyline(_leftClosed, kerb, 2.5f, true);
        DrawPolyline(_rightClosed, kerb, 2.5f, true);

        // start/finish bar
        Vector2 c0 = _track.SampleAt(0f);
        Vector2 n0 = _track.LeftNormalAt(0f);
        DrawLine(c0 + n0 * _track.HalfWidth, c0 - n0 * _track.HalfWidth, new Color(1f, 0.95f, 0.4f), 3f);

        // car body, tinted toward orange as it slides
        Vector2 f = Forward();
        Vector2 r = new(-f.Y, f.X);
        const float hl = 12f, hw = 6.5f;
        var body = new[]
        {
            _pos + f * hl + r * hw, _pos + f * hl - r * hw,
            _pos - f * hl - r * hw, _pos - f * hl + r * hw,
        };
        float slipN = Math.Clamp(MathF.Abs(_slip) / 150f, 0f, 1f);
        var grip = new Color(0.30f, 0.62f, 0.92f);
        var drift = new Color(1f, 0.55f, 0.15f);
        DrawColoredPolygon(body, grip.Lerp(drift, slipN));
        DrawLine(_pos, _pos + f * (hl + 5f), new Color(1f, 1f, 1f, 0.85f), 2f);   // nose / facing
    }

    private static Vector2[] Close(Vector2[] pts)
    {
        if (pts.Length == 0) return pts;
        var closed = new Vector2[pts.Length + 1];
        Array.Copy(pts, closed, pts.Length);
        closed[^1] = pts[0];
        return closed;
    }

    private void PlaceInGrid()
    {
        int n = GetParent()?.GetChildCount() ?? 1;
        int cols = Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(n)));
        int idx = GetIndex();
        const float cellW = 330f, cellH = 300f;       // each ~350px-wide circuit gets its own cell
        Position = new Vector2(195f + (idx % cols) * cellW, 250f + (idx / cols) * cellH);
    }
}
