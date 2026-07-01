using System;
using Godot;

namespace Bascule.Godot.Examples;

/// <summary>
/// The drift-racer's track geometry — a single closed centerline spline shared by every arena. It is
/// the <b>one source of truth</b> for both halves of the problem: the road is rendered along this
/// spline, and the reward (progress along the centerline) and the off-track test (lateral distance
/// from it) are measured against the very same curve. There is no separate collision mesh that could
/// drift out of sync with what the car sees.
///
/// Built on Godot's <see cref="Curve2D"/>: <see cref="Progress"/> wraps <c>GetClosestOffset</c>,
/// <see cref="SignedOffset"/> the lateral error, and <see cref="SampleAt"/>/<see cref="TangentAt"/>
/// feed the lookahead observations. A static <see cref="Default"/> instance (and a shared procedural
/// tarmac texture) is reused by all arenas — the layout is identical across the fleet, so the geometry
/// is computed exactly once.
/// </summary>
public sealed class RaceTrack
{
    /// <summary>Half the road width, in px. Beyond this lateral offset the car is off-track.</summary>
    public float HalfWidth { get; }

    /// <summary>Total baked length of the closed loop, in px (one lap).</summary>
    public float Length { get; }

    /// <summary>Baked centerline polyline — the points the road <see cref="global::Godot.Line2D"/> follows.</summary>
    public Vector2[] Centerline { get; }

    /// <summary>Pre-offset road edges (centerline ± normal·HalfWidth), drawn as the kerb lines.</summary>
    public Vector2[] LeftEdge { get; }
    public Vector2[] RightEdge { get; }

    private readonly Curve2D _curve;

    private static RaceTrack? _default;
    /// <summary>The shared twisty-circuit instance every arena drives on.</summary>
    public static RaceTrack Default => _default ??= BuildTwistyCircuit();

    private RaceTrack(Curve2D curve, float halfWidth)
    {
        _curve = curve;
        HalfWidth = halfWidth;
        Length = curve.GetBakedLength();
        Centerline = curve.GetBakedPoints();
        (LeftEdge, RightEdge) = BuildEdges(Centerline, halfWidth);
    }

    // --- queries used by the car for reward + observation ---

    /// <summary>Arc-length offset (0..Length) of the centerline point nearest <paramref name="localPos"/>.</summary>
    public float Progress(Vector2 localPos) => _curve.GetClosestOffset(localPos);

    /// <summary>Signed lateral distance from the centerline (px): + on the left of the travel direction,
    /// − on the right. |value| &gt; <see cref="HalfWidth"/> means off-track.</summary>
    public float SignedOffset(Vector2 localPos, float offset)
    {
        Vector2 onTrack = SampleAt(offset);
        Vector2 normal = LeftNormalAt(offset);
        return (localPos - onTrack).Dot(normal);
    }

    /// <summary>Centerline point at arc-length <paramref name="offset"/> (wrapped into the loop).</summary>
    public Vector2 SampleAt(float offset) => _curve.SampleBaked(Wrap(offset), false);

    /// <summary>Unit tangent (travel direction) at arc-length <paramref name="offset"/>.</summary>
    public Vector2 TangentAt(float offset)
    {
        Vector2 a = SampleAt(offset);
        Vector2 b = SampleAt(offset + 4f);
        Vector2 d = b - a;
        return d.LengthSquared() > 1e-6f ? d.Normalized() : Vector2.Right;
    }

    /// <summary>Left-hand normal of the travel direction at <paramref name="offset"/>.</summary>
    public Vector2 LeftNormalAt(float offset)
    {
        Vector2 t = TangentAt(offset);
        return new Vector2(t.Y, -t.X);   // rotate tangent +90° (screen-y points down)
    }

    /// <summary>Forward arc-length gap from <paramref name="from"/> to <paramref name="to"/>, signed and
    /// wrapped to [−Length/2, +Length/2] so crossing the start/finish line reads as smooth progress, not a
    /// full-lap jump.</summary>
    public float ForwardDelta(float from, float to)
    {
        float d = to - from;
        if (d > Length * 0.5f) d -= Length;
        else if (d < -Length * 0.5f) d += Length;
        return d;
    }

    private float Wrap(float offset)
    {
        offset %= Length;
        if (offset < 0f) offset += Length;
        return offset;
    }

    // --- construction ---

    // A smooth closed "twisty circuit": a base radius perturbed by a few sinusoids so curvature varies —
    // tight corners (where the car must scrub speed / slide) alternate with open sweepers (where it can
    // wind back up). Parametric-radial guarantees a valid, non-self-intersecting closed loop, and the y
    // squash makes it an organic circuit rather than a wobbly circle.
    private static RaceTrack BuildTwistyCircuit()
    {
        const int samples = 300;
        const float r0 = 175f, ySquash = 0.82f;
        var curve = new Curve2D { BakeInterval = 6f };

        // A longer circuit (~1.3k px lap). Amplitudes kept modest so the tightest centerline radius
        // (≈ r0 − Σamp ≈ 126px) stays well above the car's ≈52px turn radius — every corner is navigable;
        // the policy chooses to slide, it isn't forced off. Still varied enough that tight corners punish
        // carrying too much speed.
        for (int i = 0; i < samples; i++)
        {
            float t = i / (float)samples * MathF.Tau;
            float r = r0
                    + 24f * MathF.Sin(2f * t)
                    + 16f * MathF.Sin(3f * t + 1.1f)
                    + 9f * MathF.Sin(5f * t + 0.5f);
            curve.AddPoint(new Vector2(r * MathF.Cos(t), r * ySquash * MathF.Sin(t)));
        }
        curve.AddPoint(curve.GetPointPosition(0));   // close the loop back to the start
        return new RaceTrack(curve, halfWidth: 40f);
    }

    private static (Vector2[] left, Vector2[] right) BuildEdges(Vector2[] center, float halfWidth)
    {
        int n = center.Length;
        var left = new Vector2[n];
        var right = new Vector2[n];
        for (int i = 0; i < n; i++)
        {
            Vector2 prev = center[(i - 1 + n) % n];
            Vector2 next = center[(i + 1) % n];
            Vector2 t = (next - prev);
            t = t.LengthSquared() > 1e-6f ? t.Normalized() : Vector2.Right;
            var normal = new Vector2(t.Y, -t.X);
            left[i] = center[i] + normal * halfWidth;
            right[i] = center[i] - normal * halfWidth;
        }
        return (left, right);
    }

    // --- shared procedural tarmac (asphalt speckle), built once ---

    private static Texture2D? _tarmac;
    public static Texture2D Tarmac => _tarmac ??= BuildTarmac();

    private static Texture2D BuildTarmac()
    {
        var noise = new FastNoiseLite
        {
            NoiseType = FastNoiseLite.NoiseTypeEnum.SimplexSmooth,
            Frequency = 0.18f,
        };
        var ramp = new Gradient();
        ramp.SetColor(0, new Color(0.34f, 0.34f, 0.37f));   // medium-grey asphalt (so dark rubber shows)
        ramp.SetColor(1, new Color(0.44f, 0.44f, 0.47f));   // lighter aggregate speckle
        return new NoiseTexture2D
        {
            Width = 96,
            Height = 96,
            Seamless = true,
            Noise = noise,
            ColorRamp = ramp,
        };
    }
}
