using System;
using System.Collections.Generic;
using Godot;
using FluentSvg;
using NVec = System.Numerics.Vector2;

namespace Bascule.Godot.Examples;

/// <summary>
/// Turns the live fleet into a <b>replayed race</b>. Every <see cref="RaceCar"/> is stepped live (by the
/// trainer), but the overlay never draws them where they are right now — instead it <b>records each car's
/// best completed run</b> (the furthest single episode it has driven so far), then plays all of those runs
/// back <b>in lockstep from the shared start line</b>. You watch the pack fan out, cars drop off where they
/// crashed, and the <b>ultimate winner</b> — the one that got furthest — pulls away ringed and fully opaque
/// while the rest fade in proportion to how far behind they ended up. The replay loops; each lap it rebuilds
/// from the current bests, so as training improves the recorded race gets faster and cleaner. The rear wheels
/// lay persistent <b>skidmarks</b> from the replayed poses, so the racing line rubbers in over the loops.
/// Requires <see cref="RaceCar.Overlay"/> set (cars then skip their own grid/road drawing and just expose
/// their pose + episode distance).
/// </summary>
[GlobalClass]
public partial class RaceOverlay : Node2D
{
    /// <summary>Dimmest a far-behind car fades to (the winner is always fully opaque).</summary>
    [Export(PropertyHint.Range, "0.05,1,0.01")] public float GhostAlphaFloor { get; set; } = 0.12f;

    /// <summary>Uniform zoom so the circuit fills the window.</summary>
    [Export] public float ViewScale { get; set; } = 1.7f;

    /// <summary>Push the circuit down a little so it clears the training HUD at the top.</summary>
    [Export] public float VerticalOffset { get; set; } = 40f;

    private const float SkidSlip = 0.2f;        // SlipFraction above which the rears lay rubber
    private const int MaxSkidPoints = 36000;    // FIFO cap — long buffer so the line rubbers in over loops
    private const int EndHoldFrames = 70;       // pause on the final frame before the next take restarts
    private const int MinRunFrames = 20;        // ignore instant-crash episodes when picking a best run
    private const int MaxRunFrames = 600;       // safety: force-finalize a run if no reset was detected

    private readonly RaceTrack _track = RaceTrack.Default;
    private readonly List<RaceCar> _cars = new();

    // --- recording: per-car in-progress buffer + best completed run ---
    private struct Frame { public Vector2 Pos; public float Heading; public float Slip; public float Steer; }
    private sealed class Run { public Frame[] Frames = Array.Empty<Frame>(); public float Distance; }

    private List<Frame>[] _cur = Array.Empty<List<Frame>>();
    private Run?[] _best = Array.Empty<Run?>();
    private float[] _lastDist = Array.Empty<float>();

    // --- replay: a snapshot of the current bests, played in lockstep ---
    private Run?[] _replay = Array.Empty<Run?>();
    private float[] _alpha = Array.Empty<float>();   // per-car opacity from finish gap
    private int _winner = -1;
    private int _replayFrame;
    private int _replayLen;
    private int _holdCounter;
    private bool _replayReady;

    // --- skidmarks (laid from the replayed poses) ---
    private readonly List<Vector2> _skid = new();
    private (Vector2 l, Vector2 r)[] _skidPrev = Array.Empty<(Vector2, Vector2)>();
    private bool[] _skidPrevValid = Array.Empty<bool>();

    private Line2D? _road;
    private Vector2[] _leftClosed = Array.Empty<Vector2>();
    private Vector2[] _rightClosed = Array.Empty<Vector2>();

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        _road = new Line2D
        {
            Points = Close(_track.Centerline),
            Width = _track.HalfWidth * 2f,
            JointMode = Line2D.LineJointMode.Round,
            BeginCapMode = Line2D.LineCapMode.Round,
            EndCapMode = Line2D.LineCapMode.Round,
            DefaultColor = new Color(0.5f, 0.5f, 0.53f),    // clean light-grey asphalt, no baked-in marks
            Antialiased = true,
            ZIndex = -1,
        };
        AddChild(_road);
        _leftClosed = Close(_track.LeftEdge);
        _rightClosed = Close(_track.RightEdge);
    }

    public override void _Process(double delta)
    {
        if (Engine.IsEditorHint()) return;
        Scale = new Vector2(ViewScale, ViewScale);
        var win = (Vector2)GetWindow().Size;
        Position = new Vector2(win.X * 0.5f, win.Y * 0.5f + VerticalOffset);
        if (_cars.Count == 0) Discover();
        QueueRedraw();
    }

    // Record + advance the replay on the sim clock so it's frame-rate independent.
    public override void _PhysicsProcess(double delta)
    {
        if (Engine.IsEditorHint()) return;
        if (_cars.Count == 0) { Discover(); return; }

        RecordLiveFrame();

        if (!_replayReady && AllHaveBest()) BuildReplaySet();
        if (_replayReady) AdvanceReplay();
    }

    // ---- recording ----

    private void RecordLiveFrame()
    {
        for (int i = 0; i < _cars.Count; i++)
        {
            RaceCar car = _cars[i];
            float d = car.EpisodeDistance;

            // EpisodeDistance is monotonic within an episode and snaps to ~0 on reset, so a drop means a
            // new episode just started: finalize the run we just watched, then begin a fresh buffer.
            if (d < _lastDist[i] - 1f || _cur[i].Count >= MaxRunFrames)
                FinalizeRun(i, _lastDist[i]);

            _cur[i].Add(new Frame { Pos = car.CarPos, Heading = car.CarHeading, Slip = car.CarSlip, Steer = car.CarSteerAngle });
            _lastDist[i] = d;
        }
    }

    private void FinalizeRun(int i, float distance)
    {
        if (_cur[i].Count >= MinRunFrames &&
            (_best[i] == null || distance > _best[i]!.Distance))
        {
            _best[i] = new Run { Frames = TrimTrailingStill(_cur[i]), Distance = distance };
        }
        _cur[i].Clear();
    }

    // Drop trailing frames where the car has effectively stopped (stalled on-track without crashing), so a
    // run ends at its last real motion. The replay take runs to the longest run, so this makes the take end
    // when the last actually-moving car stops — instead of holding for seconds on parked cars.
    private static Frame[] TrimTrailingStill(List<Frame> frames)
    {
        const float moveThresh = 0.5f;   // px/frame (~30px/s); below this the car is "stopped"
        int last = frames.Count - 1;
        while (last > 0 && (frames[last].Pos - frames[last - 1].Pos).Length() < moveThresh) last--;
        int keep = Math.Max(last + 1, Math.Min(frames.Count, MinRunFrames));
        var arr = new Frame[keep];
        for (int k = 0; k < keep; k++) arr[k] = frames[k];
        return arr;
    }

    private bool AllHaveBest()
    {
        for (int i = 0; i < _best.Length; i++)
            if (_best[i] == null) return false;
        return true;
    }

    // ---- replay ----

    private void BuildReplaySet()
    {
        _replay = new Run?[_cars.Count];
        _replayLen = 1;
        _winner = -1;
        float bestDist = float.NegativeInfinity;
        for (int i = 0; i < _cars.Count; i++)
        {
            _replay[i] = _best[i];
            if (_best[i] == null) continue;
            _replayLen = Math.Max(_replayLen, _best[i]!.Frames.Length);
            if (_best[i]!.Distance > bestDist) { bestDist = _best[i]!.Distance; _winner = i; }
        }

        // Fade each car by how far behind the winner it finished: ratio 1 → opaque, small ratio → near-floor.
        _alpha = new float[_cars.Count];
        for (int i = 0; i < _cars.Count; i++)
        {
            if (_replay[i] == null) { _alpha[i] = 0f; continue; }
            float ratio = bestDist > 1e-3f ? Math.Clamp(_best[i]!.Distance / bestDist, 0f, 1f) : 1f;
            _alpha[i] = i == _winner ? 1f : GhostAlphaFloor + (1f - GhostAlphaFloor) * MathF.Pow(ratio, 1.4f);
        }

        _replayFrame = 0;
        _holdCounter = 0;
        _replayReady = true;
        Array.Clear(_skidPrevValid, 0, _skidPrevValid.Length);
    }

    private void AdvanceReplay()
    {
        LaySkidForFrame();

        _replayFrame++;
        if (_replayFrame >= _replayLen)
        {
            _replayFrame = _replayLen - 1;          // hold on the final positions
            if (++_holdCounter > EndHoldFrames)
                BuildReplaySet();                   // rebuild from current bests and run it again
        }
    }

    private void LaySkidForFrame()
    {
        for (int i = 0; i < _replay.Length; i++)
        {
            Run? run = _replay[i];
            if (run == null) continue;
            Frame fr = run.Frames[Math.Min(_replayFrame, run.Frames.Length - 1)];

            Vector2 f = Vector2.Right.Rotated(fr.Heading);
            Vector2 right = new(-f.Y, f.X);
            Vector2 rearC = fr.Pos - f * (GhostBodyLen * 0.66f);
            Vector2 l = rearC + right * GhostBodyWid;
            Vector2 r = rearC - right * GhostBodyWid;

            float off = _track.Progress(fr.Pos);
            bool onRoad = MathF.Abs(_track.SignedOffset(fr.Pos, off)) <= _track.HalfWidth;
            float slipFrac = Math.Clamp(MathF.Abs(fr.Slip) / 150f, 0f, 1f);

            if (_skidPrevValid[i] && onRoad && slipFrac > SkidSlip
                && (l - _skidPrev[i].l).LengthSquared() < 400f)
            {
                _skid.Add(_skidPrev[i].l); _skid.Add(l);
                _skid.Add(_skidPrev[i].r); _skid.Add(r);
            }
            _skidPrev[i] = (l, r);
            _skidPrevValid[i] = true;
        }
        if (_skid.Count > MaxSkidPoints) _skid.RemoveRange(0, 4000);
    }

    // ---- drawing ----

    public override void _Draw()
    {
        if (_cars.Count == 0) return;

        if (_skid.Count >= 2)
            DrawMultiline(_skid.ToArray(), new Color(0.05f, 0.05f, 0.06f, 0.22f), 2.4f);

        var kerb = new Color(0.92f, 0.92f, 0.96f, 0.85f);
        DrawPolyline(_leftClosed, kerb, 2f, true);
        DrawPolyline(_rightClosed, kerb, 2f, true);

        Vector2 c0 = _track.SampleAt(0f);
        Vector2 n0 = _track.LeftNormalAt(0f);
        DrawLine(c0 + n0 * _track.HalfWidth, c0 - n0 * _track.HalfWidth, new Color(1f, 0.95f, 0.4f), 3f);

        if (!_replayReady)
        {
            // Still gathering the first run from each car — show them live and faint until the race is ready.
            foreach (var car in _cars)
                DrawCar(car.CarPos, car.CarHeading, car.CarSlip, car.CarSteerAngle, 0.28f, false);
            return;
        }

        // Non-winners first, winner on top so its ring/outline reads cleanly.
        for (int i = 0; i < _replay.Length; i++)
        {
            if (i == _winner || _replay[i] == null) continue;
            Frame fr = _replay[i]!.Frames[Math.Min(_replayFrame, _replay[i]!.Frames.Length - 1)];
            DrawCar(fr.Pos, fr.Heading, fr.Slip, fr.Steer, _alpha[i], false);
        }
        if (_winner >= 0 && _replay[_winner] != null)
        {
            Frame fr = _replay[_winner]!.Frames[Math.Min(_replayFrame, _replay[_winner]!.Frames.Length - 1)];
            DrawCar(fr.Pos, fr.Heading, fr.Slip, fr.Steer, 1f, true);
        }
    }

    private const float GhostBodyLen = 12.5f;   // half-length
    private const float GhostBodyWid = 6f;      // half-width

    private void DrawCar(Vector2 p, float heading, float slip, float steer, float alpha, bool leader)
    {
        Vector2 f = Vector2.Right.Rotated(heading);
        Vector2 right = new(-f.Y, f.X);
        float bl = leader ? 14.5f : GhostBodyLen;
        float bw = leader ? 7f : GhostBodyWid;
        float slipFrac = Math.Clamp(MathF.Abs(slip) / 150f, 0f, 1f);

        // wheels first (body overlaps their inner halves). Front pair steers with the recorded input.
        Vector2 steerDir = Vector2.Right.Rotated(heading + steer * 1.3f);
        var wheel = new Color(0.09f, 0.09f, 0.11f, alpha);
        DrawColoredPolygon(Quad(p - f * (bl * 0.66f) + right * bw, f, 4.5f, 2f), wheel);
        DrawColoredPolygon(Quad(p - f * (bl * 0.66f) - right * bw, f, 4.5f, 2f), wheel);
        DrawColoredPolygon(Quad(p + f * (bl * 0.62f) + right * bw, steerDir, 4.5f, 2f), wheel);
        DrawColoredPolygon(Quad(p + f * (bl * 0.62f) - right * bw, steerDir, 4.5f, 2f), wheel);

        // tapered body, tinted toward orange as it slides
        var body = new[]
        {
            p + f * bl,
            p + f * (bl * 0.55f) + right * bw,
            p - f * bl + right * bw,
            p - f * bl - right * bw,
            p + f * (bl * 0.55f) - right * bw,
        };
        var grip = new Color(0.32f, 0.64f, 0.95f);
        var drift = new Color(1f, 0.55f, 0.15f);
        Color col = grip.Lerp(drift, slipFrac);
        col.A = alpha;
        DrawColoredPolygon(body, col);

        // cockpit
        DrawColoredPolygon(Quad(p + f * (bl * 0.04f), f, bl * 0.34f, bw * 0.52f),
            new Color(0.04f, 0.05f, 0.08f, alpha * 0.9f));

        if (leader)
        {
            DrawPolyline(Close(body), new Color(1f, 1f, 1f, 0.95f), 2f, true);
            DrawArc(p, bl + 9f, 0f, MathF.Tau, 28, new Color(1f, 0.95f, 0.4f, 0.9f), 2f, true);
        }
    }

    private static Vector2[] Quad(Vector2 c, Vector2 dir, float halfLen, float halfWid)
    {
        Vector2 s = new(-dir.Y, dir.X);
        return new[]
        {
            c + dir * halfLen + s * halfWid, c + dir * halfLen - s * halfWid,
            c - dir * halfLen - s * halfWid, c - dir * halfLen + s * halfWid,
        };
    }

    // ---- SVG export (built with FluentSvg: github.com/mfagerlund/FluentSvg) --------------------

    /// <summary>True once a replay set has been built (every car has logged at least one best run).</summary>
    public bool ReplayReady => _replayReady;

    private const float SvgFps = 20f;        // playback rate the SMIL durations assume (recording is 60 Hz,
                                             // so 20 plays the motion at a relaxed ~3× slow-motion)
    private const int SvgTargetFrames = 140; // motion frames after subsampling (keeps the file small)
    private const int SvgHoldFrames = 20;    // hold on the crash/finish pose before the loop restarts (~1s)

    /// <summary>
    /// Write the current replay as a self-contained, looping <b>animated SVG</b> — the same fan-out the
    /// overlay draws (winner ringed + opaque, the rest faded by finish gap). Each car is a reusable
    /// <c>&lt;symbol&gt;</c> driven by SMIL translate + rotate so it steers to face its heading in a GitHub
    /// README. No screen-capture: it is rebuilt from each car's recorded pose trajectory. Assembled with
    /// <b>FluentSvg</b>, which formats every number invariant-culture regardless of the machine locale.
    /// </summary>
    public void ExportSvg(string absolutePath)
    {
        if (!_replayReady) { GD.PushWarning("[RaceOverlay] ExportSvg called before the replay was ready."); return; }

        // World bounds from the road edges, plus room for the car and ring.
        float minX = float.PositiveInfinity, minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity, maxY = float.NegativeInfinity;
        foreach (var p in _track.LeftEdge) Extend(p);
        foreach (var p in _track.RightEdge) Extend(p);
        void Extend(Vector2 p)
        {
            minX = MathF.Min(minX, p.X); minY = MathF.Min(minY, p.Y);
            maxX = MathF.Max(maxX, p.X); maxY = MathF.Max(maxY, p.Y);
        }
        float margin = _track.HalfWidth + 28f;
        minX -= margin; minY -= margin; maxX += margin; maxY += margin;

        // Shared lockstep timeline: subsample global frame indices, then hold the last one. Even spacing
        // needs no keyTimes — folding the hold frames into the value list makes the loop dwell on the end.
        int gstep = Math.Max(1, (int)MathF.Ceiling(_replayLen / (float)SvgTargetFrames));
        var idx = new List<int>();
        for (int g = 0; g < _replayLen; g += gstep) idx.Add(g);
        if (idx[^1] != _replayLen - 1) idx.Add(_replayLen - 1);
        int motionCount = idx.Count;
        for (int k = 0; k < SvgHoldFrames; k++) idx.Add(_replayLen - 1);
        string dur = Svg.Tos(idx.Count / SvgFps) + "s";

        var svg = new Svg(absolutePath, title: "Bascule — the racing fleet replay") { Margin = NVec.Zero };
        svg.AddRectangleFromTo(new NVec(minX, minY), new NVec(maxX, maxY)).SetFill("#15151a").ClearStroke();

        // Track: thick centerline = asphalt, thin edges = kerb, then the start line.
        svg.AddPath(ToNVec(_track.Centerline)).SetClosed().SetFill("transparent")
           .SetStroke("#5a5a60", _track.HalfWidth * 2f)
           .SetAttribute("stroke-linejoin", "round").SetAttribute("stroke-linecap", "round");
        svg.AddPath(ToNVec(_track.LeftEdge)).SetClosed().SetFill("transparent").SetStroke("#ebebf5", 2).SetStrokeOpacity(0.85f);
        svg.AddPath(ToNVec(_track.RightEdge)).SetClosed().SetFill("transparent").SetStroke("#ebebf5", 2).SetStrokeOpacity(0.85f);
        Vector2 c0 = _track.SampleAt(0f), n0 = _track.LeftNormalAt(0f);
        Vector2 sa = c0 + n0 * _track.HalfWidth, sb = c0 - n0 * _track.HalfWidth;
        svg.AddLine(new NVec(sa.X, sa.Y), new NVec(sb.X, sb.Y)).SetStroke("#fff266", 3);

        // Two reusable car symbols, nose toward +x: a small faded ghost and the ringed leader.
        BuildCarSymbol(svg, "ghost", GhostBodyLen, GhostBodyWid, leader: false);
        BuildCarSymbol(svg, "leader", 14.5f, 7f, leader: true);

        // Cars: non-winners first, winner last (drawn on top).
        for (int i = 0; i < _replay.Length; i++)
            if (i != _winner && _replay[i] != null) AddCar(svg, i, idx, dur, leader: false);
        if (_winner >= 0 && _replay[_winner] != null) AddCar(svg, _winner, idx, dur, leader: true);

        svg.SaveToFile();
        GD.Print($"[RaceOverlay] wrote {absolutePath} ({_replay.Length} cars, {motionCount} frames, {dur} loop).");
    }

    /// <summary>One replayed car: a <c>&lt;use&gt;</c> of the right symbol, driven across its recorded
    /// trajectory by translate + additive rotate so it steers to face its heading; faded by finish gap.</summary>
    private void AddCar(Svg svg, int i, List<int> idx, string dur, bool leader)
    {
        Run run = _replay[i]!;
        int len = run.Frames.Length;
        var pos = new List<NVec>(idx.Count);
        var head = new List<float>(idx.Count);
        foreach (int k in idx)
        {
            Frame fr = run.Frames[Math.Min(k, len - 1)];
            pos.Add(new NVec(fr.Pos.X, fr.Pos.Y));
            head.Add(fr.Heading * 180f / MathF.PI);
        }

        var car = svg.AddUse(leader ? "leader" : "ghost").SetAttribute("opacity", Svg.Tos(_alpha[i]));
        var move = svg.AddAnimateTranslate(car, dur, pos);
        var turn = svg.AddAnimateRotate(car, dur, head, additive: true);   // additive: composes with translate
        move.SetAttribute("repeatCount", "indefinite");
        turn.SetAttribute("repeatCount", "indefinite");
    }

    /// <summary>Define a car once as a &lt;symbol&gt; centred on the origin, nose toward +x: a tapered body
    /// (winner gets a white outline) plus a dark cockpit.</summary>
    private static void BuildCarSymbol(Svg svg, string id, float bl, float bw, bool leader)
    {
        var sym = svg.AddSymbol(id);
        var body = svg.AddPolygon(
            new NVec(bl, 0f), new NVec(bl * 0.55f, bw), new NVec(-bl, bw), new NVec(-bl, -bw), new NVec(bl * 0.55f, -bw))
            .SetFill("#52a3f2");
        if (leader) body.SetStroke("#ffffff", 1.6f); else body.ClearStroke();
        sym.AddChild(body);
        sym.AddChild(svg.AddPolygon(
            new NVec(0.38f * bl, 0.52f * bw), new NVec(-0.30f * bl, 0.52f * bw),
            new NVec(-0.30f * bl, -0.52f * bw), new NVec(0.38f * bl, -0.52f * bw))
            .SetFill("#0a0d14").ClearStroke());
    }

    private static IEnumerable<NVec> ToNVec(Vector2[] pts)
    {
        var list = new List<NVec>(pts.Length);
        foreach (var p in pts) list.Add(new NVec(p.X, p.Y));
        return list;
    }

    private void Discover()
    {
        _cars.Clear();
        Collect(GetTree().Root);
        int n = _cars.Count;
        _cur = new List<Frame>[n];
        _best = new Run?[n];
        _lastDist = new float[n];
        _skidPrev = new (Vector2, Vector2)[n];
        _skidPrevValid = new bool[n];
        for (int i = 0; i < n; i++) _cur[i] = new List<Frame>();
    }

    private void Collect(Node node)
    {
        if (node is RaceCar car) _cars.Add(car);
        foreach (var child in node.GetChildren()) Collect(child);
    }

    private static Vector2[] Close(Vector2[] pts)
    {
        if (pts.Length == 0) return pts;
        var closed = new Vector2[pts.Length + 1];
        Array.Copy(pts, closed, pts.Length);
        closed[^1] = pts[0];
        return closed;
    }
}
