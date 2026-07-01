using Godot;

namespace Bascule.Godot.Examples;

/// <summary>
/// "Show me what it learned": loads the model the training demo saved and runs it in Inference mode on a
/// grid of pucks. Windowed, you watch each puck hug its green target and swerve away whenever the red
/// enemy's danger ring closes in. Headless with <c>++ --svg=&lt;path&gt;</c>, it records one full episode
/// from the first arena and writes it as an animated SVG (used to generate the README assets).
/// </summary>
[GlobalClass]
public partial class PuckWatchDemo : Node2D
{
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Where to load the trained model from (matches <c>PuckWorldDemo.SaveModelPath</c>).</summary>
    [Export] public string ModelPath { get; set; } = "user://puckworld_model.tres";

    [Export] public Vector2I WindowSize { get; set; } = new(1080, 860);

    /// <summary>Headless run quits after this many physics ticks (the watch is meant to be windowed).</summary>
    [Export] public int HeadlessRunTicks { get; set; } = 800;

    /// <summary>If set (or passed as <c>++ --svg=&lt;path&gt;</c>), a headless run records one full episode
    /// from the first arena, writes an animated SVG to this OS path, and quits.</summary>
    [Export] public string ExportSvgPath { get; set; } = "";

    private bool _ok;
    private int _ticks;
    private PuckArena? _first;
    private bool _exported;
    private PuckArena[] _arenas = System.Array.Empty<PuckArena>();
    private double _distSum;
    private long _dangerCount, _sampleCount;

    public override void _EnterTree()
    {
        // Eval must run on the SAME task geometry the model trained on — apply the same --ring/--discrete/etc.
        PuckArena.ApplyCmdlineOverrides();
        PuckArena.Recording = !string.IsNullOrEmpty(ExportSvgPath) || HasSvgArg();
    }

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        // ++ --model=<path> --svg=<path> let one scene generate an SVG per checkpoint without a scene each.
        foreach (string arg in OS.GetCmdlineUserArgs())
        {
            if (arg.StartsWith("--model=")) ModelPath = arg["--model=".Length..];
            else if (arg.StartsWith("--svg=")) ExportSvgPath = arg["--svg=".Length..];
        }

        if (!IsHeadless())
        {
            var win = GetWindow();
            win.Size = WindowSize;
            win.MoveToCenter();
        }

        Agent ??= FindAgent();
        if (Agent == null) { GD.PushError("PuckWatchDemo: no LearningAgent child found."); return; }

        if (!ResourceLoader.Exists(ModelPath))
        {
            GD.PushError($"[PuckWatchDemo] no model at '{ModelPath}'. Run PuckWorldDemo (training) first.");
            return;
        }
        var model = ResourceLoader.Load<ModelResource>(ModelPath);
        if (model == null || !model.HasModel)
        {
            GD.PushError($"[PuckWatchDemo] '{ModelPath}' did not load as a usable ModelResource.");
            return;
        }

        Agent.Model = model;
        Agent.StartRun();              // AutoStart=false, so we start after assigning the model
        _ok = true;
        GD.Print($"[PuckWatchDemo] loaded {ModelPath}; {Agent.ArenaCount} pucks on screen.");

        if (IsHeadless()) Engine.MaxFps = 0;
    }

    public override void _PhysicsProcess(double delta)
    {
        if (Engine.IsEditorHint() || !IsHeadless()) return;
        if (!_ok) { GetTree().Quit(); return; }   // headless: model failed to load — quit, don't spin forever
        _ticks++;

        // SVG-export mode: wait for the first arena to finish one full episode, write it, then quit.
        if (!string.IsNullOrEmpty(ExportSvgPath))
        {
            _first ??= FindArena(this);
            if (!_exported && _first is { HasEpisode: true })
            {
                _first.ExportSvg(ExportSvgPath);
                _exported = true;
                GetTree().Quit();
                return;
            }
            if (_ticks >= 8000)   // safety: an episode is MaxSteps long, so this is generous
            {
                GD.PushError($"[PuckWatchDemo] no episode recorded after {_ticks} ticks; no SVG written to {ExportSvgPath}.");
                GetTree().Quit();
            }
            return;
        }

        // Low-variance grading: average puck→target distance and danger-time across every arena and tick.
        if (_arenas.Length == 0) _arenas = CollectArenas(this);
        foreach (var a in _arenas)
        {
            _distSum += a.TargetDistance;
            if (a.InDanger) _dangerCount++;
            _sampleCount++;
        }

        if (_ticks >= HeadlessRunTicks)
        {
            float meanDist = _sampleCount > 0 ? (float)(_distSum / _sampleCount) : float.NaN;
            float fracDanger = _sampleCount > 0 ? (float)_dangerCount / _sampleCount : float.NaN;
            GD.Print($"[PuckWatchDemo] EVAL over {_sampleCount} samples: meanTargetDist={meanDist:0.000}  " +
                     $"fracInDanger={fracDanger:0.000}  (good: dist<0.15, danger<0.10)");
            GetTree().Quit();
        }
    }

    private static PuckArena[] CollectArenas(Node root)
    {
        var list = new System.Collections.Generic.List<PuckArena>();
        void Rec(Node n) { if (n is PuckArena a) list.Add(a); foreach (var c in n.GetChildren()) Rec(c); }
        Rec(root);
        return list.ToArray();
    }

    private LearningAgent? FindAgent()
    {
        foreach (var child in GetChildren())
            if (child is LearningAgent la) return la;
        return null;
    }

    private static PuckArena? FindArena(Node node)
    {
        if (node is PuckArena a) return a;
        foreach (var child in node.GetChildren())
            if (FindArena(child) is { } found) return found;
        return null;
    }

    private static bool HasSvgArg()
    {
        foreach (string arg in OS.GetCmdlineUserArgs())
            if (arg.StartsWith("--svg=")) return true;
        return false;
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
