using Godot;

namespace Bascule.Godot.Examples;

/// <summary>
/// "Show me what it learned": loads the model the training demo saved and runs it in Inference mode on a
/// fleet of cars drawn on <b>one shared track</b> (see <see cref="RaceOverlay"/>) — leader opaque, the
/// rest faded ghosts. Each car starts at a random point on the lap and resets independently, so the
/// pack stays spread around the circuit and you can read the learned line at a glance.
/// </summary>
[GlobalClass]
public partial class RaceWatchDemo : Node2D
{
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Where to load the trained model from (matches <c>CarRaceDemo.SaveModelPath</c>).</summary>
    [Export] public string ModelPath { get; set; } = "user://drift_racer_model.tres";

    [Export] public Vector2I WindowSize { get; set; } = new(1100, 920);

    /// <summary>Headless run quits after this many physics ticks (the watch is meant to be windowed).</summary>
    [Export] public int HeadlessRunTicks { get; set; } = 600;

    /// <summary>If set (or passed as <c>++ --svg=&lt;path&gt;</c>), a headless run waits for the replay to be
    /// ready, writes an animated SVG of it to this OS path, and quits. Used to generate the README assets.</summary>
    [Export] public string ExportSvgPath { get; set; } = "";

    private bool _ok;
    private int _ticks;
    private RaceOverlay? _overlay;
    private bool _exported;

    public override void _EnterTree()
        => RaceCar.Overlay = true;   // set before LearningAgent spawns arenas (root enters the tree first)

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
        if (Agent == null) { GD.PushError("RaceWatchDemo: no LearningAgent child found."); return; }
        _overlay = FindOverlay();

        if (!ResourceLoader.Exists(ModelPath))
        {
            GD.PushError($"[RaceWatchDemo] no model at '{ModelPath}'. Run CarRaceDemo (training) first.");
            return;
        }
        var model = ResourceLoader.Load<ModelResource>(ModelPath);
        if (model == null || !model.HasModel)
        {
            GD.PushError($"[RaceWatchDemo] '{ModelPath}' did not load as a usable ModelResource.");
            return;
        }

        Agent.Model = model;
        Agent.StartRun();              // AutoStart=false, so we start after assigning the model
        _ok = true;
        GD.Print($"[RaceWatchDemo] loaded {ModelPath}; {Agent.ArenaCount} cars on one track.");

        if (IsHeadless()) { Engine.MaxFps = 0; }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (Engine.IsEditorHint() || !IsHeadless()) return;
        if (!_ok) { GetTree().Quit(); return; }   // headless: model failed to load — quit, don't spin forever
        _ticks++;

        // SVG-export mode: let the overlay record until a full replay is ready, write it, then quit.
        if (!string.IsNullOrEmpty(ExportSvgPath))
        {
            if (!_exported && _overlay != null && _overlay.ReplayReady)
            {
                _overlay.ExportSvg(ExportSvgPath);
                _exported = true;
                GetTree().Quit();
                return;
            }
            if (_ticks >= 8000)   // safety: some early policies never give every car a clean run
            {
                GD.PushError($"[RaceWatchDemo] replay not ready after {_ticks} ticks; no SVG written to {ExportSvgPath}.");
                GetTree().Quit();
            }
            return;
        }

        if (_ticks >= HeadlessRunTicks) GetTree().Quit();   // headless smoke-test guard
    }

    private LearningAgent? FindAgent()
    {
        foreach (var child in GetChildren())
            if (child is LearningAgent la) return la;
        return null;
    }

    private RaceOverlay? FindOverlay()
    {
        foreach (var child in GetChildren())
            if (child is RaceOverlay ov) return ov;
        return null;
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
