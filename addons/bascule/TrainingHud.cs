using Godot;

namespace Bascule.Godot;

/// <summary>
/// A drop-in on-screen training overlay for any <see cref="LearningAgent"/>: add it as a sibling of the
/// agent, press play, and it shows iteration, mean return (with running best) and a live
/// <see cref="RewardGraph"/> of mean return per iteration — no per-demo plumbing. It finds the agent
/// itself (explicit <see cref="Agent"/>, else a sibling, else anywhere in the tree) and subscribes to
/// <c>IterationCompleted</c>.
///
/// It owns only the <em>generic</em> readout; domain-specific lines (a turret's hit-rate, say) are the
/// host's to compute — set <see cref="ExtraText"/> each frame and they appear above the graph legend.
/// In the editor and in headless runs it builds nothing, so it is inert wherever there is no viewport.
///
/// This is the in-game counterpart to the planned editor dock: the dock is the authored-time, dockable
/// version of the same loss/reward/episode plots; this is just enough feedback to watch a live run.
/// </summary>
[GlobalClass]
public partial class TrainingHud : CanvasLayer
{
    /// <summary>The agent to report on. If unset, the first <see cref="LearningAgent"/> sibling (then any
    /// in the tree) is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Heading shown on the first line (e.g. "Turret PPO").</summary>
    [Export] public string Title { get; set; } = "Training";

    /// <summary>If &gt; 0, the iteration line reads "n / Target" and the status flips to TRAINED once
    /// reached. 0 hides the target and just shows the live iteration count.</summary>
    [Export] public int TargetIterations { get; set; }

    /// <summary>Top-left corner of the text block (the graph sits just below it).</summary>
    [Export] public Vector2 HudPosition { get; set; } = new(14, 10);

    [Export] public int FontSize { get; set; } = 16;

    /// <summary>Draw the mean-return sparkline below the text.</summary>
    [Export] public bool ShowGraph { get; set; } = true;

    /// <summary>Draw a second sparkline of the mean PPO loss per iteration.</summary>
    [Export] public bool ShowLoss { get; set; } = true;

    [Export] public Vector2 GraphSize { get; set; } = new(300, 110);

    /// <summary>Stack the graphs vertically (true, a narrow left band) or place them side by side
    /// (false, a short top band). Pick whichever leaves the arenas clear in a given demo.</summary>
    [Export] public bool StackGraphs { get; set; } = true;

    /// <summary>Optionally resize and re-center the game window when the overlay builds — handy for demos
    /// whose default window is too small to leave the HUD a clear band. Off by default.</summary>
    [Export] public bool ConfigureWindow { get; set; }

    [Export] public Vector2I WindowSize { get; set; } = new(1300, 700);

    /// <summary>Domain-specific lines (already newline-separated, no trailing newline) inserted between
    /// the mean-return line and the graph legend. Set it from the host's <c>_Process</c>; default empty.</summary>
    public string ExtraText { get; set; } = "";

    private Label? _hud;
    private RewardGraph? _graph;
    private RewardGraph? _lossGraph;
    private float _lineHeight;
    private float _bestReturn = float.NegativeInfinity;

    public override void _Ready()
    {
        // Nothing to draw without a viewport — stay inert in the editor and in headless runs.
        if (Engine.IsEditorHint() || DisplayServer.GetName() == "headless") return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushWarning("TrainingHud: no LearningAgent found; the overlay will be empty.");
            return;
        }
        Agent.IterationCompleted += OnIteration;

        if (ConfigureWindow)
        {
            var win = GetWindow();
            win.Size = WindowSize;
            win.MoveToCenter();
        }

        BuildHud();
    }

    private void BuildHud()
    {
        _hud = new Label { Position = HudPosition };
        _hud.AddThemeFontSizeOverride("font_size", FontSize);
        _hud.AddThemeColorOverride("font_color", new Color(0.92f, 0.96f, 1f));
        _hud.AddThemeColorOverride("font_outline_color", new Color(0f, 0f, 0f));
        _hud.AddThemeConstantOverride("outline_size", 5);
        AddChild(_hud);

        _lineHeight = _hud.GetThemeFont("font").GetHeight(FontSize)
                      + _hud.GetThemeConstant("line_spacing");

        if (ShowGraph)
        {
            _graph = new RewardGraph
            {
                Size = GraphSize,
                TrendColor = new Color(0.45f, 0.95f, 0.6f),
                Caption = "meanReturn / iter  (bold = smoothed)",
            };
            AddChild(_graph);
        }
        if (ShowLoss)
        {
            _lossGraph = new RewardGraph
            {
                Size = GraphSize,
                TrendColor = new Color(1f, 0.7f, 0.3f),
                Caption = "PPO loss / iter  (bold = smoothed)",
            };
            AddChild(_lossGraph);
        }
    }

    public override void _Process(double delta)
    {
        if (_hud == null || Agent == null) return;

        string status = TargetIterations > 0 && Agent.Iteration >= TargetIterations
            ? "TRAINED (still running)" : "TRAINING";
        string iterLine = TargetIterations > 0
            ? $"iteration   {Agent.Iteration} / {TargetIterations}"
            : $"iteration   {Agent.Iteration}";
        string best = float.IsNegativeInfinity(_bestReturn) ? "—" : _bestReturn.ToString("0.0");

        int lines = 4;
        string text =
            $"{Title} — {status}\n" +
            $"{iterLine}\n" +
            $"meanReturn  {Agent.LastMeanReturn:0.0}   (best {best})\n" +
            $"loss        {Agent.LastLoss:0.000}";
        if (Agent.TotalSkippedUpdates > 0)
        {
            text += $"\nstability   guarded {Agent.TotalSkippedUpdates} unstable update(s)";
            lines += 1;
        }
        if (!string.IsNullOrEmpty(ExtraText))
        {
            text += "\n" + ExtraText;
            lines += CountLines(ExtraText);
        }
        _hud.Text = text;

        // Lay the graphs out below the text, in stable "slots": stacked vertically or side by side.
        float top = HudPosition.Y + lines * _lineHeight + 8f;
        int slot = 0;
        if (_graph != null) _graph.Position = SlotPosition(top, slot++);
        if (_lossGraph != null) _lossGraph.Position = SlotPosition(top, slot++);
    }

    private Vector2 SlotPosition(float top, int slot) => StackGraphs
        ? new Vector2(HudPosition.X, top + slot * (GraphSize.Y + 10f))
        : new Vector2(HudPosition.X + slot * (GraphSize.X + 12f), top);

    private void OnIteration(int iteration, float meanReturn)
    {
        if (meanReturn > _bestReturn) _bestReturn = meanReturn;
        _graph?.Push(meanReturn);
        if (Agent != null) _lossGraph?.Push(Agent.LastLoss);   // LastLoss is set before this signal fires
    }

    private LearningAgent? FindAgent()
    {
        var parent = GetParent();
        if (parent != null)
            foreach (var c in parent.GetChildren())
                if (c is LearningAgent la) return la;
        return FindInTree(GetTree().Root);
    }

    private static LearningAgent? FindInTree(Node node)
    {
        if (node is LearningAgent la) return la;
        foreach (var child in node.GetChildren())
        {
            var found = FindInTree(child);
            if (found != null) return found;
        }
        return null;
    }

    private static int CountLines(string s)
    {
        int n = 1;
        foreach (var c in s) if (c == '\n') n++;
        return n;
    }
}
