#if TOOLS
using Godot;

namespace Tensotron.Godot;

/// <summary>
/// The editor-side training panel (a bottom <see cref="EditorDock"/>). It never touches the trainer
/// directly — the editor and the running game are separate processes — but receives one
/// <c>tensotron:stats</c> message per PPO iteration over the debugger channel (see
/// <see cref="TrainingDebuggerPlugin"/>): the scalars to plot plus a fresh serialized model snapshot.
/// Reward and loss are drawn with the same <see cref="RewardGraph"/> the in-game HUD uses, and
/// <b>Save Model</b> writes the latest snapshot to a <c>.tres</c> here in the editor (no round-trip to
/// the game), so the file is visible immediately.
/// </summary>
[Tool]
public partial class TrainingDock : EditorDock
{
    private Label _status = null!;
    private Label _stats = null!;
    private RewardGraph _rewardGraph = null!;
    private RewardGraph _lossGraph = null!;
    private LineEdit _pathEdit = null!;
    private Button _saveButton = null!;
    private Label _saveStatus = null!;

    private byte[] _latestModel = System.Array.Empty<byte>();
    private float _bestReturn = float.NegativeInfinity;
    private int _lastIter;
    private float _lastReturn;
    private bool _hasData;

    public override void _Ready()
    {
        var root = new VBoxContainer { CustomMinimumSize = new Vector2(0, 260) };
        root.AddThemeConstantOverride("separation", 6);
        AddChild(root);

        var header = new HBoxContainer();
        header.AddChild(new Label { Text = "Tensotron — Training", SizeFlagsHorizontal = SizeFlags.ExpandFill });
        _status = new Label
        {
            Text = "Idle — press Play on a scene with a LearningAgent in Train mode.",
            HorizontalAlignment = HorizontalAlignment.Right,
        };
        header.AddChild(_status);
        root.AddChild(header);

        _stats = new Label { Text = "—" };
        root.AddChild(_stats);

        var graphs = new HBoxContainer { SizeFlagsVertical = SizeFlags.ExpandFill };
        graphs.AddThemeConstantOverride("separation", 8);
        _rewardGraph = new RewardGraph
        {
            Caption = "meanReturn / iter (bold = smoothed)",
            TrendColor = new Color(0.45f, 0.95f, 0.6f),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0, 150),
        };
        _lossGraph = new RewardGraph
        {
            Caption = "PPO loss / iter (bold = smoothed)",
            TrendColor = new Color(1f, 0.7f, 0.3f),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0, 150),
        };
        graphs.AddChild(_rewardGraph);
        graphs.AddChild(_lossGraph);
        root.AddChild(graphs);

        var footer = new HBoxContainer();
        footer.AddThemeConstantOverride("separation", 6);
        footer.AddChild(new Label { Text = "Save to" });
        _pathEdit = new LineEdit { Text = "res://models/policy.tres", SizeFlagsHorizontal = SizeFlags.ExpandFill };
        footer.AddChild(_pathEdit);
        _saveButton = new Button { Text = "Save Model", Disabled = true };
        _saveButton.Pressed += OnSavePressed;
        footer.AddChild(_saveButton);
        _saveStatus = new Label { Text = "" };
        footer.AddChild(_saveStatus);
        root.AddChild(footer);
    }

    /// <summary>A fresh run connected: wipe the plots and the captured snapshot.</summary>
    public void OnRunStarted()
    {
        _rewardGraph.Clear();
        _lossGraph.Clear();
        _bestReturn = float.NegativeInfinity;
        _latestModel = System.Array.Empty<byte>();
        _hasData = false;
        _saveButton.Disabled = true;
        _saveStatus.Text = "";
        _status.Text = "Training…";
        _stats.Text = "—";
    }

    /// <summary>The run disconnected (scene stopped). Keep the last snapshot so it can still be saved.</summary>
    public void OnRunStopped()
    {
        _status.Text = _hasData ? $"Run ended at iter {_lastIter}." : "Run ended.";
    }

    /// <summary>One PPO iteration's telemetry from the running game.</summary>
    public void OnStats(int iteration, float meanReturn, float loss, long episodes, float meanEpLen,
        long skipped, byte[] model)
    {
        _lastIter = iteration;
        _lastReturn = meanReturn;
        if (meanReturn > _bestReturn) _bestReturn = meanReturn;
        if (model is { Length: > 0 })
        {
            _latestModel = model;
            _saveButton.Disabled = false;
        }
        _hasData = true;
        _status.Text = "Training…";

        _rewardGraph.Push(meanReturn);
        if (float.IsFinite(loss)) _lossGraph.Push(loss);

        string text =
            $"iter {iteration}    meanReturn {meanReturn:0.##} (best {_bestReturn:0.##})    " +
            $"loss {loss:0.###}    episodes {episodes}    mean ep len {meanEpLen:0.#}";
        if (skipped > 0)
            text += $"\nstability   guarded {skipped} unstable update(s)";
        _stats.Text = text;
    }

    private void OnSavePressed()
    {
        if (_latestModel.Length == 0)
        {
            _saveStatus.Text = "no model yet";
            return;
        }

        string path = _pathEdit.Text.Trim();
        if (string.IsNullOrEmpty(path)) path = "res://models/policy.tres";
        if (!path.EndsWith(".tres") && !path.EndsWith(".res")) path += ".tres";

        string dir = path.GetBaseDir();
        if (!string.IsNullOrEmpty(dir) && !DirAccess.DirExistsAbsolute(dir))
            DirAccess.MakeDirRecursiveAbsolute(dir);

        var res = new ModelResource
        {
            Data = _latestModel,
            Notes = $"Saved from the training dock at iter {_lastIter}, meanReturn {_lastReturn:0.##}.",
        };
        Error err = ResourceSaver.Save(res, path);
        if (err == Error.Ok)
        {
            _saveStatus.Text = $"saved {path}";
            EditorInterface.Singleton.GetResourceFilesystem().Scan();
        }
        else
        {
            _saveStatus.Text = $"save failed: {err}";
        }
    }
}
#endif
