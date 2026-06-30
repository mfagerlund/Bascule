#if TOOLS
using Godot;

namespace Tensotron.Godot;

/// <summary>
/// Editor entry point for the Tensotron plugin. Registered via <c>addons/tensotron/plugin.cfg</c>
/// and enabled under Project Settings -> Plugins. It adds the training dock (a bottom panel of live
/// reward/loss plots + Save Model) and the debugger plugin that feeds it telemetry from the running
/// game over the <c>tensotron:stats</c> channel.
/// </summary>
[Tool]
public partial class TensotronPlugin : EditorPlugin
{
    private TrainingDock? _dock;
    private TrainingDebuggerPlugin? _debugger;

    public override void _EnterTree()
    {
        _dock = new TrainingDock { Name = "Tensotron", Title = "Tensotron", DefaultSlot = EditorDock.DockSlot.Bottom };
        AddDock(_dock);

        _debugger = new TrainingDebuggerPlugin { Dock = _dock };
        AddDebuggerPlugin(_debugger);
    }

    public override void _ExitTree()
    {
        if (_debugger != null)
        {
            RemoveDebuggerPlugin(_debugger);
            _debugger = null;
        }
        if (_dock != null)
        {
            RemoveDock(_dock);
            _dock.QueueFree();
            _dock = null;
        }
    }
}
#endif
