#if TOOLS
using Godot;

namespace Tensotron.Godot;

/// <summary>
/// Editor entry point for the Tensotron plugin. Registered via <c>addons/tensotron/plugin.cfg</c>
/// and enabled under Project Settings -> Plugins. The training dock (live loss/reward/episode plots
/// and Save Model) is added here as it lands; for now this is the minimal valid plugin so the
/// project loads with the addon enabled.
/// </summary>
[Tool]
public partial class TensotronPlugin : EditorPlugin
{
    public override void _EnterTree()
    {
    }

    public override void _ExitTree()
    {
    }
}
#endif
