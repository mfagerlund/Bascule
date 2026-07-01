#if TOOLS
using Godot;

namespace Bascule.Godot;

/// <summary>
/// Bridges the running game's training telemetry into the editor's <see cref="TrainingDock"/>. The game
/// (<see cref="LearningAgent"/>) emits one <c>tensotron:stats</c> message per PPO iteration over the
/// remote-debugger channel; this plugin claims the <c>tensotron</c> capture prefix, decodes each message,
/// and pushes it to the dock. It also clears the dock's plots when a fresh session starts/stops. This is
/// the only path between the two processes — the dock saves models from the bytes it receives here.
/// </summary>
[Tool]
public partial class TrainingDebuggerPlugin : EditorDebuggerPlugin
{
    public TrainingDock? Dock { get; set; }

    public override bool _HasCapture(string capture) => capture == "tensotron";

    public override bool _Capture(string message, global::Godot.Collections.Array data, int sessionId)
    {
        if (message != "tensotron:stats" && message != "stats") return false;   // not ours
        if (Dock != null && data.Count >= 7)
            Dock.OnStats(
                data[0].As<int>(), data[1].As<float>(), data[2].As<float>(),
                data[3].As<long>(), data[4].As<float>(), data[5].As<long>(),
                data[6].As<byte[]>());
        return true;
    }

    public override void _SetupSession(int sessionId)
    {
        var session = GetSession(sessionId);
        session.Started += () => Dock?.OnRunStarted();
        session.Stopped += () => Dock?.OnRunStopped();
    }
}
#endif
