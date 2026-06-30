using Godot;

namespace Tensotron.Godot.Examples;

/// <summary>
/// The shipping half of the loop: load the <c>ModelResource</c> the training demo saved and run it in
/// Inference mode — greedy, GPU-free, no optimizer, no value head. This closes the round trip
/// Train → <c>.tres</c> → Inference entirely through Godot's own <c>ResourceSaver</c>/<c>ResourceLoader</c>.
///
/// The controller loads the model, hands it to a <see cref="LearningAgent"/> that was authored with
/// <c>AutoStart = false</c>, then calls <see cref="LearningAgent.StartRun"/> — the pattern for
/// configuring an agent before it initializes. It reports mean episode length, which for the trained
/// policy should sit well above the ~15–20 steps a random controller manages.
/// </summary>
[GlobalClass]
public partial class InferenceDemo : Node2D
{
    /// <summary>The Inference-mode agent to drive. If unset, the first <see cref="LearningAgent"/> child is used.</summary>
    [Export] public LearningAgent? Agent { get; set; }

    /// <summary>Where to load the trained model from (matches the training demo's save path).</summary>
    [Export] public string ModelPath { get; set; } = "user://cartpole_model.tres";

    /// <summary>Headless run quits after this many physics ticks.</summary>
    [Export] public int HeadlessRunTicks { get; set; } = 800;

    /// <summary>Physics rate used only in headless runs.</summary>
    [Export] public int HeadlessPhysicsTicksPerSecond { get; set; } = 600;

    private int _ticks;
    private bool _ok;

    public override void _Ready()
    {
        if (Engine.IsEditorHint()) return;

        Agent ??= FindAgent();
        if (Agent == null)
        {
            GD.PushError("InferenceDemo: no LearningAgent child found.");
            return;
        }

        if (!ResourceLoader.Exists(ModelPath))
        {
            GD.PushError($"[InferenceDemo] no model at '{ModelPath}'. Run the training demo (CartPoleDemo) first.");
            return;
        }

        var model = ResourceLoader.Load<ModelResource>(ModelPath);
        if (model == null || !model.HasModel)
        {
            GD.PushError($"[InferenceDemo] '{ModelPath}' did not load as a usable ModelResource.");
            return;
        }

        Agent.Model = model;
        Agent.StartRun();   // the agent is authored AutoStart=false, so we start it after assigning the model
        _ok = true;
        GD.Print($"[InferenceDemo] loaded {ModelPath}; running greedy inference on {Agent.ArenaCount} arenas.");

        if (IsHeadless())
        {
            Engine.PhysicsTicksPerSecond = HeadlessPhysicsTicksPerSecond;
            Engine.MaxFps = 0;
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!_ok || Engine.IsEditorHint()) return;
        _ticks++;

        if (_ticks % 200 == 0 && Agent != null)
            GD.Print($"[InferenceDemo] tick {_ticks}: meanEpisodeLength={Agent.MeanEpisodeLength:0.0} " +
                     $"over {Agent.EpisodesCompleted} episodes.");

        if (IsHeadless() && _ticks >= HeadlessRunTicks)
        {
            GD.Print($"[InferenceDemo] done: meanEpisodeLength={Agent?.MeanEpisodeLength:0.0} " +
                     $"over {Agent?.EpisodesCompleted} episodes. Quitting.");
            GetTree().Quit();
        }
    }

    private LearningAgent? FindAgent()
    {
        foreach (var child in GetChildren())
            if (child is LearningAgent la) return la;
        return null;
    }

    private static bool IsHeadless() => DisplayServer.GetName() == "headless";
}
