using Godot;
using Bascule.RL;

namespace Bascule.Godot;

/// <summary>
/// A trained policy saved as a Godot <see cref="Resource"/> (.tres / .res). It is the thin Godot-side
/// wrapper around <see cref="ModelSerializer"/>: the portable model blob lives in <see cref="Data"/>
/// (stored as a PackedByteArray in the resource file), and <see cref="LoadPolicy"/> turns it back into
/// an inference-ready <see cref="InferencePolicy"/>. This is what the editor's Save Model writes and
/// what Inference mode loads to ship.
/// </summary>
[Tool]
[GlobalClass]
public partial class ModelResource : Resource
{
    /// <summary>The serialized model blob (see <see cref="ModelSerializer"/>): sizes, ControlSpec and weights.</summary>
    [Export]
    public byte[] Data { get; set; } = System.Array.Empty<byte>();

    /// <summary>Free-text note (e.g. which scene/run produced this), for the inspector.</summary>
    [Export(PropertyHint.MultilineText)]
    public string Notes { get; set; } = "";

    /// <summary>True once a model has been stored.</summary>
    public bool HasModel => Data.Length > 0;

    /// <summary>Serialize a trained network + its control layout into a new resource.</summary>
    public static ModelResource From(ActorCritic actorCritic, ControlSpec controls)
        => new() { Data = ModelSerializer.Save(actorCritic, controls) };

    /// <summary>Reconstruct the inference-ready policy from <see cref="Data"/>.</summary>
    public InferencePolicy LoadPolicy()
    {
        if (!HasModel)
            throw new System.InvalidOperationException("ModelResource has no model data to load.");
        return ModelSerializer.Load(Data);
    }
}
