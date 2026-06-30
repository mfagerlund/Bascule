namespace Tensotron.Rl;

public sealed record ControlChannel(string Name, float Min, float Max, bool IsDiscrete = false);

/// <summary>
/// The keystone abstraction. A control surface declares its channels, so a gun, a car, a leg joint,
/// and a shader param are all "just controls" to the optimizer — it never needs to know what the
/// thing actually is.
/// </summary>
public sealed record ControlSpec(ControlChannel[] Channels)
{
    /// <summary>Action-vector length the policy must produce — one value per channel.</summary>
    public int Count => Channels.Length;
}
