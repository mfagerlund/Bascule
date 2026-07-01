namespace Bascule.RL;

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

    /// <summary>Channel-by-channel structural comparison. The compiler-generated record <c>==</c> compares
    /// the <see cref="Channels"/> array by <em>reference</em>, so it does NOT detect two specs with the same
    /// channels — use this when validating that arenas/models share a layout. On mismatch,
    /// <paramref name="reason"/> names the first difference; <c>null</c> on a match.</summary>
    public bool ChannelsMatch(ControlSpec? other, out string? reason)
    {
        if (other is null) { reason = "the other control spec is null"; return false; }
        if (Count != other.Count) { reason = $"channel count {Count} vs {other.Count}"; return false; }
        for (int i = 0; i < Count; i++)
        {
            ControlChannel a = Channels[i], b = other.Channels[i];
            if (a.Name != b.Name || a.Min != b.Min || a.Max != b.Max || a.IsDiscrete != b.IsDiscrete)
            {
                reason = $"channel {i} '{a.Name}'({a.Min}..{a.Max}, discrete={a.IsDiscrete}) vs " +
                         $"'{b.Name}'({b.Min}..{b.Max}, discrete={b.IsDiscrete})";
                return false;
            }
        }
        reason = null;
        return true;
    }
}
