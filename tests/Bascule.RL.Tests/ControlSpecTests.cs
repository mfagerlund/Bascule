namespace Bascule.RL.Tests;

public class ControlSpecTests
{
    [Fact]
    public void Count_matches_channels_and_ranges_are_preserved()
    {
        var spec = new ControlSpec(new[]
        {
            new ControlChannel("YawDelta", -1f, 1f),
            new ControlChannel("PitchDelta", -1f, 1f),
            new ControlChannel("Fire", 0f, 1f, IsDiscrete: true),
        });

        Assert.Equal(3, spec.Count);
        Assert.False(spec.Channels[0].IsDiscrete);
        Assert.True(spec.Channels[2].IsDiscrete);
        Assert.Equal(-1f, spec.Channels[0].Min);
        Assert.Equal(1f, spec.Channels[2].Max);
    }
}
