using System;
using Godot;
using Bascule.RL;

namespace Bascule.Godot.Examples;

/// <summary>
/// The observation role: reports the geometry the policy needs to aim and shoot — the bearing to the
/// target as (sin, cos) of the error, the last aim command, and the target's normalized orbital speed
/// (so the policy can lead a moving target). A separate node from the gun and the reward, discovered by
/// its <see cref="IObservationSource"/> interface alone. Mirrors <c>TurretEnv.GetState</c> exactly.
/// </summary>
[Tool]
[GlobalClass]
public partial class TargetSensor : Node, IObservationSource
{
    private TurretArena? _arena;

    public override void _Ready() => _arena = GetParent() as TurretArena;

    public int Size => 4;

    public void Write(Span<float> dst)
    {
        float err = _arena?.BearingError ?? 0f;
        dst[0] = MathF.Sin(err);
        dst[1] = MathF.Cos(err);
        dst[2] = _arena?.NormAim ?? 0f;
        dst[3] = _arena?.NormOmega ?? 0f;
    }
}
