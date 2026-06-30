using System;
using Godot;
using Tensotron.Rl;

namespace Tensotron.Godot.Examples;

/// <summary>
/// The control role of the turret arena: a continuous "Aim" channel (turn rate) and a discrete "Fire"
/// channel (shoot / hold) — the example's reason for existing, since "Fire" forces the categorical
/// policy head. It owns no state; <see cref="Apply"/> just forwards the action to its parent
/// <see cref="TurretArena"/>, which integrates the step. Being its own node (sibling to the sensor and
/// reward nodes) is the point: the trainer discovers it by the interface, not by what it is.
/// </summary>
[Tool]
[GlobalClass]
public partial class TurretGun : Node, IControlSurface
{
    private TurretArena? _arena;

    public override void _Ready() => _arena = GetParent() as TurretArena;

    public ControlSpec Spec { get; } = new(new[]
    {
        new ControlChannel("Aim", -1f, 1f),                     // continuous: normalized turn rate
        new ControlChannel("Fire", 0f, 1f, IsDiscrete: true),   // discrete: hold (0) or shoot (1)
    });

    public void Apply(ReadOnlySpan<float> action, float dt)
        => _arena?.Advance(action[0], (int)MathF.Round(action[1]) >= 1, dt);
}
