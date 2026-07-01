using Godot;
using Bascule.RL;

namespace Bascule.Godot.Examples;

/// <summary>
/// The reward role: surfaces the arena's per-step reward (aim shaping plus the shot payoff) and ends
/// the episode at the fixed step cap. The world reset itself lives on <see cref="TurretArena"/> (via
/// <see cref="IEpisodeReset"/>), so this node's own <see cref="ResetEpisode"/> is a no-op — it only
/// reports. A third separate node, discovered by <see cref="IRewardSource"/> alone.
/// </summary>
[Tool]
[GlobalClass]
public partial class HitReward : Node, IRewardSource
{
    private TurretArena? _arena;

    public override void _Ready() => _arena = GetParent() as TurretArena;

    public float Reward => _arena?.LastStepReward ?? 0f;
    public bool Done => _arena != null && _arena.Steps >= _arena.MaxSteps;
    public void ResetEpisode() { }   // world reset is TurretArena's job (IEpisodeReset)
}
