using System.Collections.Generic;
using Godot;

namespace Bascule.Godot;

/// <summary>
/// A small autoscaled sparkline for a stream of scalars — one mean-return point per PPO iteration is
/// the intended use. Each value is drawn twice: a faint raw line (the noisy short-horizon estimate) and
/// a bold EMA trend (the fair "is it improving" read through that noise), with a zero baseline whenever
/// the range straddles it. Reusable on its own — <see cref="TrainingHud"/> drives one, but any node can
/// <see cref="Push"/> samples into it.
/// </summary>
[Tool]   // also drawn inside the editor's training dock, so it must run as a tool script
[GlobalClass]
public partial class RewardGraph : Control
{
    /// <summary>EMA factor for the bold trend line (0..1; higher follows the raw line more closely).</summary>
    [Export(PropertyHint.Range, "0.01,1.0")] public float Smoothing { get; set; } = 0.3f;

    /// <summary>Colour of the bold trend line; the faint raw line reuses it at low alpha. Set it per
    /// series so stacked graphs (reward vs loss) are distinguishable.</summary>
    [Export] public Color TrendColor { get; set; } = new(0.45f, 0.95f, 0.6f);

    /// <summary>Optional caption drawn in the panel's top-left, so the graph is self-labeling.</summary>
    [Export] public string Caption { get; set; } = "";

    private readonly List<float> _values = new();

    /// <summary>Append a sample and redraw.</summary>
    public void Push(float value)
    {
        _values.Add(value);
        QueueRedraw();
    }

    /// <summary>Drop all samples (e.g. when a fresh run starts).</summary>
    public void Clear()
    {
        _values.Clear();
        QueueRedraw();
    }

    public override void _Draw()
    {
        var size = Size;
        DrawRect(new Rect2(Vector2.Zero, size), new Color(0f, 0f, 0f, 0.45f));        // panel
        DrawRect(new Rect2(Vector2.Zero, size), new Color(1f, 1f, 1f, 0.15f), false, 1f);

        if (!string.IsNullOrEmpty(Caption))
            DrawString(ThemeDB.FallbackFont, new Vector2(6, 13), Caption,
                HorizontalAlignment.Left, -1, 11, new Color(1f, 1f, 1f, 0.7f));

        if (_values.Count < 2) return;

        float min = float.MaxValue, max = float.MinValue;
        foreach (var v in _values) { if (v < min) min = v; if (v > max) max = v; }
        if (max - min < 1e-3f) max = min + 1f;

        float Y(float v) => size.Y * (1f - (v - min) / (max - min));

        if (min < 0f && max > 0f)   // zero baseline, if the range straddles it
        {
            float y0 = Y(0f);
            DrawLine(new Vector2(0, y0), new Vector2(size.X, y0), new Color(1f, 1f, 1f, 0.25f), 1f);
        }

        // Raw per-iteration value (faint): the noisy short-horizon, exploration-policy estimate.
        var rawColor = new Color(TrendColor.R, TrendColor.G, TrendColor.B, 0.35f);
        var raw = new Vector2[_values.Count];
        for (int i = 0; i < _values.Count; i++)
            raw[i] = new Vector2(size.X * i / (_values.Count - 1), Y(_values[i]));
        DrawPolyline(raw, rawColor, 1f, true);

        // EMA trend (bold): the fair read of "is it improving" through that per-iteration noise.
        var ema = new Vector2[_values.Count];
        float acc = _values[0];
        for (int i = 0; i < _values.Count; i++)
        {
            acc += Smoothing * (_values[i] - acc);
            ema[i] = new Vector2(size.X * i / (_values.Count - 1), Y(acc));
        }
        DrawPolyline(ema, TrendColor, 2.5f, true);
    }
}
