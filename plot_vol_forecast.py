import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def plot_live_forecast(symbol: str, outdir: str):
    symbol_upper = symbol.upper()
    json_path = os.path.join(
        outdir, symbol_upper, f"{symbol_upper.lower()}_VOL_live_forecast.json"
    )

    if not os.path.exists(json_path):
        print(f"[!] File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    ref_price = data.get("reference_price", 0)

    # Daily Levels
    high = data.get("projected_high", 0)
    low = data.get("projected_low", 0)
    peak = data.get("projected_peak", 0)
    bottom = data.get("projected_bottom", 0)

    # 5D Levels
    high_5d = data.get("projected_high_5d", 0)
    low_5d = data.get("projected_low_5d", 0)
    peak_5d = data.get("projected_peak_5d", 0)
    bottom_5d = data.get("projected_bottom_5d", 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor="#121212")
    fig.suptitle(
        f"{symbol_upper} Volatility Excursion Forecast",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.grid(color="#333333", linestyle="--", alpha=0.5)

    # --- Daily Plot ---
    ax1 = axes[0]
    ax1.set_title("1D Forecast (Tomorrow)", color="white")

    # Draw reference line
    ax1.axhline(
        ref_price, color="#aaaaaa", linestyle="-", linewidth=2, label="Latest Close"
    )

    # Draw standard range
    ax1.axhspan(low, high, color="#2c3e50", alpha=0.5, label="Expected Range")

    # Draw 90% peak/bottom lines
    ax1.axhline(
        peak, color="#e74c3c", linestyle="--", linewidth=2, label="90% Peak (Ceiling)"
    )
    ax1.axhline(
        bottom, color="#2ecc71", linestyle="--", linewidth=2, label="90% Bottom (Floor)"
    )

    # Annotations
    ax1.text(0.5, peak, f"  {peak:.2f}", color="#e74c3c", va="bottom", ha="left")
    ax1.text(0.5, high, f"  {high:.2f}", color="white", va="bottom", ha="left")
    ax1.text(
        0.5, ref_price, f"  {ref_price:.2f}", color="#aaaaaa", va="bottom", ha="left"
    )
    ax1.text(0.5, low, f"  {low:.2f}", color="white", va="top", ha="left")
    ax1.text(0.5, bottom, f"  {bottom:.2f}", color="#2ecc71", va="top", ha="left")

    ax1.set_xlim(0, 1)
    ax1.set_xticks([])
    ax1.legend(
        loc="upper left", facecolor="#1e1e1e", edgecolor="#333333", labelcolor="white"
    )

    # --- Weekly Plot ---
    ax2 = axes[1]
    ax2.set_title("5D Forecast (Next Week)", color="white")

    ax2.axhline(
        ref_price, color="#aaaaaa", linestyle="-", linewidth=2, label="Latest Close"
    )
    ax2.axhspan(
        low_5d, high_5d, color="#2c3e50", alpha=0.5, label="Expected Weekly Range"
    )
    ax2.axhline(
        peak_5d, color="#e74c3c", linestyle="--", linewidth=2, label="90% Weekly Peak"
    )
    ax2.axhline(
        bottom_5d,
        color="#2ecc71",
        linestyle="--",
        linewidth=2,
        label="90% Weekly Bottom",
    )

    ax2.text(0.5, peak_5d, f"  {peak_5d:.2f}", color="#e74c3c", va="bottom", ha="left")
    ax2.text(0.5, high_5d, f"  {high_5d:.2f}", color="white", va="bottom", ha="left")
    ax2.text(
        0.5, ref_price, f"  {ref_price:.2f}", color="#aaaaaa", va="bottom", ha="left"
    )
    ax2.text(0.5, low_5d, f"  {low_5d:.2f}", color="white", va="top", ha="left")
    ax2.text(0.5, bottom_5d, f"  {bottom_5d:.2f}", color="#2ecc71", va="top", ha="left")

    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    ax2.legend(
        loc="upper left", facecolor="#1e1e1e", edgecolor="#333333", labelcolor="white"
    )

    plt.tight_layout()
    out_path = os.path.join(
        outdir, symbol_upper, f"{symbol_upper.lower()}_VOL_dashboard.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Successfully generated VOL dashboard: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="/home/km/Universal-ML/")
    args = parser.parse_args()
    plot_live_forecast(args.symbol, args.outdir)
