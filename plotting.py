import argparse
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

LOGS_DIR = "logs"
PLOTS_DIR = "plots"

HYPERPARAM_GROUPS = {
    "learning_rate": {
        "baseline_val": "1e-3",
        "runs": {
            "train_shakespeare_char_lr1e2.log": "1e-2",
            "train_shakespeare_char_lr5e3.log": "5e-3",
            "train_shakespeare_char_lr5e4.log": "5e-4",
            "train_shakespeare_char_lr1e4.log": "1e-4",
        },
    },
    "dropout": {
        "baseline_val": "0.2",
        "runs": {
            "train_shakespeare_char_dropout0p0.log": "0.0",
            "train_shakespeare_char_dropout0p1.log": "0.1",
            "train_shakespeare_char_dropout0p4.log": "0.4",
        },
    },
    "n_layer": {
        "baseline_val": "6",
        "runs": {
            "train_shakespeare_char_nlayer2.log": "2",
            "train_shakespeare_char_nlayer4.log": "4",
            "train_shakespeare_char_nlayer8.log": "8",
        },
    },
    "n_embd": {
        "baseline_val": "384",
        "runs": {
            "train_shakespeare_char_nembd128.log": "128",
            "train_shakespeare_char_nembd256.log": "256",
        },
    },
    "max_iters": {
        "baseline_val": "5000",
        "runs": {
            "train_shakespeare_char_maxiters1000.log": "1000",
            "train_shakespeare_char_maxiters2500.log": "2500",
            "train_shakespeare_char_maxiters10000.log": "10000",
        },
    },
    "block_size": {
        "baseline_val": "256",
        "runs": {
            "train_shakespeare_char_bs64.log": "64",
            "train_shakespeare_char_bs128.log": "128",
        },
    },
}

STEP_PATTERN = re.compile(
    r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)"
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def lighten(hex_color, amount=0.4):
    r, g, b = mcolors.to_rgb(hex_color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)


def parse_log(filepath):
    steps, train_losses, val_losses = [], [], []
    with open(filepath, "r") as f:
        for line in f:
            m = STEP_PATTERN.match(line)
            if m:
                steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    return steps, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Plot training logs")
    parser.add_argument(
        "--val", action="store_true", help="Also plot validation loss"
    )
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    baseline_path = os.path.join(LOGS_DIR, "train_shakespeare_char_baseline.log")
    bl_steps, bl_train, bl_val = parse_log(baseline_path)

    for hp_name, group in HYPERPARAM_GROUPS.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        color = COLORS[0]
        ax.plot(
            bl_steps, bl_train, color=color,
            label=f"baseline ({group['baseline_val']})",
            linewidth=1.4, alpha=0.85,
        )
        if args.val:
            ax.plot(
                bl_steps, bl_val, color=lighten(color),
                label=f"baseline ({group['baseline_val']}) val",
                linewidth=1.4, alpha=0.85, linestyle="--",
            )

        for i, (logfile, val) in enumerate(group["runs"].items(), start=1):
            path = os.path.join(LOGS_DIR, logfile)
            steps, train_losses, val_losses = parse_log(path)
            color = COLORS[i % len(COLORS)]
            ax.plot(
                steps, train_losses, color=color,
                label=f"{hp_name}={val}",
                linewidth=1.4, alpha=0.85,
            )
            if args.val:
                ax.plot(
                    steps, val_losses, color=lighten(color),
                    label=f"{hp_name}={val} val",
                    linewidth=1.4, alpha=0.85, linestyle="--",
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss vs. Step — varying {hp_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(PLOTS_DIR, f"{hp_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
