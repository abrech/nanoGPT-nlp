import argparse
import json
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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


def load_groups(groups_path):
    with open(groups_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    baseline_log = cfg.get("baseline_log", "baseline.log")
    groups = cfg.get("groups", {})
    return baseline_log, groups


def main():
    parser = argparse.ArgumentParser(description="Plot training logs grouped by hyperparameter.")
    parser.add_argument("--logs-dir", default="logs/shakespeare_char",
                        help="Directory that contains the .log files to plot")
    parser.add_argument("--plots-dir", default="plots",
                        help="Directory where the PNGs are written")
    parser.add_argument("--groups", default="plotting.config.json",
                        help="JSON file describing the baseline log and the hyperparameter groups")
    parser.add_argument("--val", action="store_true",
                        help="Also plot validation loss as a dashed lighter line")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    baseline_log, groups = load_groups(args.groups)
    baseline_path = os.path.join(args.logs_dir, baseline_log)
    bl_steps, bl_train, bl_val = parse_log(baseline_path)

    for hp_name, group in groups.items():
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
            path = os.path.join(args.logs_dir, logfile)
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
        ax.set_title(f"Loss vs. Step - varying {hp_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(args.plots_dir, f"{hp_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
