import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

STEP_PATTERN = re.compile(
    r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)"
)


def parse_log(filepath):
    steps, train_losses, val_losses = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = STEP_PATTERN.match(line)
            if m:
                steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    return steps, train_losses, val_losses


def plot_all(runs, loss_index, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in runs:
        steps = data[0]
        losses = data[loss_index]
        ax.plot(steps, losses, label=name, linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot train and val loss vs. step across every log in a directory."
    )
    parser.add_argument("--logs-dir", default="logs/shakespeare_char",
                        help="Directory containing the .log files to plot")
    parser.add_argument("--plots-dir", default="plots",
                        help="Directory where the PNGs are written")
    parser.add_argument("--pattern", default="*.log",
                        help="Glob pattern for log files inside --logs-dir")
    parser.add_argument("--train-out", default="assignment_train_loss.png",
                        help="Filename for the training-loss figure (inside --plots-dir)")
    parser.add_argument("--val-out", default="assignment_val_loss.png",
                        help="Filename for the validation-loss figure (inside --plots-dir)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(logs_dir.glob(args.pattern))
    if not log_files:
        print(f"No logs matching {args.pattern} in {logs_dir}")
        return

    runs = []
    for log_path in log_files:
        parsed = parse_log(log_path)
        if not parsed[0]:
            print(f"Skipping {log_path.name}: no 'step ...' lines found")
            continue
        runs.append((log_path.stem, parsed))

    if not runs:
        print("Nothing to plot, all logs were empty.")
        return

    plot_all(runs, loss_index=1,
             title=f"Training loss vs. step ({logs_dir})",
             out_path=os.path.join(plots_dir, args.train_out))
    plot_all(runs, loss_index=2,
             title=f"Validation loss vs. step ({logs_dir})",
             out_path=os.path.join(plots_dir, args.val_out))


if __name__ == "__main__":
    main()
