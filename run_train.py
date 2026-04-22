import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_configs(config_glob):
    return sorted(Path(p) for p in glob.glob(config_glob, recursive=True))


def run_train_for_config(config_path, log_dir, dry_run=False):
    cmd = [sys.executable, str(ROOT / 'train.py'), str(config_path)]
    log_path = log_dir / f'{config_path.stem}.log'
    print(f'=== Training with {config_path.name} ===')
    print(' '.join(cmd))
    print(f'Logging output to {log_path}')
    if dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def collect_summaries(summary_glob):
    summaries = []
    for summary_path in sorted(glob.glob(summary_glob, recursive=True)):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            summaries.append(data)
        except Exception as e:
            print(f"Warning: failed to load {summary_path}: {e}")
    return summaries


def print_summary_table(summaries):
    if not summaries:
        print('No training summaries found.')
        return
    headers = ['Experiment', 'LR', 'Layers', 'Embd', 'Block', 'Dropout', 'Iters', 'Train Loss', 'Val Loss', 'Time (min)']
    print('\n' + '| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for s in summaries:
        row = [
            s.get('experiment', ''),
            str(s.get('learning_rate', '')),
            str(s.get('n_layer', '')),
            str(s.get('n_embd', '')),
            str(s.get('block_size', '')),
            str(s.get('dropout', '')),
            str(s.get('max_iters', '')),
            f"{s.get('final_train_loss', ''):.4f}" if isinstance(s.get('final_train_loss'), (int, float)) else str(s.get('final_train_loss', '')),
            f"{s.get('final_val_loss', ''):.4f}" if isinstance(s.get('final_val_loss'), (int, float)) else str(s.get('final_val_loss', '')),
            f"{s.get('training_time_min', ''):.2f}" if isinstance(s.get('training_time_min'), (int, float)) else str(s.get('training_time_min', '')),
        ]
        print('| ' + ' | '.join(row) + ' |')


def main():
    parser = argparse.ArgumentParser(
        description='Run train.py for every config file matching a glob pattern.'
    )
    parser.add_argument('--config-glob', type=str,
                        default='config/shakespeare_char/optimal_*.py',
                        help='Glob pattern for config files to train, relative to repo root')
    parser.add_argument('--log-dir', type=Path,
                        default=ROOT / 'logs' / 'shakespeare_char',
                        help='Directory for per-run training logs')
    parser.add_argument('--summary-glob', type=str,
                        default='out/shakespeare_char/optimal_*/training_summary.json',
                        help='Glob pattern for training_summary.json files to aggregate at the end')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only run configs whose filename contains this substring')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing them')
    args = parser.parse_args()

    configs = find_configs(args.config_glob)
    if args.filter is not None:
        configs = [p for p in configs if args.filter in p.name]

    if not configs:
        print(f'No training config files matched: {args.config_glob}')
        return

    for config_path in configs:
        run_train_for_config(config_path, args.log_dir, dry_run=args.dry_run)

    if not args.dry_run:
        summaries = collect_summaries(args.summary_glob)
        print_summary_table(summaries)


if __name__ == '__main__':
    main()
