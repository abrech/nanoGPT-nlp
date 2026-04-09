import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / 'config'


def find_train_configs():
    return sorted(CONFIG_DIR.glob('train_shakespeare_char_*.py'))


LOG_DIR = ROOT / 'logs'


def run_train_for_config(config_path, dry_run=False):
    cmd = [sys.executable, str(ROOT / 'train.py'), str(config_path)]
    log_path = LOG_DIR / f'{config_path.stem}.log'
    print(f'=== Training with {config_path.name} ===')
    print(' '.join(cmd))
    print(f'Logging output to {log_path}')
    if dry_run:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def collect_summaries():
    summaries = []
    for summary_path in sorted(ROOT.glob('out-shakespeare-*/training_summary.json')):
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
    parser = argparse.ArgumentParser(description='Run train.py for all shakespeare char config files.')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing them')
    parser.add_argument('--include-baseline', action='store_true', help='Include the baseline config file in the run list')
    parser.add_argument('--filter', type=str, default=None, help='Only run configs matching this substring')
    args = parser.parse_args()

    configs = find_train_configs()
    if not args.include_baseline:
        configs = [p for p in configs if p.name != 'train_shakespeare_char_baseline.py']
    if args.filter is not None:
        configs = [p for p in configs if args.filter in p.name]

    if not configs:
        print('No training config files found.')
        return

    for config_path in configs:
        run_train_for_config(config_path, dry_run=args.dry_run)

    if not args.dry_run:
        summaries = collect_summaries()
        print_summary_table(summaries)


if __name__ == '__main__':
    main()
