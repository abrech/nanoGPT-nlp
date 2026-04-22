import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

N_HEAD_FOR_N_EMBD = {
    128: 4,
    256: 4,
    384: 6,
}

VARIANT_NAME_KEYS = {
    'learning_rate': 'lr',
    'n_layer': 'nlayer',
    'n_embd': 'nembd',
    'block_size': 'bs',
    'dropout': 'dropout',
    'max_iters': 'maxiters',
}

SUPPORTED_PARAMS = list(VARIANT_NAME_KEYS.keys())


def format_learning_rate(value):
    s = f"{value:.0e}"
    s = s.replace('e-0', 'e').replace('e-', 'e').replace('+', '')
    return s


def format_variant_value(param, value):
    if param == 'learning_rate':
        return format_learning_rate(value)
    if param == 'dropout':
        return str(value).replace('.', 'p')
    return str(int(value))


def variant_name(param, value):
    short = VARIANT_NAME_KEYS[param]
    return f"{short}{format_variant_value(param, value)}"


def parse_baseline_values(baseline_text):
    """Extract baseline values for each supported hyperparam by literal-eval'ing the RHS."""
    values = {}
    for param in SUPPORTED_PARAMS:
        m = re.search(rf'^{param}\s*=\s*(.+?)\s*(?:#.*)?$', baseline_text, flags=re.M)
        if not m:
            raise ValueError(f"Could not find '{param}' in baseline config")
        try:
            values[param] = ast.literal_eval(m.group(1))
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse value for '{param}': {m.group(1)!r}") from e
    return values


def replace_config_value(text, key, value):
    replacement = f"{key} = {repr(value)}"
    new_text, count = re.subn(rf'^{key}\s*=.*$', replacement, text, flags=re.M)
    if count == 0:
        raise ValueError(f"Could not find key '{key}' in baseline config")
    return new_text


def make_variant_config(param, value, baseline_text, baseline_values, out_root):
    if value == baseline_values[param]:
        return None

    config_text = baseline_text
    config_text = replace_config_value(config_text, param, value)

    if param == 'n_embd':
        n_head = N_HEAD_FOR_N_EMBD.get(int(value))
        if n_head is None:
            raise ValueError(f"No n_head mapping for n_embd={value}")
        config_text = replace_config_value(config_text, 'n_head', n_head)

    variant = variant_name(param, value)
    new_out_dir = f"{out_root.as_posix().rstrip('/')}/{variant}"
    config_text = replace_config_value(config_text, 'out_dir', new_out_dir)
    config_text = replace_config_value(config_text, 'wandb_run_name', variant)
    return variant, config_text


def generate_variant_configs(baseline_config, hyperparameters_path, config_dir, out_root):
    baseline_text = baseline_config.read_text(encoding='utf-8')
    baseline_values = parse_baseline_values(baseline_text)

    with open(hyperparameters_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    config_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    for param in SUPPORTED_PARAMS:
        values = hyperparams.get(param, [])
        for value in values:
            if value == baseline_values[param]:
                continue
            result = make_variant_config(param, value, baseline_text, baseline_values, out_root)
            if result is None:
                continue
            variant, config_text = result
            config_path = config_dir / f'{variant}.py'
            config_path.write_text(config_text, encoding='utf-8')
            generated_paths.append(config_path)
            print(f"Generated config: {config_path}")

    return generated_paths


def run_sample_for_config(config_path, num_samples, dry_run=False):
    cmd = [sys.executable, str(ROOT / 'sample.py'), str(config_path), f'--num_samples={num_samples}']
    print('\n=== Running sample.py with', config_path.name, '===')
    print(' '.join(cmd))

    out_dir = None
    for line in config_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('out_dir'):
            out_dir = line.split('=', 1)[1].strip().strip("'\"")
            break

    if not out_dir:
        print("Could not determine out_dir from config, skipping")
        return

    ckpt_path = ROOT / out_dir / 'ckpt.pt'
    if not ckpt_path.exists():
        print(f"Skipping sample run because checkpoint not found: {ckpt_path}")
        return

    samples_path = ROOT / out_dir / 'samples.txt'
    print(f"Saving {num_samples} samples to {samples_path}")
    if dry_run:
        return
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(samples_path, 'w', encoding='utf-8') as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-hyperparameter variant config files and optionally run sample.py for each.'
    )
    parser.add_argument('--baseline-config', type=Path,
                        default=ROOT / 'config' / 'shakespeare_char' / 'baseline.py',
                        help='Path to the baseline config that variants are derived from')
    parser.add_argument('--hyperparameters', type=Path,
                        default=ROOT / 'hyperparameters.config.json',
                        help='JSON file with lists of hyperparameter values to sweep')
    parser.add_argument('--config-dir', type=Path, default=None,
                        help='Where to write the generated variant configs '
                             '(defaults to the baseline config directory)')
    parser.add_argument('--out-root', type=Path, default=None,
                        help="Parent dir used for each variant's out_dir "
                             "(defaults to out/<baseline-config parent name>)")
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to generate per variant')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate variant config files; do not run sample.py')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without actually running sample.py')
    args = parser.parse_args()

    baseline_config = args.baseline_config.resolve()
    if not baseline_config.exists():
        parser.error(f"Baseline config not found: {baseline_config}")

    config_dir = (args.config_dir or baseline_config.parent).resolve()
    out_root = (args.out_root or (ROOT / 'out' / baseline_config.parent.name)).resolve()

    try:
        out_root_rel = out_root.relative_to(ROOT)
    except ValueError:
        out_root_rel = out_root

    generated = generate_variant_configs(
        baseline_config=baseline_config,
        hyperparameters_path=args.hyperparameters,
        config_dir=config_dir,
        out_root=out_root_rel,
    )

    if args.generate_only:
        print(f"Generated {len(generated)} variant config(s).")
        return

    for config_path in generated:
        run_sample_for_config(config_path, args.num_samples, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
