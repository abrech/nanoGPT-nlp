import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / 'config'
BASE_CONFIG_PATH = CONFIG_DIR / 'train_shakespeare_char_baseline.py'
HYPERPARAMETERS_PATH = ROOT / 'hyperparameters.config.json'

BASELINE = {
    'learning_rate': 1e-3,
    'n_layer': 6,
    'n_embd': 384,
    'dropout': 0.2,
    'max_iters': 5000,
    'block_size': 256,
}

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


def read_baseline_config():
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Baseline config not found: {BASE_CONFIG_PATH}")
    return BASE_CONFIG_PATH.read_text(encoding='utf-8')


def load_hyperparameters():
    with open(HYPERPARAMETERS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def replace_config_value(text, key, value):
    replacement = f"{key} = {repr(value)}"
    new_text, count = re.subn(rf'^{key}\s*=.*$', replacement, text, flags=re.M)
    if count == 0:
        raise ValueError(f"Could not find key '{key}' in baseline config")
    return new_text


def make_variant_config(param, value, baseline_text):
    if param not in BASELINE:
        raise ValueError(f"Unsupported hyperparameter: {param}")
    if value == BASELINE[param]:
        return None

    config_text = baseline_text
    config_text = replace_config_value(config_text, param, value)

    if param == 'n_embd':
        n_head = N_HEAD_FOR_N_EMBD.get(int(value))
        if n_head is None:
            raise ValueError(f"No n_head mapping for n_embd={value}")
        config_text = replace_config_value(config_text, 'n_head', n_head)

    variant = variant_name(param, value)
    config_text = replace_config_value(config_text, 'out_dir', f'out-shakespeare-{variant}')
    config_text = replace_config_value(config_text, 'wandb_run_name', variant)
    return variant, config_text


def generate_variant_configs():
    hyperparams = load_hyperparameters()
    baseline_text = read_baseline_config()
    generated_paths = []

    for param in ['learning_rate', 'n_layer', 'n_embd', 'block_size', 'dropout', 'max_iters']:
        values = hyperparams.get(param, [])
        for value in values:
            if value == BASELINE[param]:
                continue
            result = make_variant_config(param, value, baseline_text)
            if result is None:
                continue
            variant, config_text = result
            config_path = CONFIG_DIR / f'train_shakespeare_char_{variant}.py'
            config_path.write_text(config_text, encoding='utf-8')
            generated_paths.append(config_path)
            print(f"Generated config: {config_path}")

    return generated_paths


def run_sample_for_config(config_path, dry_run=False):
    cmd = [sys.executable, str(ROOT / 'sample.py'), str(config_path), '--num_samples=3']
    print('\n=== Running sample.py with', config_path.name, '===')
    print(' '.join(cmd))

    out_dir = None
    for line in config_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('out_dir'):
            out_dir = line.split('=', 1)[1].strip().strip("'\"")
            break

    if out_dir:
        ckpt_path = ROOT / out_dir / 'ckpt.pt'
        if not ckpt_path.exists():
            print(f"Skipping sample run because checkpoint not found: {ckpt_path}")
            return

        samples_path = ROOT / out_dir / 'samples.txt'
        print(f"Saving 3 samples to {samples_path}")
        if not dry_run:
            with open(samples_path, 'w', encoding='utf-8') as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        print("Could not determine out_dir from config, skipping")



def main():
    parser = argparse.ArgumentParser(description='Generate config variants and run sample.py for each one.')
    parser.add_argument('--generate-only', action='store_true', help='Only generate variant config files and do not run sample.py')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing sample.py')
    args = parser.parse_args()

    generated = generate_variant_configs()
    if args.generate_only:
        print(f"Generated {len(generated)} variant config(s).")
        return

    for config_path in generated:
        run_sample_for_config(config_path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
