# nanoGPT-nlp

A fork of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) to make working on the assignment easier.

## Structure

```
config/              one subdir per dataset, each holds train configs
  shakespeare_char/    baseline + all the hyperparam variants
  pride_and_prejudice/ + the _char variant
  shakespeare/         not used for assignment
  openwebtext/         not used for assignment
data/                one subdir per dataset + prepare.py
out/                 training output
  shakespeare_char/<variant>/
  pride_and_prejudice/<variant>/
  ...
logs/                console output of each training run, same <dataset>/<variant>.log layout
plots/               images written by plotting.py
assets/              not used for assignment
```

`model files are ignored in git to avoid a huge repo`

## Quickstart

```bash
# 1. prepare dataset
python data/shakespeare_char/prepare.py

# 2. train something
python train.py config/shakespeare_char/baseline.py

# 3. sample from the trained model
python sample.py --out_dir=out/shakespeare_char/baseline --num_samples=3

# 4. plot the loss curves, etc.
python plotting.py --logs-dir logs/shakespeare_char --val
```

## Scripts

### `data/<dataset>/prepare.py`

Uses the raw text (downloads it if it's not there yet) and tokenizes it:

- characters for `shakespeare_char` / `pride_and_prejudice_char`
- custom BPE for `pride_and_prejudice`

### `train.py`

Trains a model. Takes a config file as the first positional arg and then `--key=value` console args. Writes a checkpoint + a `training_summary.json` into the `out_dir` from the config. 

```bash
python train.py config/shakespeare_char/baseline.py --compile=False
```

### `sample.py`

Uses a model saved in an `out_dir` to generate some text.

```bash
python sample.py --out_dir=out/shakespeare_char/lr5e4 --num_samples=3
```

### `run_samples.py`

Takes a baseline config and a JSON of hyperparameters (see `hyperparameters.config.json`), and for each value that differs from the baseline it creates a config `config/<dataset>/<variant>.py` with the right `out_dir` / `wandb_run_name`. Also runs `sample.py` on each variant afterwards and saves the output to `out/<dataset>/<variant>/samples.txt`.

```bash
python run_samples.py \
    --baseline-config config/shakespeare_char/baseline.py \
    --hyperparameters hyperparameters.config.json
# or: add --generate-only to just write configs without sampling
```

### `run_train.py`

Runs `train.py` with the specified configs, capturing console output to `logs/<dataset>/<variant>.log`. Then it reads the `training_summary.json` from every matching `out/` folder and prints a markdown table with the final train/val loss and wall time.

```bash
python run_train.py \
    --config-glob "config/shakespeare_char/optimal_*.py" \
    --log-dir logs/shakespeare_char \
    --summary-glob "out/shakespeare_char/optimal_*/training_summary.json"
```

### `plotting.py`

Parses the `step N: train loss X, val loss Y` lines out of every `.log` file listed in `plotting.config.json`, then makes one image per hyperparameter group (learning_rate, dropout, n_layer, n_embd, max_iters, block_size). Pass `--val` to also draw the validation curve. The generated plots are not the once specified in the assignment, but they are useful to get an overview.

```bash
python plotting.py --logs-dir logs/shakespeare_char --plots-dir plots --val
```

### `plot_assignment.py`

Creates the assignment plot by using console outputs in `.log`.

```bash
python plot_assignment.py --logs-dir logs/shakespeare_char --plots-dir plots
# -> plots/assignment_train_loss.png
# -> plots/assignment_val_loss.png
```

### `hyperparameters.config.json`

The values `run_samples.py` uses. One list per hyperparameter, if a value equals the baseline's value it's skipped automatically:

```json
{
  "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
  "n_layer":       [2, 4, 6, 8],
  "n_embd":        [128, 256, 384],
  "block_size":    [64, 128, 256],
  "dropout":       [0.0, 0.1, 0.2, 0.4],
  "max_iters":     [1000, 2500, 5000, 10000]
}
```

### `plotting.config.json`

The dataset layout the plotter uses:

```json
{
  "baseline_log": "baseline.log",
  "groups": {
    "learning_rate": { "baseline_val": "1e-3", "runs": { "lr1e2.log": "1e-2", ... } },
    ...
  }
}
```

