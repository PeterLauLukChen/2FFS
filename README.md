# Paper 387 Supplementary Materials

This folder contains the code used to generate stochastic minimax-tree environments and benchmark 2FFS against tree-search baselines.

The code is intentionally lightweight and uses only the Python standard library. Use Python 3.10 or newer because the implementation uses modern type annotations such as `float | None`.

## Folder Structure

- `env/`: stochastic minimax-tree generator, validator, and a small `d2` smoke-test dataset.
- `method/`: implementations of MOON/2FFS and all benchmark methods.
- `method/run_benchmarks.py`: unified benchmark runner over a directory of generated trees.

Plotting notebooks, generated figures, and local benchmark outputs are not part of the clean reproducibility package.

## Methods

The unified runner supports the following method names:

- `twoffs`: 2FFS, the proposed two-fidelity minimax-tree search method (2FFS).
- `fast_minimax`: minimax-style fast expansion using deterministic biased fast intervals (Fast-Minimax).
- `mcts_bai`: BAI-MCTS-style stochastic sampling baseline (MCTS-BAI).
- `fixed_depth_slow`: fixed-depth fast expansion followed by slow-oracle sampling (slow-only).

## Environment Hyperparameters

The generator creates full alternating max/min trees. Each non-root node has a deterministic fast oracle and a stochastic slow oracle.

- `depth`: tree depth `D`.
- `branching`: branching factor `b`.
- `num_trees`: number of independent trees to generate.
- `seed`: base random seed; tree `i` uses `seed + i`.
- `beta`: fast-oracle bias strength. The fast bias envelope is
  `B(h)=beta*h/D`, where `h` is remaining depth.
- `slow_cost`: cost used by MOON/2FFS route comparisons for one slow sample.
- `slow_sigma`: standard deviation of the stochastic slow oracle.
- `root_gap_low`, `root_gap_high`: range for the root action gap.
- `internal_gap_low`, `internal_gap_high`: range for internal sibling gaps.
- `value_floor`, `value_ceiling`: clipping range for generated minimax values.

The paper's hard `D=5,b=8` setting used values of the following form:

```bash
--depth 5
--branching 8
--num-trees 30
--beta 0.45
--slow-cost 2
--slow-sigma 0.01
--root-gap-low 0.002
--root-gap-high 0.012
--internal-gap-low 0.0005
--internal-gap-high 0.006
```

## Generate Trees

Example: generate 100 hard `D=5,b=8` trees.

```bash
python3 code/env/generate_minimax_trees.py \
  --out-dir code/env/d5-b8-hard \
  --num-trees 100 \
  --depth 5 \
  --branching 8 \
  --seed 20260424 \
  --beta 0.45 \
  --slow-cost 2 \
  --slow-sigma 0.01 \
  --root-gap-low 0.002 \
  --root-gap-high 0.012 \
  --internal-gap-low 0.0005 \
  --internal-gap-high 0.006 \
  --progress-every 1 \
  --overwrite
```

Validate the generated trees:

```bash
python3 code/env/validate_minimax_trees.py \
  --env-dir code/env/d5-b8-hard \
  --expected-count 100
```

## Generate a Sweep

The generator also accepts a JSON sweep config.

```json
{
  "base_out_dir": "code/env",
  "defaults": {
    "num_trees": 30,
    "seed": 20260424,
    "slow_cost": 2,
    "slow_sigma": 0.01,
    "root_gap_low": 0.002,
    "root_gap_high": 0.012,
    "internal_gap_low": 0.0005,
    "internal_gap_high": 0.006
  },
  "sweep": [
    {"name": "d5-b8-beta030", "d": 5, "b": 8, "beta": 0.30},
    {"name": "d5-b8-beta045", "d": 5, "b": 8, "beta": 0.45},
    {"name": "d5-b8-beta060", "d": 5, "b": 8, "beta": 0.60}
  ]
}
```

Run the sweep:

```bash
python3 code/env/generate_minimax_trees.py \
  --sweep-config code/env/sweep_beta_d5_b8.json \
  --progress-every 1 \
  --overwrite
```

## Run Benchmarks

Run all methods on one environment directory:

```bash
python3 code/method/run_benchmarks.py \
  --env-dir code/env/d5-b8-hard \
  --out-dir code/results/d5-b8-hard \
  --methods twoffs fast_minimax mcts_bai ugape_mcts fixed_depth_slow \
  --fixed-depth 2 \
  --delta 0.05 \
  --twoffs-max-outer-rounds 200000 \
  --mcts-max-rounds 500000 \
  --ugape-max-rounds 500000 \
  --fixed-depth-max-rounds 500000 \
  --progress-every 1 \
  --overwrite
```

For a quick smoke test using the included `d2` dataset:

```bash
python3 code/method/run_benchmarks.py \
  --env-dir code/env/d2 \
  --out-dir code/results/d2-smoke \
  --methods twoffs fast_minimax \
  --fixed-depth 1 \
  --delta 0.05 \
  --progress-every 10 \
  --overwrite
```

## Output Files

The unified runner writes:

- `benchmark_results.jsonl`: one JSON row per tree and method.
- `benchmark_results.csv`: flat per-run table.
- `summary.json`: nested aggregate metrics.
- `summary_table.csv`: compact aggregate table for paper tables.
- `run_config.json`: benchmark configuration.

The main reported metrics are:

- `stopped_rate`: fraction of runs satisfying the fixed-confidence stopping rule.
- `correct_rate`: fraction of recommended actions equal to the true optimal root action.
- `sampling_count`: number of node samples or node visits, equal to fast queries plus slow queries (sampling count in paper).
- `computation_bpu_time`: implementation-level bookkeeping primitive units (operation count in paper).
- `runtime_sec`: wall-clock runtime in seconds.

`sampling_count` is the fairest cross-method sampling-efficiency measure.`computation_bpu_time` is a diagnostic for implementation overhead and should be interpreted together with runtime.

## Reproducing Paper-Style Ablations

Beta sweep:

```bash
python3 code/env/generate_minimax_trees.py \
  --sweep-config code/env/sweep_beta_d5_b8.json \
  --progress-every 1 \
  --overwrite
```

Sigma sweep:

```bash
python3 code/env/generate_minimax_trees.py \
  --sweep-config code/env/sweep_sigma_d5_b8.json \
  --progress-every 1 \
  --overwrite
```

Then run `code/method/run_benchmarks.py` on each generated environment. For
slow baselines, increasing `--mcts-max-rounds` and `--fixed-depth-max-rounds`
may be necessary on hard trees.
