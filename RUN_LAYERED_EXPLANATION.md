# `run_layered.py` Explanation

This file explains what `run_layered.py` does, how the layered DAG size is determined, and what files it creates.

## 1. Purpose

`run_layered.py` runs only the **layered DAG** experiments.

It:

1. builds layered Gaussian DAGs
2. creates conditioning tasks
3. runs three samplers:
   - single-site Gibbs
   - block Gibbs
   - random-walk Metropolis-Hastings
4. saves plots in `figures/`
5. saves CSV tables in `tables/`

## 2. What Parameters It Loops Over

The script loops over:

- `dimension`
- `depth`
- `sparsity`
- `correlation`
- `observation_strategy`
- `repetition`

For each configuration, the script compares three samplers:

- `single_site_gibbs(...)`
- `block_gibbs(...)`
- `random_walk_metropolis_hastings(...)`

All three samplers are run on the same conditioned posterior target for a fair comparison.

## 3. How `dimension` Depends on `depth` and Width

In the current implementation, the layered generator takes:

- total `dimension = d`
- total `depth = K`

and then splits the `d` nodes across the `K` layers as evenly as possible.

So width is **not** a direct input in the code right now.

Instead, width is implied by `dimension` and `depth`.

Mathematically, the average layer width is approximately

```math
\text{average width} \approx \frac{d}{K}
```

where:

```math
d = \text{dimension}, \qquad K = \text{depth}
```

The exact layer sizes are formed by dividing `dimension` as evenly as possible across the layers.

Examples:

- `dimension = 12`, `depth = 3` gives layer sizes `[4,4,4]`
- `dimension = 10`, `depth = 3` gives layer sizes `[4,3,3]`
- `dimension = 40`, `depth = 4` gives average width about `10`
- `dimension = 40`, `depth = 8` gives average width about `5`

So in the current code:

```math
\text{dimension} \approx \text{depth} \times \text{average width}
```

and you change width indirectly by changing either:

- `dimension`
- `depth`

## 4. How `correlation` Is Changed

`correlation` is currently used as a **proxy for dependence strength**, not as a direct correlation coefficient.

It controls the range of edge weights in the structural equations:

```math
X_j = \sum_{i \in \mathrm{pa}(j)} w_{ij} X_i + \varepsilon_j
```

The generator uses:

```python
weight_scale = correlation
min_abs_weight = max(0.05, correlation / 2.0)
```

So if `correlation = c`, then edge magnitudes satisfy roughly

```math
|w_{ij}| \in [\max(0.05, c/2), c]
```

Larger `correlation` means stronger linear dependence and usually a more correlated posterior.

## 5. Meaning of the Other Parameters

### Sparsity

`sparsity` is the probability that a possible edge between two adjacent layers is included.

If `i` is in layer `L_k` and `j` is in layer `L_{k+1}`, then:

```math
\mathbb{P}(i \to j) = \text{sparsity}
```

### Observation Strategy

The observation strategy controls how observed nodes are selected before posterior sampling.

The script supports:

- `uniform`
- `shallow`
- `deep`

These strategies change which nodes are more likely to be observed, and therefore change the posterior target seen by the samplers.

### Repetition

`repetition` means rerunning the same high-level configuration with fresh randomness for:

- the layered DAG itself
- the chosen observed nodes
- the sampler trajectories

This is useful because sampler performance can vary across random graph draws and random conditioning tasks.

## 6. What the Script Does Internally

At a high level, `run_layered.py` does:

1. parse command-line arguments
2. create a new run-specific output folder
3. create `figures/` and `tables/` inside that folder
4. save the run parameters and command
5. run `run_layered_experiment_grid(...)`
6. save the full experiment dataframe
7. create summary tables
8. draw plots for ESS, runtime, and error metrics

So it is the single command you use to regenerate all layered-model experiment outputs.

### Run-specific folders

Each execution now creates a different output folder, by default under:

```text
runs/
```

The folder name contains:

- a timestamp
- a compact summary of the parameter ranges

For example:

```text
runs/20260420_153210_dim20-80_dep4_sp0p3-0p5_corr0p4_obsuniform-shallow-deep_rep3/
```

Inside that run folder, the script writes:

- `figures/`
- `tables/`
- `run_config.json`
- `run_command.txt`

`run_config.json` stores the parameters used in the run.

`run_command.txt` stores a runnable command showing how the run was generated.

## 7. Outputs

The script saves:

### Base run files

- `run_config.json`
- `run_command.txt`

### Tables in `tables/`

- `tables/layered_experiment_results.csv`
- summary CSVs by `correlation`
- summary CSVs by `dimension`
- summary CSVs by `sparsity`

Examples:

- `tables/ess_summary_by_correlation.csv`
- `tables/ess_summary_by_dimension.csv`
- `tables/ess_summary_by_sparsity.csv`
- `tables/runtime_summary_by_correlation.csv`
- `tables/mean_rmse_summary_by_dimension.csv`
- `tables/covariance_error_summary_by_sparsity.csv`

### Figures in `figures/`

For each metric:

- ESS
- runtime
- mean RMSE
- variance RMSE
- covariance Frobenius error

it saves plots against:

- `correlation`
- `dimension`
- `sparsity`

Examples:

- `figures/ess_vs_correlation.png`
- `figures/ess_vs_dimension.png`
- `figures/ess_vs_sparsity.png`
- `figures/runtime_vs_correlation.png`
- `figures/mean_rmse_vs_dimension.png`
- `figures/covariance_error_vs_sparsity.png`

All figures are faceted by `observation_strategy`.

## 8. Metrics Used

The script plots and tabulates these metrics.

### ESS

`ess_mean` is the average effective sample size across latent coordinates.

### Runtime

`runtime_seconds` is the measured wall-clock runtime for one sampler run.

### Mean RMSE

```math
\mathrm{RMSE}_{\text{mean}}
=
\sqrt{\frac{1}{d}\sum_{j=1}^d (\hat{\mu}_j - \mu_j)^2}
```

### Variance RMSE

```math
\mathrm{RMSE}_{\text{var}}
=
\sqrt{\frac{1}{d}\sum_{j=1}^d (\hat{\Sigma}_{jj} - \Sigma_{jj})^2}
```

### Covariance Frobenius Error

```math
\|\hat{\Sigma} - \Sigma\|_F
=
\sqrt{\sum_{i=1}^d \sum_{j=1}^d (\hat{\Sigma}_{ij} - \Sigma_{ij})^2}
```

## 9. Log Scale

The plotting functions currently use log scale for:

- runtime
- error metrics

So RMSE and covariance-error plots are shown on a logarithmic y-axis. This is useful when differences between methods span orders of magnitude.

## 10. How To Run

Run the default experiment:

```bash
python3 run_layered.py
```

Run a smaller test:

```bash
python3 run_layered.py \
  --dimensions 8 10 \
  --depths 2 \
  --sparsities 0.3 0.5 \
  --correlations 0.4 \
  --observation-strategies uniform shallow \
  --repetitions 1 \
  --n-samples 100 \
  --burn-in 50 \
  --thinning 1
  
python3 run_layered.py \
  --dimensions 40 45 50 55 60 65 70 75 80 85 90 \
  --depths 4 \
  --sparsities 0.60 \
  --correlations 0.8\
  --observation-strategies deep uniform shallow \
  --repetitions 10 \
  --n-samples 1000 \
  --burn-in 500 
```

Use a custom run name:

```bash
python3 run_layered.py --run-name dimension_sweep
```

Change total dimension:

```bash
python3 run_layered.py --dimensions 20 40 80
```

Change depth:

```bash
python3 run_layered.py --depths 3 5 8
```

Change sparsity:

```bash
python3 run_layered.py --sparsities 0.1 0.3 0.5
```

Change dependence strength:

```bash
python3 run_layered.py --correlations 0.2 0.4 0.6 0.8
```

## 11. Main Command-Line Arguments

The most important arguments are:

- `--output-dir`
- `--run-name`
- `--dimensions`
- `--depths`
- `--sparsities`
- `--correlations`
- `--observation-strategies`
- `--observed-fraction`
- `--repetitions`
- `--n-samples`
- `--burn-in`
- `--thinning`
- `--rw-mh-proposal-scale`
- `--random-state`

## 12. Interpretation

`run_layered.py` does not fit a real dataset. It runs a controlled synthetic benchmark on layered Gaussian DAGs.

Its role is to:

- generate comparable sampler results
- study how performance changes with graph size and dependence
- create report-ready figures and tables

## 13. Short Summary

`run_layered.py` is the full layered-model experiment runner.

It:

- creates layered Gaussian DAGs
- conditions on selected observations
- runs three posterior samplers
- computes ESS, runtime, and error metrics
- saves plots in `figures/`
- saves tables in `tables/`
