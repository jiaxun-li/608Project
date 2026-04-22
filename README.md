# STATS 608 Experiment Pipeline

Minimal utilities for building sparse linear Gaussian DAG experiments.

## What is included

- `generate_chain_dag(...)`
- `generate_layered_dag(...)`
- `generate_random_sparse_dag(...)`
- `sample_from_dag(...)` for ancestral sampling
- `select_observations(...)` for creating a conditioning task
- `GaussianDAG.posterior(...)` for exact Gaussian posterior mean/covariance
- `single_site_gibbs(...)`
- `block_gibbs(...)`
- `random_walk_metropolis_hastings(...)`
- `run_layered_experiment_grid(...)`
- `effective_sample_size(...)`
- `autocorrelation(...)`
- `estimation_errors(...)`

## Quick start

```python
from stats608_pipeline import (
    generate_layered_dag,
    sample_from_dag,
    select_observations,
    single_site_gibbs,
    block_gibbs,
    random_walk_metropolis_hastings,
    run_layered_experiment_grid,
    plot_ess,
    plot_runtime,
    plot_error,
    effective_sample_size,
    estimation_errors,
)

dag = generate_layered_dag(
    dimension=40,
    depth=4,
    sparsity=0.35,
    random_state=0,
)

prior_samples = sample_from_dag(dag, n_samples=2000, random_state=1)
conditioning = select_observations(dag, observed_fraction=0.25, random_state=2)
shallow_conditioning = select_observations(
    dag,
    observed_fraction=0.25,
    strategy="shallow",
    depth_weight=2.0,
    children_weight=1.0,
    descendant_weight=1.0,
    random_state=3,
)
deep_conditioning = select_observations(
    dag,
    observed_fraction=0.25,
    strategy="deep",
    depth_weight=2.0,
    children_weight=1.0,
    descendant_weight=1.0,
    random_state=4,
)
gibbs_result = single_site_gibbs(
    dag,
    conditioning["observed_indices"],
    conditioning["observed_values"],
    n_samples=2000,
    burn_in=500,
    thinning=2,
    random_state=5,
)
block_result = block_gibbs(
    dag,
    conditioning["observed_indices"],
    conditioning["observed_values"],
    n_samples=2000,
    burn_in=500,
    thinning=2,
    random_state=6,
)
mh_result = random_walk_metropolis_hastings(
    dag,
    conditioning["observed_indices"],
    conditioning["observed_values"],
    n_samples=2000,
    proposal_scale=0.6,
    burn_in=500,
    thinning=2,
    random_state=7,
)

ess = effective_sample_size(gibbs_result["samples"])
errors = estimation_errors(
    gibbs_result["samples"],
    conditioning["posterior_mean"],
    conditioning["posterior_covariance"],
)

experiment_df = run_layered_experiment_grid(
    dimensions=[20, 40],
    depths=[3, 4],
    sparsities=[0.2, 0.4],
    correlations=[0.3, 0.6],
    observation_strategies=["uniform", "shallow", "deep"],
    observed_fraction=0.25,
    n_repetitions=3,
    n_samples=2000,
    burn_in=500,
    thinning=2,
    random_state=0,
)

fig1, axes1 = plot_ess(experiment_df, x="correlation", facet="observation_strategy")
fig2, axes2 = plot_runtime(experiment_df, x="correlation", facet="observation_strategy")
fig3, axes3 = plot_error(experiment_df, metric="mean_rmse", x="correlation", facet="observation_strategy")
```

## Validation

Run:

```bash
python3 tests/test_pipeline.py
```

To generate layered experiment outputs:

```bash
python3 run_layered.py
```

This creates a new run folder under `runs/`, and inside that folder writes:

- plots to `figures/`
- CSV tables to `tables/`
- run parameters to `run_config.json`
