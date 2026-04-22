from pathlib import Path
import sys
import os

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

from stats608_pipeline import (
    block_gibbs,
    block_sizes_from_layers,
    effective_sample_size,
    estimation_errors,
    generate_chain_dag,
    generate_layered_dag,
    generate_random_sparse_dag,
    latent_layer_blocks,
    node_structure_stats,
    observation_probabilities,
    observation_scores,
    plot_error,
    plot_ess,
    plot_runtime,
    posterior_latent_parameters,
    random_walk_metropolis_hastings,
    run_layered_experiment_grid,
    sample_observed_indices,
    select_observations,
    sample_from_dag,
    single_site_gibbs,
)


def main() -> None:
    chain = generate_chain_dag(dimension=8, random_state=0)
    layered = generate_layered_dag(dimension=12, depth=3, sparsity=0.6, random_state=1)
    random_dag = generate_random_sparse_dag(dimension=10, sparsity=0.25, max_parents=3, random_state=2)

    assert chain.adjacency.shape == (8, 8)
    assert layered.adjacency.shape == (12, 12)
    assert random_dag.adjacency.shape == (10, 10)
    assert sum(block_sizes_from_layers(layered)) == 12

    conditioning = select_observations(layered, observed_fraction=0.25, random_state=3)
    shallow_conditioning = select_observations(
        layered,
        observed_fraction=0.25,
        strategy="shallow",
        depth_weight=2.0,
        children_weight=1.0,
        descendant_weight=1.0,
        random_state=3,
    )
    deep_conditioning = select_observations(
        layered,
        observed_fraction=0.25,
        strategy="deep",
        depth_weight=2.0,
        children_weight=1.0,
        descendant_weight=1.0,
        random_state=3,
    )
    samples = sample_from_dag(random_dag, n_samples=400, random_state=4)
    shallow_indices = sample_observed_indices(layered, observed_count=3, strategy="shallow", random_state=5)
    deep_indices = sample_observed_indices(layered, observed_count=3, strategy="deep", random_state=6)

    ess = effective_sample_size(samples, max_lag=25)
    errors = estimation_errors(samples, random_dag.mean, random_dag.covariance)
    chain_stats = node_structure_stats(chain)
    uniform_probs = observation_probabilities(layered, strategy="uniform")
    shallow_probs = observation_probabilities(chain, strategy="shallow")
    deep_probs = observation_probabilities(chain, strategy="deep")
    shallow_scores = observation_scores(chain, strategy="shallow")
    deep_scores = observation_scores(chain, strategy="deep")
    sampler_target = select_observations(chain, observed_fraction=0.25, random_state=7)
    posterior_params = posterior_latent_parameters(
        chain,
        sampler_target["observed_indices"],
        sampler_target["observed_values"],
    )
    gibbs_result = single_site_gibbs(
        chain,
        sampler_target["observed_indices"],
        sampler_target["observed_values"],
        n_samples=1500,
        burn_in=500,
        thinning=2,
        random_state=8,
    )
    rw_mh_result = random_walk_metropolis_hastings(
        chain,
        sampler_target["observed_indices"],
        sampler_target["observed_values"],
        n_samples=1500,
        proposal_scale=0.6,
        burn_in=500,
        thinning=2,
        random_state=9,
    )
    gibbs_mean_error = np.max(
        np.abs(gibbs_result["samples"].mean(axis=0) - sampler_target["posterior_mean"])
    )
    rw_mh_mean_error = np.max(
        np.abs(rw_mh_result["samples"].mean(axis=0) - sampler_target["posterior_mean"])
    )
    layered_target = select_observations(layered, observed_fraction=0.25, random_state=10)
    layer_params = posterior_latent_parameters(
        layered,
        layered_target["observed_indices"],
        layered_target["observed_values"],
    )
    layer_blocks = latent_layer_blocks(layered, layer_params["latent_indices"])
    block_result = block_gibbs(
        layered,
        layered_target["observed_indices"],
        layered_target["observed_values"],
        n_samples=1500,
        burn_in=500,
        thinning=2,
        random_state=11,
    )
    block_mean_error = np.max(
        np.abs(block_result["samples"].mean(axis=0) - layered_target["posterior_mean"])
    )

    assert ess.shape == (10,)
    assert np.all(ess > 0.0)
    assert set(errors) == {"mean_rmse", "variance_rmse", "covariance_frobenius_error"}
    assert conditioning["posterior_mean"].shape == (12,)
    assert conditioning["posterior_covariance"].shape == (12, 12)
    assert conditioning["observation_strategy"] == "uniform"
    assert shallow_conditioning["observation_strategy"] == "shallow"
    assert deep_conditioning["observation_strategy"] == "deep"
    assert np.all(np.isin(shallow_indices, np.arange(12)))
    assert np.all(np.isin(deep_indices, np.arange(12)))
    assert np.array_equal(chain_stats["depth"], np.arange(8))
    assert np.array_equal(chain_stats["children_count"], np.array([1, 1, 1, 1, 1, 1, 1, 0]))
    assert np.array_equal(chain_stats["descendant_count"], np.array([7, 6, 5, 4, 3, 2, 1, 0]))
    assert np.allclose(uniform_probs, np.full(12, 1 / 12))
    assert np.all(np.diff(shallow_scores) < 0)
    assert np.all(np.diff(deep_scores) > 0)
    assert np.all(np.diff(shallow_probs) < 0)
    assert np.all(np.diff(deep_probs) > 0)
    assert posterior_params["posterior_precision"].shape[0] == chain.dimension - sampler_target["observed_indices"].size
    assert gibbs_result["samples"].shape == (1500, chain.dimension)
    assert rw_mh_result["samples"].shape == (1500, chain.dimension)
    assert block_result["samples"].shape == (1500, layered.dimension)
    assert gibbs_mean_error < 0.2
    assert rw_mh_mean_error < 0.35
    assert block_mean_error < 0.2
    assert 0.0 < rw_mh_result["acceptance_rate"] < 1.0
    assert sum(block.size for block in layer_blocks) == layer_params["latent_indices"].size
    assert len(block_result["blocks"]) == len(layer_blocks)

    experiment_df = run_layered_experiment_grid(
        dimensions=[8],
        depths=[2],
        sparsities=[0.5],
        correlations=[0.4],
        observation_strategies=["uniform", "shallow"],
        observed_fraction=0.25,
        n_repetitions=1,
        n_samples=200,
        burn_in=100,
        thinning=1,
        random_state=12,
    )
    assert experiment_df.shape[0] == 6
    assert set(experiment_df["method"]) == {"single_site_gibbs", "block_gibbs", "rw_mh"}
    assert set(experiment_df["observation_strategy"]) == {"uniform", "shallow"}
    assert {"runtime_seconds", "ess_mean", "mean_rmse", "variance_rmse", "covariance_frobenius_error"}.issubset(experiment_df.columns)
    ess_fig, ess_axes = plot_ess(experiment_df, x="correlation", facet="observation_strategy")
    runtime_fig, runtime_axes = plot_runtime(experiment_df, x="correlation", facet="observation_strategy")
    error_fig, error_axes = plot_error(experiment_df, metric="mean_rmse", x="correlation", facet="observation_strategy")
    assert ess_axes.shape == (1, 2)
    assert runtime_axes.shape == (1, 2)
    assert error_axes.shape == (1, 2)
    ess_fig.clf()
    runtime_fig.clf()
    error_fig.clf()

    print("Pipeline smoke test passed.")


if __name__ == "__main__":
    main()
