
"""Enhanced experiment pipeline for the STATS 608 Gaussian DAG project."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

import experiment_pipeline as base


Array = np.ndarray
GaussianDAG = base.GaussianDAG

# Re-export useful base utilities
_rng = base._rng
generate_chain_dag = base.generate_chain_dag
generate_layered_dag = base.generate_layered_dag
generate_random_sparse_dag = base.generate_random_sparse_dag
sample_from_dag = base.sample_from_dag
node_structure_stats = base.node_structure_stats
observation_scores = base.observation_scores
observation_probabilities = base.observation_probabilities
sample_observed_indices = base.sample_observed_indices
observed_mask_from_fraction = base.observed_mask_from_fraction
select_observations = base.select_observations
posterior_latent_parameters = base.posterior_latent_parameters
_initialize_latent_state = base._initialize_latent_state
_embed_latent_samples = base._embed_latent_samples
latent_layer_blocks = base.latent_layer_blocks
single_site_gibbs = base.single_site_gibbs
block_gibbs = base.block_gibbs
random_walk_metropolis_hastings = base.random_walk_metropolis_hastings
autocorrelation = base.autocorrelation
effective_sample_size = base.effective_sample_size
estimation_errors = base.estimation_errors


def normalize_log_weights(log_weights: Sequence[float]) -> Array:
    log_weights = np.asarray(log_weights, dtype=float)
    if log_weights.ndim != 1:
        raise ValueError("log_weights must be one-dimensional")
    if log_weights.size == 0:
        return np.array([], dtype=float)
    max_log_weight = float(np.max(log_weights))
    shifted = np.exp(log_weights - max_log_weight)
    total = float(np.sum(shifted))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("could not normalize log weights")
    return shifted / total


def importance_weighted_ess(weights: Sequence[float]) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    normalized = weights / total
    denom = float(np.sum(normalized**2))
    return 0.0 if denom <= 0.0 else float(1.0 / denom)


def weighted_estimation_errors(
    samples: Array,
    weights: Sequence[float],
    exact_mean: Sequence[float],
    exact_covariance: Array,
) -> dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    exact_mean = np.asarray(exact_mean, dtype=float)
    exact_covariance = np.asarray(exact_covariance, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, dimension)")
    if weights.shape != (samples.shape[0],):
        raise ValueError("weights length must equal number of samples")
    normalized = weights / weights.sum()
    sample_mean = normalized @ samples
    centered = samples - sample_mean
    sample_covariance = (centered.T * normalized) @ centered
    diag_exact = np.diag(exact_covariance)
    diag_est = np.diag(sample_covariance)
    return {
        "mean_rmse": float(np.sqrt(np.mean((sample_mean - exact_mean) ** 2))),
        "variance_rmse": float(np.sqrt(np.mean((diag_est - diag_exact) ** 2))),
        "covariance_frobenius_error": float(np.linalg.norm(sample_covariance - exact_covariance, ord="fro")),
    }


def posterior_diagnostics(
    posterior_precision: Array,
    posterior_covariance: Array,
    *,
    tol: float = 1e-12,
) -> dict[str, float]:
    q = np.asarray(posterior_precision, dtype=float)
    cov = np.asarray(posterior_covariance, dtype=float)
    if q.shape != cov.shape or q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("posterior_precision and posterior_covariance must be square matrices of equal shape")
    n = q.shape[0]
    if n == 0:
        return {
            "posterior_condition_number": 1.0,
            "mean_abs_posterior_corr": 0.0,
            "max_abs_posterior_corr": 0.0,
            "precision_density": 0.0,
        }
    condition_number = float(np.linalg.cond(cov))
    if n == 1:
        return {
            "posterior_condition_number": condition_number,
            "mean_abs_posterior_corr": 0.0,
            "max_abs_posterior_corr": 0.0,
            "precision_density": 0.0,
        }
    variances = np.maximum(np.diag(cov), 0.0)
    std = np.sqrt(variances)
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)
    offdiag_upper = corr[np.triu_indices(n, k=1)]

    offdiag_precision = q.copy()
    np.fill_diagonal(offdiag_precision, 0.0)
    precision_density = float(np.count_nonzero(np.abs(offdiag_precision) > tol) / (n * (n - 1)))
    return {
        "posterior_condition_number": condition_number,
        "mean_abs_posterior_corr": float(np.mean(np.abs(offdiag_upper))),
        "max_abs_posterior_corr": float(np.max(np.abs(offdiag_upper))),
        "precision_density": precision_density,
    }


def target_structure_diagnostics(dag: GaussianDAG, observed_indices: Sequence[int]) -> dict[str, float | int]:
    observed = np.asarray(observed_indices, dtype=int)
    stats = node_structure_stats(dag)
    n_edges = int(np.count_nonzero(dag.adjacency))
    max_possible_edges = dag.dimension * (dag.dimension - 1) / 2
    graph_density = float(n_edges / max_possible_edges) if max_possible_edges > 0 else 0.0
    if observed.size == 0:
        observed_mean_depth = 0.0
        observed_mean_descendants = 0.0
        observed_mean_children = 0.0
    else:
        observed_mean_depth = float(np.mean(stats["depth"][observed]))
        observed_mean_descendants = float(np.mean(stats["descendant_count"][observed]))
        observed_mean_children = float(np.mean(stats["children_count"][observed]))
    return {
        "n_edges": n_edges,
        "graph_density": graph_density,
        "empirical_depth": int(len(dag.layers)),
        "observed_mean_depth": observed_mean_depth,
        "observed_mean_descendants": observed_mean_descendants,
        "observed_mean_children": observed_mean_children,
    }


def singleton_latent_blocks(latent_indices: Sequence[int]) -> list[Array]:
    latent = np.asarray(latent_indices, dtype=int)
    return [np.array([idx], dtype=int) for idx in range(latent.size)]


def topological_window_latent_blocks(
    dag: GaussianDAG,
    latent_indices: Sequence[int],
    max_block_size: int,
) -> list[Array]:
    if max_block_size < 1:
        raise ValueError("max_block_size must be positive")
    latent = np.asarray(latent_indices, dtype=int)
    latent_pos = {int(node): pos for pos, node in enumerate(latent)}
    ordered_positions = [latent_pos[int(node)] for node in dag.topological_order if int(node) in latent_pos]
    return [
        np.asarray(ordered_positions[start : start + max_block_size], dtype=int)
        for start in range(0, len(ordered_positions), max_block_size)
    ]


def random_latent_blocks(
    latent_indices: Sequence[int],
    max_block_size: int,
    random_state: int | np.random.Generator | None = None,
) -> list[Array]:
    if max_block_size < 1:
        raise ValueError("max_block_size must be positive")
    latent = np.asarray(latent_indices, dtype=int)
    rng = _rng(random_state)
    permutation = rng.permutation(latent.size)
    return [
        np.asarray(permutation[start : start + max_block_size], dtype=int)
        for start in range(0, latent.size, max_block_size)
    ]


def precision_bfs_latent_blocks(
    dag: GaussianDAG,
    latent_indices: Sequence[int],
    posterior_precision: Array,
    max_block_size: int,
    *,
    tol: float = 1e-12,
) -> list[Array]:
    if max_block_size < 1:
        raise ValueError("max_block_size must be positive")
    latent = np.asarray(latent_indices, dtype=int)
    q = np.asarray(posterior_precision, dtype=float)
    latent_dim = latent.size
    if q.shape != (latent_dim, latent_dim):
        raise ValueError("posterior_precision shape must match latent_indices")
    if latent_dim == 0:
        return []
    topo_rank = {int(node): pos for pos, node in enumerate(dag.topological_order)}
    precision_graph = np.abs(q) > tol
    np.fill_diagonal(precision_graph, False)
    remaining: set[int] = set(range(latent_dim))
    blocks: list[Array] = []
    while remaining:
        seed = min(remaining, key=lambda pos: topo_rank[int(latent[pos])])
        queue = [seed]
        queued = {seed}
        block: list[int] = []
        while queue and len(block) < max_block_size:
            pos = queue.pop(0)
            if pos not in remaining:
                continue
            remaining.remove(pos)
            block.append(pos)
            neighbors = np.flatnonzero(precision_graph[pos])
            neighbors = sorted(
                (int(nb) for nb in neighbors),
                key=lambda nb: (-abs(q[pos, nb]), topo_rank[int(latent[nb])]),
            )
            for nb in neighbors:
                if nb in remaining and nb not in queued:
                    queue.append(nb)
                    queued.add(nb)
        blocks.append(np.asarray(block, dtype=int))
    return blocks


def latent_blocks_by_strategy(
    dag: GaussianDAG,
    latent_indices: Sequence[int],
    posterior_precision: Array,
    *,
    strategy: str = "depth",
    max_block_size: int = 8,
    random_state: int | np.random.Generator | None = None,
    tol: float = 1e-12,
) -> list[Array]:
    if strategy in {"singletons", "single_site"}:
        return singleton_latent_blocks(latent_indices)
    if strategy in {"depth", "layers", "layer_depth", "topological_depth"}:
        return latent_layer_blocks(dag, latent_indices)
    if strategy in {"topological", "topological_window", "window"}:
        return topological_window_latent_blocks(dag, latent_indices, max_block_size=max_block_size)
    if strategy in {"random", "random_blocks"}:
        return random_latent_blocks(latent_indices, max_block_size=max_block_size, random_state=random_state)
    if strategy in {"precision_bfs", "moral_bfs", "posterior_precision"}:
        return precision_bfs_latent_blocks(
            dag=dag,
            latent_indices=latent_indices,
            posterior_precision=posterior_precision,
            max_block_size=max_block_size,
            tol=tol,
        )
    raise ValueError(
        "strategy must be one of: singletons, depth, topological_window, random_blocks, precision_bfs"
    )


def validate_block_partition(blocks: Sequence[Array], latent_dim: int) -> None:
    if latent_dim == 0:
        if len(blocks) != 0:
            raise ValueError("nonempty blocks for zero-dimensional latent target")
        return
    if not blocks:
        raise ValueError("blocks must be nonempty when latent_dim is positive")
    covered = np.sort(np.concatenate([np.asarray(block, dtype=int) for block in blocks]))
    expected = np.arange(latent_dim)
    if not np.array_equal(covered, expected):
        raise ValueError("blocks must form a partition of latent coordinates")


def block_precision_mass(posterior_precision: Array, blocks: Sequence[Array]) -> dict[str, float | int]:
    q = np.asarray(posterior_precision, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("posterior_precision must be square")
    n = q.shape[0]
    if n <= 1:
        return {
            "block_within_precision_mass": 1.0,
            "block_cut_ratio": 0.0,
            "n_blocks": int(len(blocks)),
            "mean_block_size": float(np.mean([len(b) for b in blocks])) if blocks else 0.0,
            "max_block_size_actual": int(max((len(b) for b in blocks), default=0)),
        }
    offdiag = q.copy()
    np.fill_diagonal(offdiag, 0.0)
    total_mass = float(np.sum(offdiag**2))
    if total_mass <= 0.0:
        within_fraction = 1.0
    else:
        within = 0.0
        for block in blocks:
            block = np.asarray(block, dtype=int)
            if block.size > 1:
                qb = offdiag[np.ix_(block, block)]
                within += float(np.sum(qb**2))
        within_fraction = within / total_mass
    sizes = [int(len(block)) for block in blocks]
    return {
        "block_within_precision_mass": float(within_fraction),
        "block_cut_ratio": float(1.0 - within_fraction),
        "n_blocks": int(len(blocks)),
        "mean_block_size": float(np.mean(sizes)) if sizes else 0.0,
        "max_block_size_actual": int(max(sizes, default=0)),
    }


def gibbs_iteration_matrix(posterior_precision: Array, blocks: Sequence[Array]) -> Array:
    q = np.asarray(posterior_precision, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("posterior_precision must be square")
    n = q.shape[0]
    validate_block_partition(blocks, n)
    if n == 0:
        return np.zeros((0, 0))
    transform = np.eye(n)
    all_positions = np.arange(n)
    for block in blocks:
        block = np.asarray(block, dtype=int)
        rest = np.setdiff1d(all_positions, block, assume_unique=True)
        if rest.size == 0:
            transform[block, :] = 0.0
            continue
        q_bb = q[np.ix_(block, block)]
        q_br = q[np.ix_(block, rest)]
        transform[block, :] = -np.linalg.solve(q_bb, q_br @ transform[rest, :])
    return transform


def gibbs_spectral_radius(posterior_precision: Array, blocks: Sequence[Array]) -> float:
    q = np.asarray(posterior_precision, dtype=float)
    if q.shape == (0, 0):
        return 0.0
    matrix = gibbs_iteration_matrix(q, blocks)
    if matrix.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))


def _initial_latent_state_for_strategy(
    dag: GaussianDAG,
    latent_indices: Array,
    posterior_mean: Array,
    strategy: str,
    random_state: int | np.random.Generator | None = None,
) -> Array | None:
    if strategy in {"posterior_mean", "oracle", "mean"}:
        return posterior_mean.copy()
    if strategy in {"none", "sampler_default"}:
        return None
    if strategy in {"zero", "zeros"}:
        return np.zeros_like(posterior_mean)
    if strategy in {"prior", "prior_imputation"}:
        prior_draw = sample_from_dag(dag, n_samples=1, random_state=random_state)[0]
        return prior_draw[latent_indices].copy()
    raise ValueError("initial_state_strategy must be one of posterior_mean, zero, prior, none")


def graph_block_gibbs(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    n_samples: int,
    burn_in: int = 0,
    thinning: int = 1,
    block_strategy: str = "depth",
    max_block_size: int = 8,
    initial_state: Sequence[float] | None = None,
    random_state: int | np.random.Generator | None = None,
    collect_rao_blackwell: bool = True,
) -> dict[str, Array | float | list[Array] | str]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning < 1:
        raise ValueError("thinning must be at least 1")
    rng = _rng(random_state)
    params = posterior_latent_parameters(dag, observed_indices, observed_values)
    latent_indices = params["latent_indices"]
    observed = np.asarray(observed_indices, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    latent_dim = latent_indices.size
    if latent_dim == 0:
        repeated = np.repeat(values[None, :], n_samples, axis=0)
        full_samples = np.zeros((n_samples, dag.dimension))
        full_samples[:, observed] = repeated
        return {
            "samples": full_samples,
            "latent_samples": np.zeros((n_samples, 0)),
            "latent_indices": latent_indices,
            "observed_indices": observed,
            "observed_values": values,
            "blocks": [],
            "block_strategy": block_strategy,
        }
    posterior_mean = params["posterior_mean"]
    posterior_precision = params["posterior_precision"]
    canonical_shift = params["canonical_shift"]
    state = _initialize_latent_state(latent_dim, posterior_mean, initial_state)
    blocks = latent_blocks_by_strategy(
        dag=dag,
        latent_indices=latent_indices,
        posterior_precision=posterior_precision,
        strategy=block_strategy,
        max_block_size=max_block_size,
        random_state=rng,
    )
    validate_block_partition(blocks, latent_dim)
    all_positions = np.arange(latent_dim)
    block_updates = []
    for block in blocks:
        block = np.asarray(block, dtype=int)
        rest = np.setdiff1d(all_positions, block, assume_unique=True)
        q_bb = posterior_precision[np.ix_(block, block)]
        q_br = posterior_precision[np.ix_(block, rest)]
        conditional_cov = np.linalg.inv(q_bb)
        conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
        conditional_chol = np.linalg.cholesky(conditional_cov)
        block_updates.append((block, rest, q_br, conditional_cov, conditional_chol))
    total_steps = burn_in + n_samples * thinning
    collected = np.empty((n_samples, latent_dim))
    rb_means = np.empty((n_samples, latent_dim)) if collect_rao_blackwell else None
    rb_second_moments = np.empty((n_samples, latent_dim)) if collect_rao_blackwell else None
    sample_idx = 0
    for step in range(total_steps):
        will_collect = step >= burn_in and (step - burn_in) % thinning == 0
        if will_collect and collect_rao_blackwell:
            rb_mean_current = np.empty(latent_dim)
            rb_second_current = np.empty(latent_dim)
        else:
            rb_mean_current = None
            rb_second_current = None
        for block, rest, q_br, conditional_cov, conditional_chol in block_updates:
            mean_rhs = canonical_shift[block].copy()
            if rest.size > 0:
                mean_rhs -= q_br @ state[rest]
            conditional_mean = conditional_cov @ mean_rhs
            if rb_mean_current is not None and rb_second_current is not None:
                rb_mean_current[block] = conditional_mean
                rb_second_current[block] = np.diag(conditional_cov) + conditional_mean**2
            state[block] = conditional_mean + conditional_chol @ rng.normal(size=block.size)
        if will_collect:
            collected[sample_idx] = state
            if collect_rao_blackwell and rb_means is not None and rb_second_moments is not None:
                rb_means[sample_idx] = rb_mean_current
                rb_second_moments[sample_idx] = rb_second_current
            sample_idx += 1
    full_samples = _embed_latent_samples(dag, latent_indices, observed, values, collected)
    result: dict[str, Array | float | list[Array] | str] = {
        "samples": full_samples,
        "latent_samples": collected,
        "latent_indices": latent_indices,
        "observed_indices": observed,
        "observed_values": values,
        "blocks": blocks,
        "block_strategy": block_strategy,
    }
    if collect_rao_blackwell and rb_means is not None and rb_second_moments is not None:
        result["rao_blackwell_latent_means"] = rb_means
        result["rao_blackwell_latent_second_moments"] = rb_second_moments
    return result


def exact_posterior_sampler(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    n_samples: int,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Array | float]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = _rng(random_state)
    params = posterior_latent_parameters(dag, observed_indices, observed_values)
    latent_indices = params["latent_indices"]
    observed = np.asarray(observed_indices, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    latent_dim = latent_indices.size
    if latent_dim == 0:
        latent_samples = np.zeros((n_samples, 0))
    else:
        cov = 0.5 * (params["posterior_covariance"] + params["posterior_covariance"].T)
        latent_samples = rng.multivariate_normal(
            mean=params["posterior_mean"],
            cov=cov,
            size=n_samples,
        )
    return {
        "samples": _embed_latent_samples(dag, latent_indices, observed, values, latent_samples),
        "latent_samples": latent_samples,
        "latent_indices": latent_indices,
        "observed_indices": observed,
        "observed_values": values,
    }


def likelihood_weighting_sis(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    n_samples: int,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Array | float]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = _rng(random_state)
    observed = np.asarray(observed_indices, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    if observed.size != values.size:
        raise ValueError("observed_indices and observed_values must have the same length")
    observed_map = {int(idx): float(value) for idx, value in zip(observed, values)}
    samples = np.zeros((n_samples, dag.dimension))
    log_weights = np.zeros(n_samples, dtype=float)
    constant = -0.5 * np.log(2.0 * np.pi)
    for node in dag.topological_order:
        parents = np.flatnonzero(dag.adjacency[:, node] != 0.0)
        parent_mean = 0.0 if parents.size == 0 else samples[:, parents] @ dag.adjacency[parents, node]
        sd = float(dag.noise_scales[node])
        if int(node) in observed_map:
            observed_value = observed_map[int(node)]
            samples[:, node] = observed_value
            standardized = (observed_value - parent_mean) / sd
            log_weights += constant - np.log(sd) - 0.5 * standardized**2
        else:
            samples[:, node] = rng.normal(loc=parent_mean, scale=sd, size=n_samples)
    params = posterior_latent_parameters(dag, observed, values)
    latent_indices = params["latent_indices"]
    normalized_weights = normalize_log_weights(log_weights)
    return {
        "samples": samples,
        "latent_samples": samples[:, latent_indices],
        "latent_indices": latent_indices,
        "observed_indices": observed,
        "observed_values": values,
        "log_weights": log_weights,
        "normalized_weights": normalized_weights,
        "weighted_ess": importance_weighted_ess(normalized_weights),
    }


def metropolis_adjusted_langevin(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    n_samples: int,
    step_size: float = 0.5,
    burn_in: int = 0,
    thinning: int = 1,
    initial_state: Sequence[float] | None = None,
    preconditioned: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Array | float]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning < 1:
        raise ValueError("thinning must be at least 1")
    rng = _rng(random_state)
    params = posterior_latent_parameters(dag, observed_indices, observed_values)
    latent_indices = params["latent_indices"]
    observed = np.asarray(observed_indices, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    latent_dim = latent_indices.size
    if latent_dim == 0:
        repeated = np.repeat(values[None, :], n_samples, axis=0)
        full_samples = np.zeros((n_samples, dag.dimension))
        full_samples[:, observed] = repeated
        return {
            "samples": full_samples,
            "latent_samples": np.zeros((n_samples, 0)),
            "latent_indices": latent_indices,
            "observed_indices": observed,
            "observed_values": values,
            "acceptance_rate": 1.0,
        }
    mu = params["posterior_mean"]
    q = params["posterior_precision"]
    state = _initialize_latent_state(latent_dim, mu, initial_state)
    if preconditioned:
        metric_diag = 1.0 / np.diag(q)
    else:
        metric_diag = np.ones(latent_dim)
    proposal_var = step_size**2 * metric_diag
    log_proposal_norm = -0.5 * float(np.sum(np.log(2.0 * np.pi * proposal_var)))

    def grad_log_target(x: Array) -> Array:
        return -(q @ (x - mu))

    def log_target(x: Array) -> float:
        centered = x - mu
        return float(-0.5 * centered @ q @ centered)

    def proposal_mean(x: Array) -> Array:
        return x + 0.5 * step_size**2 * metric_diag * grad_log_target(x)

    def log_q_density(to_state: Array, from_state: Array) -> float:
        diff = to_state - proposal_mean(from_state)
        return log_proposal_norm - 0.5 * float(np.sum((diff**2) / proposal_var))

    current_log_prob = log_target(state)
    total_steps = burn_in + n_samples * thinning
    collected = np.empty((n_samples, latent_dim))
    accepted = 0
    sample_idx = 0
    for step in range(total_steps):
        mean = proposal_mean(state)
        proposal = mean + rng.normal(scale=np.sqrt(proposal_var), size=latent_dim)
        proposal_log_prob = log_target(proposal)
        log_alpha = (
            proposal_log_prob
            + log_q_density(state, proposal)
            - current_log_prob
            - log_q_density(proposal, state)
        )
        if np.log(rng.random()) < min(0.0, log_alpha):
            state = proposal
            current_log_prob = proposal_log_prob
            accepted += 1
        if step >= burn_in and (step - burn_in) % thinning == 0:
            collected[sample_idx] = state
            sample_idx += 1
    full_samples = _embed_latent_samples(dag, latent_indices, observed, values, collected)
    return {
        "samples": full_samples,
        "latent_samples": collected,
        "latent_indices": latent_indices,
        "observed_indices": observed,
        "observed_values": values,
        "acceptance_rate": accepted / total_steps,
        "preconditioned": float(preconditioned),
    }


def _generate_dag_for_experiment(
    graph_type: str,
    dimension: int,
    sparsity: float,
    correlation: float,
    *,
    depth: int | None = None,
    max_parents: int | None = None,
    noise_scale: float | Sequence[float] = 1.0,
    noise_heterogeneity: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> GaussianDAG:
    rng = _rng(random_state)
    if noise_heterogeneity > 0.0:
        if not np.isscalar(noise_scale):
            base_noise = np.asarray(noise_scale, dtype=float)
            if base_noise.shape != (dimension,):
                raise ValueError("noise_scale sequence must match the dimension")
        else:
            base_noise = np.full(dimension, float(noise_scale))
        multiplier = rng.lognormal(
            mean=-0.5 * noise_heterogeneity**2,
            sigma=noise_heterogeneity,
            size=dimension,
        )
        noise = base_noise * multiplier
    else:
        noise = noise_scale
    weight_scale = correlation
    min_abs_weight = max(0.05, correlation / 2.0)
    if graph_type == "layered":
        if depth is None:
            raise ValueError("depth must be provided for layered graphs")
        return generate_layered_dag(
            dimension=dimension,
            depth=depth,
            sparsity=sparsity,
            weight_scale=weight_scale,
            min_abs_weight=min_abs_weight,
            noise_scale=noise,
            random_state=rng,
        )
    if graph_type in {"random", "random_sparse"}:
        return generate_random_sparse_dag(
            dimension=dimension,
            sparsity=sparsity,
            max_parents=max_parents,
            weight_scale=weight_scale,
            min_abs_weight=min_abs_weight,
            noise_scale=noise,
            random_state=rng,
        )
    if graph_type == "chain":
        return generate_chain_dag(
            dimension=dimension,
            weight_scale=weight_scale,
            min_abs_weight=min_abs_weight,
            noise_scale=noise,
            random_state=rng,
        )
    raise ValueError("graph_type must be one of: layered, random_sparse, chain")


def _evaluate_rao_blackwell_errors(result: dict, posterior_mean: Array, posterior_covariance: Array) -> dict[str, float]:
    if "rao_blackwell_latent_means" not in result:
        return {
            "rb_mean_rmse": np.nan,
            "rb_variance_rmse": np.nan,
        }
    rb_means = np.asarray(result["rao_blackwell_latent_means"], dtype=float)
    rb_seconds = np.asarray(result["rao_blackwell_latent_second_moments"], dtype=float)
    rb_mean_est = rb_means.mean(axis=0)
    rb_second_est = rb_seconds.mean(axis=0)
    rb_var_est = rb_second_est - rb_mean_est**2
    exact_var = np.diag(posterior_covariance)
    return {
        "rb_mean_rmse": float(np.sqrt(np.mean((rb_mean_est - posterior_mean) ** 2))),
        "rb_variance_rmse": float(np.sqrt(np.mean((rb_var_est - exact_var) ** 2))),
    }


def run_deepened_experiment_grid(
    dimensions: Sequence[int],
    sparsities: Sequence[float],
    correlations: Sequence[float],
    observation_strategies: Sequence[str],
    *,
    graph_types: Sequence[str] = ("layered", "random_sparse"),
    depths: Sequence[int] = (4,),
    max_parents_values: Sequence[int | None] = (4,),
    observed_fractions: Sequence[float] = (0.25,),
    n_repetitions: int = 1,
    n_samples: int = 1000,
    burn_in: int = 500,
    thinning: int = 1,
    noise_scale: float | Sequence[float] = 1.0,
    noise_heterogeneity: float = 0.0,
    depth_weight: float = 1.0,
    children_weight: float = 1.0,
    descendant_weight: float = 1.0,
    initial_state_strategy: str = "prior",
    block_max_size: int = 8,
    rw_mh_proposal_scale: float = 0.4,
    mala_step_size: float = 0.55,
    ess_max_lag: int | None = 100,
    methods: Sequence[str] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    if n_repetitions < 1:
        raise ValueError("n_repetitions must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning < 1:
        raise ValueError("thinning must be at least 1")
    if methods is None:
        methods = (
            "single_site_gibbs",
            "block_depth",
            "block_topological",
            "block_random",
            "block_precision_bfs",
            "rw_mh",
            "mala_diag",
            "likelihood_weighting_sis",
        )
    rng = _rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    for repetition in range(n_repetitions):
        for graph_type in graph_types:
            graph_depths = depths if graph_type == "layered" else (None,)
            graph_max_parents = max_parents_values if graph_type in {"random", "random_sparse"} else (None,)
            for dimension in dimensions:
                for depth in graph_depths:
                    for max_parents in graph_max_parents:
                        for sparsity in sparsities:
                            for correlation in correlations:
                                dag_seed = int(rng.integers(0, 2**32 - 1))
                                dag = _generate_dag_for_experiment(
                                    graph_type=graph_type,
                                    dimension=dimension,
                                    sparsity=sparsity,
                                    correlation=correlation,
                                    depth=depth,
                                    max_parents=max_parents,
                                    noise_scale=noise_scale,
                                    noise_heterogeneity=noise_heterogeneity,
                                    random_state=dag_seed,
                                )
                                for observed_fraction in observed_fractions:
                                    for observation_strategy in observation_strategies:
                                        conditioning_seed = int(rng.integers(0, 2**32 - 1))
                                        conditioning = select_observations(
                                            dag,
                                            observed_fraction=observed_fraction,
                                            strategy=observation_strategy,
                                            depth_weight=depth_weight,
                                            children_weight=children_weight,
                                            descendant_weight=descendant_weight,
                                            random_state=conditioning_seed,
                                        )
                                        observed_indices = conditioning["observed_indices"]
                                        observed_values = conditioning["observed_values"]
                                        posterior_params = posterior_latent_parameters(dag, observed_indices, observed_values)
                                        latent_indices = posterior_params["latent_indices"]
                                        latent_dim = int(latent_indices.size)
                                        posterior_mean = posterior_params["posterior_mean"]
                                        posterior_precision = posterior_params["posterior_precision"]
                                        posterior_covariance = posterior_params["posterior_covariance"]
                                        posterior_diag = posterior_diagnostics(posterior_precision, posterior_covariance)
                                        structure_diag = target_structure_diagnostics(dag, observed_indices)
                                        init_seed = int(rng.integers(0, 2**32 - 1))
                                        initial_state = _initial_latent_state_for_strategy(
                                            dag,
                                            latent_indices,
                                            posterior_mean,
                                            initial_state_strategy,
                                            random_state=init_seed,
                                        )
                                        block_diag_by_method: dict[str, dict[str, float | int]] = {}
                                        for method_name in methods:
                                            if method_name == "single_site_gibbs":
                                                blocks = singleton_latent_blocks(latent_indices)
                                            elif method_name == "block_depth":
                                                blocks = latent_blocks_by_strategy(
                                                    dag,
                                                    latent_indices,
                                                    posterior_precision,
                                                    strategy="depth",
                                                    max_block_size=block_max_size,
                                                )
                                            elif method_name == "block_topological":
                                                blocks = latent_blocks_by_strategy(
                                                    dag,
                                                    latent_indices,
                                                    posterior_precision,
                                                    strategy="topological_window",
                                                    max_block_size=block_max_size,
                                                )
                                            elif method_name == "block_random":
                                                blocks = latent_blocks_by_strategy(
                                                    dag,
                                                    latent_indices,
                                                    posterior_precision,
                                                    strategy="random_blocks",
                                                    max_block_size=block_max_size,
                                                    random_state=int(rng.integers(0, 2**32 - 1)),
                                                )
                                            elif method_name == "block_precision_bfs":
                                                blocks = latent_blocks_by_strategy(
                                                    dag,
                                                    latent_indices,
                                                    posterior_precision,
                                                    strategy="precision_bfs",
                                                    max_block_size=block_max_size,
                                                )
                                            else:
                                                continue
                                            spectral = 0.0 if latent_dim == 0 else gibbs_spectral_radius(posterior_precision, blocks)
                                            diag = block_precision_mass(posterior_precision, blocks)
                                            diag["gibbs_spectral_radius"] = float(spectral)
                                            block_diag_by_method[method_name] = diag
                                        for method_name in methods:
                                            sampler_seed = int(rng.integers(0, 2**32 - 1))
                                            start = perf_counter()
                                            if method_name == "likelihood_weighting_sis":
                                                result = likelihood_weighting_sis(
                                                    dag=dag,
                                                    observed_indices=observed_indices,
                                                    observed_values=observed_values,
                                                    n_samples=n_samples,
                                                    random_state=sampler_seed,
                                                )
                                            elif method_name == "single_site_gibbs":
                                                result = single_site_gibbs(
                                                    dag=dag,
                                                    observed_indices=observed_indices,
                                                    observed_values=observed_values,
                                                    n_samples=n_samples,
                                                    burn_in=burn_in,
                                                    thinning=thinning,
                                                    initial_state=initial_state,
                                                    random_state=sampler_seed,
                                                )
                                            elif method_name.startswith("block_"):
                                                strategy = {
                                                    "block_depth": "depth",
                                                    "block_topological": "topological_window",
                                                    "block_random": "random_blocks",
                                                    "block_precision_bfs": "precision_bfs",
                                                }[method_name]
                                                result = graph_block_gibbs(
                                                    dag=dag,
                                                    observed_indices=observed_indices,
                                                    observed_values=observed_values,
                                                    n_samples=n_samples,
                                                    burn_in=burn_in,
                                                    thinning=thinning,
                                                    block_strategy=strategy,
                                                    max_block_size=block_max_size,
                                                    initial_state=initial_state,
                                                    random_state=sampler_seed,
                                                    collect_rao_blackwell=True,
                                                )
                                            elif method_name == "rw_mh":
                                                result = random_walk_metropolis_hastings(
                                                    dag=dag,
                                                    observed_indices=observed_indices,
                                                    observed_values=observed_values,
                                                    n_samples=n_samples,
                                                    proposal_scale=rw_mh_proposal_scale,
                                                    burn_in=burn_in,
                                                    thinning=thinning,
                                                    initial_state=initial_state,
                                                    random_state=sampler_seed,
                                                )
                                            elif method_name == "mala_diag":
                                                result = metropolis_adjusted_langevin(
                                                    dag=dag,
                                                    observed_indices=observed_indices,
                                                    observed_values=observed_values,
                                                    n_samples=n_samples,
                                                    step_size=mala_step_size,
                                                    burn_in=burn_in,
                                                    thinning=thinning,
                                                    initial_state=initial_state,
                                                    preconditioned=True,
                                                    random_state=sampler_seed,
                                                )
                                            else:
                                                raise ValueError(f"unsupported method: {method_name}")
                                            runtime = perf_counter() - start
                                            latent_samples = result["latent_samples"]
                                            if latent_dim == 0:
                                                ess_values = np.array([n_samples], dtype=float)
                                                errors = {"mean_rmse": 0.0, "variance_rmse": 0.0, "covariance_frobenius_error": 0.0}
                                                weighted_ess_value = np.nan
                                            elif method_name == "likelihood_weighting_sis":
                                                normalized_weights = result["normalized_weights"]
                                                weighted_ess_value = float(result["weighted_ess"])
                                                ess_values = np.array([weighted_ess_value], dtype=float)
                                                errors = weighted_estimation_errors(
                                                    latent_samples,
                                                    normalized_weights,
                                                    posterior_mean,
                                                    posterior_covariance,
                                                )
                                            else:
                                                ess_values = effective_sample_size(latent_samples, max_lag=ess_max_lag)
                                                weighted_ess_value = np.nan
                                                errors = estimation_errors(latent_samples, posterior_mean, posterior_covariance)
                                            row: dict[str, float | int | str] = {
                                                "graph_type": "random_sparse" if graph_type == "random" else graph_type,
                                                "method": method_name,
                                                "repetition": int(repetition),
                                                "dimension": int(dimension),
                                                "depth": int(depth) if depth is not None else np.nan,
                                                "max_parents": int(max_parents) if max_parents is not None else np.nan,
                                                "sparsity": float(sparsity),
                                                "correlation": float(correlation),
                                                "observed_fraction": float(observed_fraction),
                                                "observation_strategy": str(observation_strategy),
                                                "initial_state_strategy": initial_state_strategy,
                                                "n_observed": int(observed_indices.size),
                                                "n_latent": latent_dim,
                                                "runtime_seconds": float(runtime),
                                                "ess_mean": float(np.mean(ess_values)),
                                                "ess_min": float(np.min(ess_values)),
                                                "ess_max": float(np.max(ess_values)),
                                                "ess_per_second": float(np.mean(ess_values) / runtime) if runtime > 0.0 else np.nan,
                                                "weighted_ess": weighted_ess_value,
                                                "weighted_ess_fraction": float(weighted_ess_value / n_samples) if np.isfinite(weighted_ess_value) else np.nan,
                                                "mean_rmse": errors["mean_rmse"],
                                                "variance_rmse": errors["variance_rmse"],
                                                "covariance_frobenius_error": errors["covariance_frobenius_error"],
                                                "acceptance_rate": float(result.get("acceptance_rate", np.nan)),
                                            }
                                            row.update(posterior_diag)
                                            row.update(structure_diag)
                                            if method_name in block_diag_by_method:
                                                row.update(block_diag_by_method[method_name])
                                            else:
                                                row.update(
                                                    {
                                                        "block_within_precision_mass": np.nan,
                                                        "block_cut_ratio": np.nan,
                                                        "n_blocks": np.nan,
                                                        "mean_block_size": np.nan,
                                                        "max_block_size_actual": np.nan,
                                                        "gibbs_spectral_radius": np.nan,
                                                    }
                                                )
                                            row.update(_evaluate_rao_blackwell_errors(result, posterior_mean, posterior_covariance))
                                            rows.append(row)
    return pd.DataFrame(rows)


def run_block_size_sweep(
    *,
    graph_type: str = "random_sparse",
    dimension: int = 60,
    depth: int = 4,
    sparsity: float = 0.18,
    correlation: float = 0.8,
    max_parents: int | None = 6,
    observed_fraction: float = 0.4,
    observation_strategies: Sequence[str] = ("uniform", "deep"),
    block_sizes: Sequence[int] = (2, 4, 8, 12, 16),
    methods: Sequence[str] = ("block_topological", "block_random", "block_precision_bfs"),
    n_repetitions: int = 3,
    n_samples: int = 800,
    burn_in: int = 300,
    thinning: int = 1,
    ess_max_lag: int | None = 100,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = _rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    for repetition in range(n_repetitions):
        dag_seed = int(rng.integers(0, 2**32 - 1))
        dag = _generate_dag_for_experiment(
            graph_type=graph_type,
            dimension=dimension,
            depth=depth if graph_type == "layered" else None,
            sparsity=sparsity,
            correlation=correlation,
            max_parents=max_parents,
            random_state=dag_seed,
        )
        for observation_strategy in observation_strategies:
            cond_seed = int(rng.integers(0, 2**32 - 1))
            conditioning = select_observations(
                dag,
                observed_fraction=observed_fraction,
                strategy=observation_strategy,
                random_state=cond_seed,
            )
            observed_indices = conditioning["observed_indices"]
            observed_values = conditioning["observed_values"]
            params = posterior_latent_parameters(dag, observed_indices, observed_values)
            latent_indices = params["latent_indices"]
            posterior_mean = params["posterior_mean"]
            posterior_precision = params["posterior_precision"]
            posterior_covariance = params["posterior_covariance"]
            diagnostics = posterior_diagnostics(posterior_precision, posterior_covariance)
            initial_state = _initial_latent_state_for_strategy(
                dag,
                latent_indices,
                posterior_mean,
                "prior",
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            for block_size in block_sizes:
                for method in methods:
                    strategy = {
                        "block_topological": "topological_window",
                        "block_random": "random_blocks",
                        "block_precision_bfs": "precision_bfs",
                    }[method]
                    seed = int(rng.integers(0, 2**32 - 1))
                    start = perf_counter()
                    result = graph_block_gibbs(
                        dag=dag,
                        observed_indices=observed_indices,
                        observed_values=observed_values,
                        n_samples=n_samples,
                        burn_in=burn_in,
                        thinning=thinning,
                        block_strategy=strategy,
                        max_block_size=block_size,
                        initial_state=initial_state,
                        random_state=seed,
                        collect_rao_blackwell=True,
                    )
                    runtime = perf_counter() - start
                    ess = effective_sample_size(result["latent_samples"], max_lag=ess_max_lag)
                    errors = estimation_errors(result["latent_samples"], posterior_mean, posterior_covariance)
                    blocks = result["blocks"]
                    block_diag = block_precision_mass(posterior_precision, blocks)
                    block_diag["gibbs_spectral_radius"] = gibbs_spectral_radius(posterior_precision, blocks) if posterior_precision.size else 0.0
                    row: dict[str, float | int | str] = {
                        "graph_type": graph_type,
                        "method": method,
                        "repetition": repetition,
                        "dimension": dimension,
                        "depth": depth if graph_type == "layered" else np.nan,
                        "max_parents": max_parents if graph_type != "layered" else np.nan,
                        "sparsity": sparsity,
                        "correlation": correlation,
                        "observed_fraction": observed_fraction,
                        "observation_strategy": observation_strategy,
                        "block_size": int(block_size),
                        "runtime_seconds": float(runtime),
                        "ess_mean": float(np.mean(ess)),
                        "ess_per_second": float(np.mean(ess) / runtime) if runtime > 0.0 else np.nan,
                        "mean_rmse": errors["mean_rmse"],
                        "variance_rmse": errors["variance_rmse"],
                        "covariance_frobenius_error": errors["covariance_frobenius_error"],
                    }
                    row.update(diagnostics)
                    row.update(block_diag)
                    row.update(_evaluate_rao_blackwell_errors(result, posterior_mean, posterior_covariance))
                    rows.append(row)
    return pd.DataFrame(rows)
