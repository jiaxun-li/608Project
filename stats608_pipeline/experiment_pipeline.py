"""Experiment utilities for sparse linear Gaussian DAGs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class GaussianDAG:
    """Sparse linear Gaussian DAG with exact Gaussian posterior utilities."""

    adjacency: Array
    noise_scales: Array
    topological_order: Array
    layers: list[Array]

    def __post_init__(self) -> None:
        adjacency = np.asarray(self.adjacency, dtype=float)
        noise_scales = np.asarray(self.noise_scales, dtype=float)
        order = np.asarray(self.topological_order, dtype=int)
        layers = [np.asarray(layer, dtype=int) for layer in self.layers]

        n = adjacency.shape[0]
        if adjacency.shape != (n, n):
            raise ValueError("adjacency must be square")
        if noise_scales.shape != (n,):
            raise ValueError("noise_scales must have shape (dimension,)")
        if order.shape != (n,):
            raise ValueError("topological_order must contain one entry per node")
        if np.any(np.diag(adjacency) != 0.0):
            raise ValueError("adjacency must have zero diagonal")
        if np.any(noise_scales <= 0.0):
            raise ValueError("noise scales must be positive")

        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "noise_scales", noise_scales)
        object.__setattr__(self, "topological_order", order)
        object.__setattr__(self, "layers", layers)

    @property
    def dimension(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def precision(self) -> Array:
        identity = np.eye(self.dimension)
        innovation_matrix = identity - self.adjacency.T
        noise_precision = np.diag(1.0 / np.square(self.noise_scales))
        return innovation_matrix.T @ noise_precision @ innovation_matrix

    @property
    def covariance(self) -> Array:
        return np.linalg.inv(self.precision)

    @property
    def mean(self) -> Array:
        return np.zeros(self.dimension)

    def posterior(
        self,
        observed_indices: Sequence[int],
        observed_values: Sequence[float],
    ) -> tuple[Array, Array]:
        """Return exact posterior mean/covariance of unobserved nodes."""
        observed = np.asarray(observed_indices, dtype=int)
        values = np.asarray(observed_values, dtype=float)
        if observed.size != values.size:
            raise ValueError("observed_indices and observed_values must have the same length")
        if observed.size == 0:
            return self.mean.copy(), self.covariance.copy()

        all_indices = np.arange(self.dimension)
        latent = np.setdiff1d(all_indices, observed, assume_unique=False)
        sigma = self.covariance

        sigma_uu = sigma[np.ix_(latent, latent)]
        sigma_uo = sigma[np.ix_(latent, observed)]
        sigma_oo = sigma[np.ix_(observed, observed)]
        solved = np.linalg.solve(sigma_oo, values)
        posterior_mean = sigma_uo @ solved
        posterior_covariance = sigma_uu - sigma_uo @ np.linalg.solve(sigma_oo, sigma_uo.T)

        full_mean = np.zeros(self.dimension)
        full_covariance = np.zeros((self.dimension, self.dimension))
        full_mean[observed] = values
        full_mean[latent] = posterior_mean
        full_covariance[np.ix_(latent, latent)] = posterior_covariance
        return full_mean, full_covariance


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _normalize_layers(layers: Iterable[Iterable[int]]) -> list[Array]:
    return [np.asarray(list(layer), dtype=int) for layer in layers]


def _default_noise_scales(dimension: int, noise_scale: float | Sequence[float]) -> Array:
    if np.isscalar(noise_scale):
        return np.full(dimension, float(noise_scale))
    noise_scales = np.asarray(noise_scale, dtype=float)
    if noise_scales.shape != (dimension,):
        raise ValueError("noise_scale sequence must match the dimension")
    return noise_scales


def _draw_weight(
    rng: np.random.Generator,
    weight_scale: float,
    min_abs_weight: float,
) -> float:
    sign = rng.choice(np.array([-1.0, 1.0]))
    magnitude = rng.uniform(min_abs_weight, weight_scale)
    return float(sign * magnitude)


def generate_chain_dag(
    dimension: int,
    weight_scale: float = 0.8,
    min_abs_weight: float = 0.2,
    noise_scale: float | Sequence[float] = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> GaussianDAG:
    """Generate a chain DAG: 0 -> 1 -> ... -> d-1."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    rng = _rng(random_state)
    adjacency = np.zeros((dimension, dimension))
    for parent in range(dimension - 1):
        adjacency[parent, parent + 1] = _draw_weight(rng, weight_scale, min_abs_weight)
    layers = [np.array([idx], dtype=int) for idx in range(dimension)]
    return GaussianDAG(
        adjacency=adjacency,
        noise_scales=_default_noise_scales(dimension, noise_scale),
        topological_order=np.arange(dimension),
        layers=layers,
    )


def generate_layered_dag(
    dimension: int,
    depth: int,
    sparsity: float = 0.5,
    weight_scale: float = 0.8,
    min_abs_weight: float = 0.2,
    noise_scale: float | Sequence[float] = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> GaussianDAG:
    """Generate a layered DAG with edges only between consecutive layers."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if depth < 1 or depth > dimension:
        raise ValueError("depth must be between 1 and dimension")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")

    rng = _rng(random_state)
    layer_sizes = np.full(depth, dimension // depth, dtype=int)
    layer_sizes[: dimension % depth] += 1

    layers: list[Array] = []
    start = 0
    for size in layer_sizes:
        stop = start + int(size)
        layers.append(np.arange(start, stop, dtype=int))
        start = stop

    adjacency = np.zeros((dimension, dimension))
    for left, right in zip(layers[:-1], layers[1:]):
        for parent in left:
            for child in right:
                if rng.random() <= sparsity:
                    adjacency[parent, child] = _draw_weight(rng, weight_scale, min_abs_weight)

    return GaussianDAG(
        adjacency=adjacency,
        noise_scales=_default_noise_scales(dimension, noise_scale),
        topological_order=np.arange(dimension),
        layers=layers,
    )


def generate_random_sparse_dag(
    dimension: int,
    sparsity: float = 0.15,
    max_parents: int | None = None,
    weight_scale: float = 0.8,
    min_abs_weight: float = 0.2,
    noise_scale: float | Sequence[float] = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> GaussianDAG:
    """Generate a random sparse DAG under a random topological order."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    if max_parents is not None and max_parents < 0:
        raise ValueError("max_parents must be non-negative")

    rng = _rng(random_state)
    order = rng.permutation(dimension)
    rank = np.empty(dimension, dtype=int)
    rank[order] = np.arange(dimension)

    adjacency = np.zeros((dimension, dimension))
    for child in range(dimension):
        eligible_parents = np.flatnonzero(rank < rank[child])
        if eligible_parents.size == 0:
            continue

        selected = eligible_parents[rng.random(eligible_parents.size) <= sparsity]
        if max_parents is not None and selected.size > max_parents:
            selected = rng.choice(selected, size=max_parents, replace=False)
        for parent in selected:
            adjacency[parent, child] = _draw_weight(rng, weight_scale, min_abs_weight)

    layers = _layers_from_adjacency(adjacency, order)
    return GaussianDAG(
        adjacency=adjacency,
        noise_scales=_default_noise_scales(dimension, noise_scale),
        topological_order=order,
        layers=layers,
    )


def _layers_from_adjacency(adjacency: Array, order: Array) -> list[Array]:
    depth = np.zeros(adjacency.shape[0], dtype=int)
    for node in order:
        parents = np.flatnonzero(adjacency[:, node] != 0.0)
        if parents.size:
            depth[node] = 1 + int(np.max(depth[parents]))
    layers = [np.flatnonzero(depth == level) for level in range(int(depth.max()) + 1)]
    return _normalize_layers(layers)


def sample_from_dag(
    dag: GaussianDAG,
    n_samples: int = 1,
    random_state: int | np.random.Generator | None = None,
) -> Array:
    """Ancestral sampling from the DAG prior."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = _rng(random_state)
    samples = np.zeros((n_samples, dag.dimension))
    for node in dag.topological_order:
        parents = np.flatnonzero(dag.adjacency[:, node] != 0.0)
        parent_contrib = 0.0 if parents.size == 0 else samples[:, parents] @ dag.adjacency[parents, node]
        noise = rng.normal(loc=0.0, scale=dag.noise_scales[node], size=n_samples)
        samples[:, node] = parent_contrib + noise
    return samples


def node_structure_stats(dag: GaussianDAG) -> dict[str, Array]:
    """Return depth, children count, and descendant count for each node."""
    children = np.count_nonzero(dag.adjacency, axis=1).astype(int)

    depth = np.zeros(dag.dimension, dtype=int)
    for node in dag.topological_order:
        parents = np.flatnonzero(dag.adjacency[:, node] != 0.0)
        if parents.size:
            depth[node] = 1 + int(np.max(depth[parents]))

    descendants: list[set[int]] = [set() for _ in range(dag.dimension)]
    for node in dag.topological_order[::-1]:
        direct_children = np.flatnonzero(dag.adjacency[node, :] != 0.0)
        reachable: set[int] = set()
        for child in direct_children:
            reachable.add(int(child))
            reachable.update(descendants[int(child)])
        descendants[int(node)] = reachable

    descendant_count = np.array([len(nodes) for nodes in descendants], dtype=int)
    return {
        "depth": depth,
        "children_count": children,
        "descendant_count": descendant_count,
    }


def observation_scores(
    dag: GaussianDAG,
    strategy: str = "uniform",
    depth_weight: float = 1.0,
    children_weight: float = 1.0,
    descendant_weight: float = 1.0,
) -> Array:
    """Return unnormalized structural scores for observation selection."""
    stats = node_structure_stats(dag)
    if strategy == "uniform":
        return np.ones(dag.dimension, dtype=float)

    if strategy not in {"shallow", "deep"}:
        raise ValueError("strategy must be one of: 'uniform', 'shallow', 'deep'")
    if depth_weight < 0.0 or children_weight < 0.0 or descendant_weight < 0.0:
        raise ValueError("depth_weight, children_weight, and descendant_weight must be non-negative")

    depth = stats["depth"].astype(float)
    children = stats["children_count"].astype(float)
    descendants = stats["descendant_count"].astype(float)
    max_depth = float(depth.max()) if depth.size else 0.0

    if strategy == "shallow":
        depth_component = max_depth - depth + 1.0
        return 1.0 + depth_weight * depth_component + children_weight * children + descendant_weight * descendants

    depth_component = depth + 1.0
    return 1.0 + depth_weight * depth_component + children_weight * (children.max(initial=0.0) - children) + descendant_weight * (descendants.max(initial=0.0) - descendants)


def observed_mask_from_fraction(
    dag: GaussianDAG,
    dimension: int,
    observed_fraction: float,
    strategy: str = "uniform",
    depth_weight: float = 1.0,
    children_weight: float = 1.0,
    descendant_weight: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> Array:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not 0.0 <= observed_fraction <= 1.0:
        raise ValueError("observed_fraction must be in [0, 1]")
    rng = _rng(random_state)
    observed_count = int(round(observed_fraction * dimension))
    observed_indices = sample_observed_indices(
        dag=dag,
        observed_count=observed_count,
        strategy=strategy,
        depth_weight=depth_weight,
        children_weight=children_weight,
        descendant_weight=descendant_weight,
        random_state=rng,
    )
    mask = np.zeros(dimension, dtype=bool)
    mask[observed_indices] = True
    return mask


def sample_observed_indices(
    dag: GaussianDAG,
    observed_count: int,
    strategy: str = "uniform",
    depth_weight: float = 1.0,
    children_weight: float = 1.0,
    descendant_weight: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> Array:
    """Sample observed nodes without replacement under a chosen weighting scheme."""
    if observed_count < 0 or observed_count > dag.dimension:
        raise ValueError("observed_count must be between 0 and the dimension")
    rng = _rng(random_state)
    if observed_count == 0:
        return np.array([], dtype=int)

    probabilities = observation_probabilities(
        dag=dag,
        strategy=strategy,
        depth_weight=depth_weight,
        children_weight=children_weight,
        descendant_weight=descendant_weight,
    )
    return np.sort(rng.choice(dag.dimension, size=observed_count, replace=False, p=probabilities))


def observation_probabilities(
    dag: GaussianDAG,
    strategy: str = "uniform",
    depth_weight: float = 1.0,
    children_weight: float = 1.0,
    descendant_weight: float = 1.0,
) -> Array:
    """Return normalized node weights for observation selection."""
    scores = observation_scores(
        dag=dag,
        strategy=strategy,
        depth_weight=depth_weight,
        children_weight=children_weight,
        descendant_weight=descendant_weight,
    )
    return scores / scores.sum()


def select_observations(
    dag: GaussianDAG,
    observed_fraction: float,
    strategy: str = "uniform",
    depth_weight: float = 1.0,
    children_weight: float = 1.0,
    descendant_weight: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Array | str]:
    """Sample one latent state and return exact conditioning inputs/outputs."""
    rng = _rng(random_state)
    latent_sample = sample_from_dag(dag, n_samples=1, random_state=rng)[0]
    observed_mask = observed_mask_from_fraction(
        dag=dag,
        dimension=dag.dimension,
        observed_fraction=observed_fraction,
        strategy=strategy,
        depth_weight=depth_weight,
        children_weight=children_weight,
        descendant_weight=descendant_weight,
        random_state=rng,
    )
    observed_indices = np.flatnonzero(observed_mask)
    observed_values = latent_sample[observed_indices]
    posterior_mean, posterior_covariance = dag.posterior(observed_indices, observed_values)
    return {
        "full_sample": latent_sample,
        "observed_mask": observed_mask,
        "observed_indices": observed_indices,
        "observed_values": observed_values,
        "observation_strategy": strategy,
        "posterior_mean": posterior_mean,
        "posterior_covariance": posterior_covariance,
    }


def posterior_latent_parameters(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
) -> dict[str, Array]:
    """Return exact latent-space Gaussian posterior parameters."""
    observed = np.asarray(observed_indices, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    if observed.size != values.size:
        raise ValueError("observed_indices and observed_values must have the same length")

    all_indices = np.arange(dag.dimension)
    latent = np.setdiff1d(all_indices, observed, assume_unique=False)
    precision = dag.precision

    if latent.size == 0:
        return {
            "latent_indices": latent,
            "observed_indices": observed,
            "posterior_mean": np.array([], dtype=float),
            "posterior_precision": np.zeros((0, 0)),
            "posterior_covariance": np.zeros((0, 0)),
            "canonical_shift": np.array([], dtype=float),
        }

    q_ll = precision[np.ix_(latent, latent)]
    q_lo = precision[np.ix_(latent, observed)]
    shift = -q_lo @ values
    mean = np.linalg.solve(q_ll, shift)
    covariance = np.linalg.inv(q_ll)
    return {
        "latent_indices": latent,
        "observed_indices": observed,
        "posterior_mean": mean,
        "posterior_precision": q_ll,
        "posterior_covariance": covariance,
        "canonical_shift": shift,
    }


def _initialize_latent_state(
    latent_dim: int,
    posterior_mean: Array,
    initial_state: Sequence[float] | None,
) -> Array:
    if initial_state is None:
        return posterior_mean.copy()
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (latent_dim,):
        raise ValueError("initial_state must match the latent dimension")
    return state.copy()


def _embed_latent_samples(
    dag: GaussianDAG,
    latent_indices: Array,
    observed_indices: Array,
    observed_values: Array,
    latent_samples: Array,
) -> Array:
    full_samples = np.repeat(observed_values[None, :], latent_samples.shape[0], axis=0)
    embedded = np.zeros((latent_samples.shape[0], dag.dimension))
    embedded[:, observed_indices] = full_samples
    embedded[:, latent_indices] = latent_samples
    return embedded


def latent_layer_blocks(dag: GaussianDAG, latent_indices: Sequence[int]) -> list[Array]:
    """Map DAG layer blocks to latent-coordinate positions after conditioning."""
    latent = np.asarray(latent_indices, dtype=int)
    latent_pos = {int(node): pos for pos, node in enumerate(latent)}
    blocks: list[Array] = []
    for layer in dag.layers:
        block = np.array([latent_pos[int(node)] for node in layer if int(node) in latent_pos], dtype=int)
        if block.size > 0:
            blocks.append(block)
    return blocks


def single_site_gibbs(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    n_samples: int,
    burn_in: int = 0,
    thinning: int = 1,
    initial_state: Sequence[float] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Array | float]:
    """Single-site Gibbs sampler for the exact Gaussian latent posterior."""
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
        }

    posterior_mean = params["posterior_mean"]
    posterior_precision = params["posterior_precision"]
    canonical_shift = params["canonical_shift"]
    state = _initialize_latent_state(latent_dim, posterior_mean, initial_state)

    total_steps = burn_in + n_samples * thinning
    collected = np.empty((n_samples, latent_dim))
    sample_idx = 0
    for step in range(total_steps):
        for idx in range(latent_dim):
            conditional_mean = (
                canonical_shift[idx]
                - (posterior_precision[idx, :] @ state - posterior_precision[idx, idx] * state[idx])
            ) / posterior_precision[idx, idx]
            conditional_sd = np.sqrt(1.0 / posterior_precision[idx, idx])
            state[idx] = rng.normal(loc=conditional_mean, scale=conditional_sd)

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
    }


def block_gibbs(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    n_samples: int,
    burn_in: int = 0,
    thinning: int = 1,
    initial_state: Sequence[float] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Array | float | list[Array]]:
    """Block Gibbs sampler using layer-based latent blocks."""
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
        }

    posterior_mean = params["posterior_mean"]
    posterior_precision = params["posterior_precision"]
    canonical_shift = params["canonical_shift"]
    state = _initialize_latent_state(latent_dim, posterior_mean, initial_state)
    blocks = latent_layer_blocks(dag, latent_indices)

    total_steps = burn_in + n_samples * thinning
    collected = np.empty((n_samples, latent_dim))
    sample_idx = 0
    all_positions = np.arange(latent_dim)

    for step in range(total_steps):
        for block in blocks:
            rest = np.setdiff1d(all_positions, block, assume_unique=True)
            q_bb = posterior_precision[np.ix_(block, block)]
            mean_rhs = canonical_shift[block]
            if rest.size > 0:
                mean_rhs = mean_rhs - posterior_precision[np.ix_(block, rest)] @ state[rest]
            conditional_mean = np.linalg.solve(q_bb, mean_rhs)
            conditional_cov = np.linalg.inv(q_bb)
            state[block] = rng.multivariate_normal(mean=conditional_mean, cov=conditional_cov)

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
        "blocks": blocks,
    }


def random_walk_metropolis_hastings(
    dag: GaussianDAG,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    n_samples: int,
    proposal_scale: float = 1.0,
    burn_in: int = 0,
    thinning: int = 1,
    initial_state: Sequence[float] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Array | float]:
    """Random-walk Metropolis-Hastings for the exact Gaussian latent posterior."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if proposal_scale <= 0.0:
        raise ValueError("proposal_scale must be positive")
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

    posterior_mean = params["posterior_mean"]
    posterior_precision = params["posterior_precision"]
    state = _initialize_latent_state(latent_dim, posterior_mean, initial_state)

    def log_target(x: Array) -> float:
        centered = x - posterior_mean
        return float(-0.5 * centered @ posterior_precision @ centered)

    current_log_prob = log_target(state)
    total_steps = burn_in + n_samples * thinning
    collected = np.empty((n_samples, latent_dim))
    accepted = 0
    sample_idx = 0

    for step in range(total_steps):
        proposal = state + proposal_scale * rng.normal(size=latent_dim)
        proposal_log_prob = log_target(proposal)
        log_alpha = proposal_log_prob - current_log_prob
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
    }


def run_layered_experiment_grid(
    dimensions: Sequence[int],
    sparsities: Sequence[float],
    correlations: Sequence[float],
    observation_strategies: Sequence[str],
    *,
    depths: Sequence[int] | None = None,
    observed_fraction: float = 0.25,
    n_repetitions: int = 1,
    n_samples: int = 2000,
    burn_in: int = 500,
    thinning: int = 1,
    noise_scale: float | Sequence[float] = 1.0,
    depth_weight: float = 1.0,
    children_weight: float = 1.0,
    descendant_weight: float = 1.0,
    rw_mh_proposal_scale: float = 0.6,
    random_state: int | np.random.Generator | None = None,
):
    """Run a layered-DAG experiment grid and return results as a dataframe."""
    import pandas as pd

    if n_repetitions < 1:
        raise ValueError("n_repetitions must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning < 1:
        raise ValueError("thinning must be at least 1")

    rng = _rng(random_state)
    depth_list = [3] if depths is None else list(depths)
    rows: list[dict[str, float | int | str]] = []

    sampler_map = {
        "single_site_gibbs": single_site_gibbs,
        "block_gibbs": block_gibbs,
        "rw_mh": random_walk_metropolis_hastings,
    }

    for repetition in range(n_repetitions):
        for dimension in dimensions:
            for depth in depth_list:
                for sparsity in sparsities:
                    for correlation in correlations:
                        dag_seed = int(rng.integers(0, 2**32 - 1))
                        dag = generate_layered_dag(
                            dimension=dimension,
                            depth=depth,
                            sparsity=sparsity,
                            weight_scale=correlation,
                            min_abs_weight=max(0.05, correlation / 2.0),
                            noise_scale=noise_scale,
                            random_state=dag_seed,
                        )

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
                            posterior_params = posterior_latent_parameters(
                                dag,
                                conditioning["observed_indices"],
                                conditioning["observed_values"],
                            )

                            for method_name, sampler in sampler_map.items():
                                sampler_seed = int(rng.integers(0, 2**32 - 1))
                                sampler_kwargs = {
                                    "dag": dag,
                                    "observed_indices": conditioning["observed_indices"],
                                    "observed_values": conditioning["observed_values"],
                                    "n_samples": n_samples,
                                    "burn_in": burn_in,
                                    "thinning": thinning,
                                    "random_state": sampler_seed,
                                }
                                if method_name == "rw_mh":
                                    sampler_kwargs["proposal_scale"] = rw_mh_proposal_scale

                                start = perf_counter()
                                result = sampler(**sampler_kwargs)
                                runtime = perf_counter() - start

                                latent_samples = result["latent_samples"]
                                latent_ess = effective_sample_size(latent_samples) if latent_samples.shape[1] > 0 else np.array([n_samples], dtype=float)
                                latent_errors = estimation_errors(
                                    latent_samples,
                                    posterior_params["posterior_mean"],
                                    posterior_params["posterior_covariance"],
                                ) if latent_samples.shape[1] > 0 else {
                                    "mean_rmse": 0.0,
                                    "variance_rmse": 0.0,
                                    "covariance_frobenius_error": 0.0,
                                }

                                row: dict[str, float | int | str] = {
                                    "graph_type": "layered",
                                    "method": method_name,
                                    "repetition": repetition,
                                    "dimension": dimension,
                                    "depth": depth,
                                    "sparsity": sparsity,
                                    "correlation": correlation,
                                    "observed_fraction": observed_fraction,
                                    "observation_strategy": observation_strategy,
                                    "n_observed": int(conditioning["observed_indices"].size),
                                    "n_latent": int(posterior_params["latent_indices"].size),
                                    "runtime_seconds": runtime,
                                    "ess_mean": float(np.mean(latent_ess)),
                                    "ess_min": float(np.min(latent_ess)),
                                    "ess_max": float(np.max(latent_ess)),
                                    "mean_rmse": latent_errors["mean_rmse"],
                                    "variance_rmse": latent_errors["variance_rmse"],
                                    "covariance_frobenius_error": latent_errors["covariance_frobenius_error"],
                                }
                                if method_name == "rw_mh":
                                    row["acceptance_rate"] = float(result["acceptance_rate"])
                                else:
                                    row["acceptance_rate"] = np.nan
                                rows.append(row)

    return pd.DataFrame(rows)


def summarize_experiment_results(
    results_df,
    metric: str,
    x: str = "correlation",
    facet: str = "observation_strategy",
):
    """Aggregate a metric by method, x-axis column, and optional facet."""
    import pandas as pd

    required = {"method", metric, x}
    if facet is not None:
        required.add(facet)
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    group_cols = ["method", x]
    if facet is not None:
        group_cols.insert(0, facet)

    summary = (
        results_df.groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n_runs"})
    )
    summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    return pd.DataFrame(summary)


def plot_experiment_metric(
    results_df,
    metric: str,
    x: str = "correlation",
    facet: str = "observation_strategy",
    *,
    title: str | None = None,
    y_label: str | None = None,
    log_y: bool = False,
    show_error_bars: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Plot an aggregated experiment metric by method."""
    import matplotlib.pyplot as plt

    summary = summarize_experiment_results(results_df, metric=metric, x=x, facet=facet)
    facet_values = [None] if facet is None else list(summary[facet].drop_duplicates())
    n_panels = len(facet_values)
    if figsize is None:
        figsize = (6 * n_panels, 4)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    method_order = list(summary["method"].drop_duplicates())
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for panel_idx, facet_value in enumerate(facet_values):
        ax = axes[0, panel_idx]
        panel_df = summary if facet is None else summary[summary[facet] == facet_value]
        for method_idx, method in enumerate(method_order):
            method_df = panel_df[panel_df["method"] == method].sort_values(by=x)
            if method_df.empty:
                continue
            color = colors[method_idx % len(colors)] if colors else None
            x_values = method_df[x].to_numpy()
            y_values = method_df[f"{metric}_mean"].to_numpy()
            ax.plot(x_values, y_values, marker="o", label=method, color=color)
            if show_error_bars:
                yerr = method_df[f"{metric}_std"].to_numpy()
                ax.errorbar(x_values, y_values, yerr=yerr, fmt="none", capsize=3, color=color)

        ax.set_xlabel(x)
        ax.set_ylabel(y_label or metric)
        if facet is not None:
            ax.set_title(f"{facet}={facet_value}")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return fig, axes


def plot_ess(results_df, x: str = "correlation", facet: str = "observation_strategy", **kwargs):
    """Convenience wrapper for plotting ESS."""
    return plot_experiment_metric(
        results_df,
        metric="ess_mean",
        x=x,
        facet=facet,
        title=kwargs.pop("title", "Effective Sample Size"),
        y_label=kwargs.pop("y_label", "ESS"),
        **kwargs,
    )


def plot_runtime(results_df, x: str = "correlation", facet: str = "observation_strategy", **kwargs):
    """Convenience wrapper for plotting runtime."""
    return plot_experiment_metric(
        results_df,
        metric="runtime_seconds",
        x=x,
        facet=facet,
        title=kwargs.pop("title", "Runtime"),
        y_label=kwargs.pop("y_label", "Seconds"),
        log_y=kwargs.pop("log_y", True),
        **kwargs,
    )


def plot_error(
    results_df,
    metric: str = "mean_rmse",
    x: str = "correlation",
    facet: str = "observation_strategy",
    **kwargs,
):
    """Convenience wrapper for plotting an error metric."""
    return plot_experiment_metric(
        results_df,
        metric=metric,
        x=x,
        facet=facet,
        title=kwargs.pop("title", f"{metric}"),
        y_label=kwargs.pop("y_label", metric),
        log_y=kwargs.pop("log_y", True),
        **kwargs,
    )


def block_sizes_from_layers(dag: GaussianDAG) -> list[int]:
    return [int(layer.size) for layer in dag.layers]


def make_block_indices(dag: GaussianDAG, strategy: str = "layers") -> list[Array]:
    if strategy != "layers":
        raise ValueError(f"Unsupported block strategy: {strategy}")
    return [layer.copy() for layer in dag.layers]


def autocorrelation(samples: Array, max_lag: int = 50) -> Array:
    """Per-dimension autocorrelation up to max_lag."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, dimension)")
    n_samples, _ = samples.shape
    max_lag = min(max_lag, n_samples - 1)
    centered = samples - samples.mean(axis=0, keepdims=True)
    denom = np.sum(centered**2, axis=0)
    acf = np.empty((max_lag + 1, samples.shape[1]))
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        numer = np.sum(centered[:-lag] * centered[lag:], axis=0)
        acf[lag] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)
    return acf


def effective_sample_size(samples: Array, max_lag: int | None = None) -> Array:
    """Initial positive sequence ESS estimate per dimension."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, dimension)")
    n_samples = samples.shape[0]
    if max_lag is None:
        max_lag = min(n_samples - 1, max(10, n_samples // 2))
    rho = autocorrelation(samples, max_lag=max_lag)
    ess = np.empty(samples.shape[1])
    for dim in range(samples.shape[1]):
        pair_sums = []
        for lag in range(1, rho.shape[0] - 1, 2):
            pair_sum = rho[lag, dim] + rho[lag + 1, dim]
            if pair_sum <= 0.0:
                break
            pair_sums.append(pair_sum)
        tau = 1.0 + 2.0 * np.sum(pair_sums)
        ess[dim] = n_samples / tau
    return ess


def estimation_errors(
    samples: Array,
    exact_mean: Sequence[float],
    exact_covariance: Array,
) -> dict[str, float]:
    """Compare Monte Carlo estimates with the exact Gaussian posterior."""
    samples = np.asarray(samples, dtype=float)
    exact_mean = np.asarray(exact_mean, dtype=float)
    exact_covariance = np.asarray(exact_covariance, dtype=float)
    sample_mean = samples.mean(axis=0)
    sample_covariance = np.cov(samples, rowvar=False)
    diag_exact = np.diag(exact_covariance)
    diag_est = np.diag(sample_covariance)
    return {
        "mean_rmse": float(np.sqrt(np.mean((sample_mean - exact_mean) ** 2))),
        "variance_rmse": float(np.sqrt(np.mean((diag_est - diag_exact) ** 2))),
        "covariance_frobenius_error": float(np.linalg.norm(sample_covariance - exact_covariance, ord="fro")),
    }


def runtime_seconds(func: Callable[..., Array], /, *args, **kwargs) -> tuple[Array, float]:
    start = perf_counter()
    result = func(*args, **kwargs)
    return result, perf_counter() - start
