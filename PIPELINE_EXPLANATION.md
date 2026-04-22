# Experiment Pipeline Explanation

This file explains what was implemented in the experiment pipeline and the main formulas behind it.

## 1. Model Used

The code implements a **sparse linear Gaussian DAG** model. For each node `j`,

```math
X_j = \sum_{i \in \mathrm{pa}(j)} w_{ij} X_i + \varepsilon_j,
\qquad
\varepsilon_j \sim \mathcal{N}(0, \sigma_j^2)
```

where:

- `pa(j)` is the set of parents of node `j`
- `w_{ij}` is the edge weight from node `i` to node `j`
- `\sigma_j^2` is the noise variance for node `j`

If `W` is the weighted adjacency matrix, then in vector form:

```math
X = W^\top X + \varepsilon
```

so

```math
(I - W^\top) X = \varepsilon
```

and therefore

```math
X = (I - W^\top)^{-1} \varepsilon
```

Since `\varepsilon` is Gaussian, `X` is also Gaussian with mean `0` and covariance

```math
\Sigma = (I - W^\top)^{-1} D (I - W)^{-1},
\qquad
D = \mathrm{diag}(\sigma_1^2, \dots, \sigma_d^2)
```

The precision matrix is

```math
Q = \Sigma^{-1} = (I - W) D^{-1} (I - W^\top)
```

This is what the `GaussianDAG` class computes.

## 2. Graph Structures Implemented

Three DAG structures were added.

### Chain DAG

This is the simplest structure:

```math
0 \to 1 \to 2 \to \cdots \to d-1
```

The model becomes

```math
X_0 = \varepsilon_0
```

and for `j = 1, \dots, d-1`,

```math
X_j = w_{j-1,j} X_{j-1} + \varepsilon_j
```

In the code:

- dimension controls the number of nodes
- each edge only connects one node to the next
- each weight is sampled randomly with a random sign and bounded magnitude

This structure is useful because dependence is easy to understand and posterior correlation can still become strong.

### Layered DAG

Nodes are split into `depth` many layers:

```math
L_1, L_2, \dots, L_K
```

Edges are only allowed from one layer to the next:

```math
i \in L_k,\ j \in L_{k+1}
```

For a node `j` in layer `k+1`,

```math
X_j = \sum_{i \in L_k} w_{ij} X_i + \varepsilon_j
```

but each possible edge is included only with probability equal to the chosen sparsity parameter.

In the code:

- `dimension` is the total number of nodes
- `depth` is the number of layers
- `sparsity` is the probability that a possible cross-layer edge is included

This structure is useful for testing block Gibbs ideas because each layer is a natural candidate block.

### Random Sparse DAG

This structure first samples a random topological ordering of the nodes. If node `i` appears before node `j` in that ordering, then the edge `i \to j` is allowed.

Each allowed edge is then included independently with probability `sparsity`:

```math
\mathbb{P}(i \to j) = p
```

for all pairs consistent with the random ordering.

An optional `max_parents` cap can be used so that each node has at most that many parents.

This gives a flexible synthetic family of sparse DAGs with varying depth and local connectivity.

## 3. Data Generation

The function `sample_from_dag(...)` performs **ancestral sampling**.

Because the graph is acyclic, nodes can be sampled in topological order:

1. sample all root nodes from their Gaussian noise
2. move forward through the graph
3. for each node, compute the linear combination of its sampled parents and add Gaussian noise

Formally, once parents of `j` are already sampled,

```math
X_j \mid X_{\mathrm{pa}(j)} \sim
\mathcal{N}\left(\sum_{i \in \mathrm{pa}(j)} w_{ij} X_i,\ \sigma_j^2 \right)
```

This is the baseline prior simulation method described in your proposal.

## 4. Conditioning and Exact Posterior

The function `select_observations(...)` does three things:

1. draws one full sample from the DAG
2. randomly marks some fraction of nodes as observed
3. computes the exact posterior distribution after conditioning on those observed values

### How the observed nodes are chosen

Let:

```math
d = \text{dimension}, \qquad f = \text{observed\_fraction}
```

The code first computes the number of observed nodes as

```math
m = \mathrm{round}(f d)
```

Then it samples exactly `m` distinct node indices without replacement from the full set

```math
\{0,1,\dots,d-1\}
```

The implementation now supports three observation strategies:

- `uniform`
- `shallow`
- `deep`

For each node `j`, the code first computes three structural quantities:

- `d_j` = depth of node `j`
- `c_j` = number of direct children of node `j`
- `r_j` = number of descendants of node `j`

Here:

- depth means the length of the longest path from any source node to node `j`
- children means immediate outgoing neighbors
- descendants means all nodes reachable from `j` by directed paths

Also define a nonnegative score `a_j` for each node `j`. The sampling scheme uses these scores to do **weighted sampling without replacement**.

That means the total number of observations is fixed at `m`, but some nodes can be made more likely to appear in the observed set.

### Uniform strategy

For the `uniform` strategy,

```math
a_j = 1 \qquad \text{for all } j
```

So every node has the same selection weight.

This means:

- every subset of size `m` has the same probability
- the probability of any particular subset `S` with `|S| = m` is

```math
\mathbb{P}(\text{Observed set} = S) = \frac{1}{\binom{d}{m}}
```

- for any single node `j`, the marginal probability that it is observed is

```math
\mathbb{P}(j \text{ is observed}) = \frac{m}{d}
```

which is approximately `f`, and exactly equal to `f` only when `fd` is already an integer.

### Shallow strategy

For the `shallow` strategy, nodes with smaller depth and stronger downstream influence get larger scores.

```math
a_j =
1
+ \lambda_d (d_{\max} - d_j + 1)
+ \lambda_c c_j
+ \lambda_r r_j
```

where:

- `d_max = \max_j d_j`
- `\lambda_d = depth_weight`
- `\lambda_c = children_weight`
- `\lambda_r = descendant_weight`

After that, the weights are normalized:

```math
p_j = \frac{a_j}{\sum_{k=1}^d a_k}
```

and the algorithm samples `m` distinct observed nodes without replacement using those probabilities.

So under this rule:

- smaller depth gives larger score
- more children gives larger score
- more descendants gives larger score

This is the closest match to your requested idea: earlier nodes with more downstream influence are more likely to be observed.

### Deep strategy

For completeness, the code also includes a reverse-style strategy called `deep`, which gives more weight to nodes that are deeper and less upstream:

```math
a_j =
1
+ \lambda_d (d_j + 1)
+ \lambda_c (c_{\max} - c_j)
+ \lambda_r (r_{\max} - r_j)
```

where:

- `c_max = \max_j c_j`
- `r_max = \max_j r_j`

Again, these weights are normalized:

```math
p_j = \frac{a_j}{\sum_{k=1}^d a_k}
```

and the observed set is sampled without replacement using those normalized probabilities.

### Important detail

For the biased schemes, the probabilities above are the **initial selection weights** used by the weighted-without-replacement sampler. Because the algorithm samples several nodes without replacement, the exact marginal probability that node `j` ends up observed is not just `p_j`; it depends on the full sampling procedure.

But the direction is clear:

- under `shallow`, nodes with lower depth and more downstream influence have higher observation probability
- under `deep`, nodes with larger depth and less downstream influence have higher observation probability
- under `uniform`, all nodes are symmetric

This is still different from making each node observed independently with probability `f`.

### Final distribution of the observed set

Suppose the node scores are

```math
a_1, a_2, \dots, a_d
```

and we want to choose exactly `m` observed nodes.

The algorithm samples nodes **sequentially without replacement**. If the ordered selected nodes are

```math
(i_1, i_2, \dots, i_m),
```

then the probability of that ordered sequence is

```math
\mathbb{P}(i_1, i_2, \dots, i_m)
=
\frac{a_{i_1}}{\sum_{k=1}^d a_k}
\cdot
\frac{a_{i_2}}{\sum_{k=1}^d a_k - a_{i_1}}
\cdot
\frac{a_{i_3}}{\sum_{k=1}^d a_k - a_{i_1} - a_{i_2}}
\cdots
\frac{a_{i_m}}{\sum_{k=1}^d a_k - \sum_{r=1}^{m-1} a_{i_r}}
```

This is the weighted sampling-without-replacement distribution used by the code.

Usually we care about the **final observed set** rather than the order. If the observed set is

```math
S = \{j_1, \dots, j_m\},
```

then its probability is obtained by summing over all `m!` orderings of that set:

```math
\mathbb{P}(\text{Observed set} = S)
=
\sum_{\pi \in \mathrm{Perm}(S)}
\prod_{t=1}^m
\frac{a_{\pi_t}}
{\sum_{k=1}^d a_k - \sum_{r=1}^{t-1} a_{\pi_r}}
```

where `Perm(S)` means all permutations of the elements of `S`.

So the final distribution is:

- not independent Bernoulli sampling
- not usually a simple closed form depending only on the set size
- exactly the unordered set distribution induced by sequential weighted sampling without replacement

In the special `uniform` case, all `a_j = 1`, and this reduces to the uniform distribution over all subsets of size `m`:

```math
\mathbb{P}(\text{Observed set} = S) = \frac{1}{\binom{d}{m}}
```

for every set `S` with `|S| = m`.

If we used independent Bernoulli observation indicators, we would have:

```math
O_j \sim \mathrm{Bernoulli}(f)
```

independently for each `j`, so the total number of observed nodes would be random:

```math
\sum_{j=1}^d O_j \sim \mathrm{Binomial}(d, f)
```

But that is **not** what the current implementation does.

The current implementation always fixes the total observation count at `m = round(fd)`, which is often preferable in experiments because it makes comparisons across runs more controlled while still allowing different observation patterns.

Suppose the Gaussian vector is split into unobserved and observed parts:

```math
X =
\begin{pmatrix}
X_u \\
X_o
\end{pmatrix},
\qquad
\Sigma =
\begin{pmatrix}
\Sigma_{uu} & \Sigma_{uo} \\
\Sigma_{ou} & \Sigma_{oo}
\end{pmatrix}
```

and suppose we observe

```math
X_o = x_o
```

Then the exact Gaussian conditional distribution is

```math
X_u \mid X_o = x_o \sim \mathcal{N}(\mu_{u \mid o}, \Sigma_{u \mid o})
```

with

```math
\mu_{u \mid o} = \Sigma_{uo} \Sigma_{oo}^{-1} x_o
```

and

```math
\Sigma_{u \mid o} = \Sigma_{uu} - \Sigma_{uo} \Sigma_{oo}^{-1} \Sigma_{ou}
```

That is the main formula used in `GaussianDAG.posterior(...)`.

The implementation returns:

- the sampled full state
- which indices were observed
- the observed values
- the exact posterior mean
- the exact posterior covariance

This gives you ground truth for testing MCMC algorithms.

## 5. Block Structure Support

For block Gibbs experiments, the layered graph already provides a natural grouping:

```math
\{L_1, L_2, \dots, L_K\}
```

So the helper `make_block_indices(...)` returns the node indices layer by layer, and `block_sizes_from_layers(...)` reports the size of each block.

This does not run block Gibbs yet. It only prepares the block structure so the samplers can use it later.

## 6. Evaluation Metrics Implemented

The pipeline also includes the main metrics from your project proposal.

### Runtime

The helper `runtime_seconds(func, ...)` measures wall-clock runtime:

```math
\text{runtime} = t_{\text{end}} - t_{\text{start}}
```

### Autocorrelation

For a sample sequence `x_1, \dots, x_n`, the lag-`k` autocorrelation is estimated by

```math
\hat{\rho}_k =
\frac{\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}
{\sum_{t=1}^n (x_t - \bar{x})^2}
```

The function `autocorrelation(...)` computes this per coordinate up to a chosen maximum lag.

### Effective Sample Size

The function `effective_sample_size(...)` uses an initial positive sequence style estimate:

```math
\mathrm{ESS} \approx \frac{n}{1 + 2\sum_{k \ge 1} \rho_k}
```

In practice, the infinite sum is truncated once paired autocorrelation sums become non-positive.

Large ESS means better mixing and less correlation across the chain.

### Posterior Estimation Error

The function `estimation_errors(...)` compares Monte Carlo estimates to the exact Gaussian truth.

If `\hat{\mu}` is the sample mean and `\mu` is the exact mean, then the mean error is

```math
\mathrm{RMSE}_{\text{mean}} =
\sqrt{\frac{1}{d}\sum_{j=1}^d (\hat{\mu}_j - \mu_j)^2}
```

If `\hat{\Sigma}` is the sample covariance and `\Sigma` is the exact covariance, then the variance error is computed on the diagonals:

```math
\mathrm{RMSE}_{\text{var}} =
\sqrt{\frac{1}{d}\sum_{j=1}^d (\hat{\Sigma}_{jj} - \Sigma_{jj})^2}
```

and the full covariance error is measured by the Frobenius norm:

```math
\|\hat{\Sigma} - \Sigma\|_F
```

These metrics match the goals in the proposal:

- runtime
- effective sample size
- autocorrelation and mixing
- error in posterior mean and variance estimates relative to exact Gaussian posterior quantities

## 7. Files Added

The implementation lives in:

- `stats608_pipeline/experiment_pipeline.py`
- `stats608_pipeline/__init__.py`

Validation lives in:

- `tests/test_pipeline.py`

Quick usage notes live in:

- `README.md`

## 8. What This Gives You Next

You now have a reusable setup for:

1. generating synthetic sparse Gaussian DAGs
2. drawing prior samples
3. creating observation patterns
4. computing exact conditional Gaussian targets
5. evaluating samplers against exact ground truth

## 9. Samplers Implemented

Two posterior samplers are now implemented.

### Single-Site Gibbs

The function `single_site_gibbs(...)` samples the latent coordinates one at a time from their exact full conditional distributions.

If the latent posterior is

```math
X_u \mid X_o = x_o \sim \mathcal{N}(\mu, Q^{-1})
```

with precision matrix `Q`, then the full conditional for coordinate `i` is Gaussian:

```math
X_i \mid X_{-i}, X_o = x_o \sim
\mathcal{N}\left(
\frac{h_i - \sum_{j \ne i} Q_{ij} X_j}{Q_{ii}},
\frac{1}{Q_{ii}}
\right)
```

where

```math
h = Q \mu
```

The implementation cycles through each latent coordinate and updates it from this exact conditional Gaussian.

### Random-Walk Metropolis-Hastings

The function `random_walk_metropolis_hastings(...)` runs a generic Gaussian random-walk proposal on the latent coordinates:

```math
X' = X + \tau Z,
\qquad
Z \sim \mathcal{N}(0, I)
```

where `\tau = proposal_scale`.

Because the proposal is symmetric, the acceptance probability is

```math
\alpha(X, X') =
\min\left(
1,
\frac{\pi(X')}{\pi(X)}
\right)
```

For the Gaussian posterior,

```math
\log \pi(x) = -\frac12 (x-\mu)^\top Q (x-\mu) + C
```

so the code compares this quadratic form at the proposal and current state.

The function returns the acceptance rate along with the sampled states.

### Block Gibbs With Layer Blocks

The function `block_gibbs(...)` uses the DAG layers as Gibbs blocks.

After conditioning on observed nodes, each original DAG layer is intersected with the latent node set. Empty intersections are dropped, and the remaining latent groups become the Gibbs blocks.

If a block is indexed by `B` and the remaining latent coordinates are indexed by `-B`, then for a Gaussian posterior with precision matrix `Q` and canonical vector `h = Q\mu`, the block conditional is

```math
X_B \mid X_{-B}, X_o = x_o \sim
\mathcal{N}\left(
Q_{BB}^{-1}(h_B - Q_{B,-B} X_{-B}),
\,
Q_{BB}^{-1}
\right)
```

So each Gibbs update samples an entire layer block jointly from this multivariate Gaussian conditional.

This is the natural block version of the single-site Gibbs sampler in your layered DAG setting.

## 10. Experiment Runner

The function `run_layered_experiment_grid(...)` automates the comparison over layered DAG settings.

It loops over:

- dimension
- depth
- sparsity
- correlation
- observation strategy
- repetition

For each configuration, it:

1. generates one layered DAG
2. creates one conditioning task
3. runs all three samplers on the same posterior target
4. computes runtime, ESS, and posterior estimation errors
5. stores one row per method in a pandas dataframe

The dataframe includes columns such as:

- `method`
- `dimension`
- `depth`
- `sparsity`
- `correlation`
- `observation_strategy`
- `runtime_seconds`
- `ess_mean`
- `ess_min`
- `mean_rmse`
- `variance_rmse`
- `covariance_frobenius_error`
- `acceptance_rate`

Here, `correlation` is implemented as the edge-weight magnitude scale used when generating the layered DAG. It is a practical proxy for posterior dependence strength in the synthetic experiments.

## 11. Plotting Helpers

Three dataframe-based plotting helpers are included:

- `plot_ess(...)`
- `plot_runtime(...)`
- `plot_error(...)`

They work by first aggregating the experiment dataframe by:

- `method`
- an x-axis column such as `correlation` or `dimension`
- an optional facet column such as `observation_strategy`

The aggregation computes the mean and standard deviation of the chosen metric, then plots one line per method with optional error bars.

The generic helper is `plot_experiment_metric(...)`, and the three functions above are just convenience wrappers for common metrics.
