"""
    multipathfinder(
        log_target::PEtabBayesLogDensity, ndraws::Integer; kwargs...
    )

Run Pathfinder.jl's multipathfinder on `log_target` and draw `ndraws` approximate samples
from the target distribution.

This is a wrapper around the variation inference algorithm Pathfinder [1], specifically the
`multipathfinder` from [Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl) and
supports the same keyword arguments.

# Arguments
- `log_target`: The log-density of the target distribution.
- `ndraws`: Number of approximate posterior draws to return.
- `init`: Optional initial points for the optimization. If not provided, the
  initial points will be sampled from the prior distribution of the PEtab
  problem.
- `optimizer`: An optimizer from Optim.jl to use for the optimization. Defaults
  to LBFGS with BackTracking linesearch instead of the default linesearch in
  Pathfinder.jl.
# Keyword arguments
Keyword arguments are passed to `Pathfinder.multipathfinder`.

# References
1. Zhang, Lu, Bob Carpenter, Andrew Gelman, and Aki Vehtari. "Pathfinder:
   Parallel quasi-Newton variational inference." *Journal of Machine Learning
   Research* 23.306 (2022): 1-49.
"""
function multipathfinder end

"""
    sample_pathfinder_result(result, ndraws; rng, ndraws_per_run, importance)

Generate new draws from an existing `Pathfinder.MultiPathfinderResult`
without rerunning the single-path Pathfinder optimization steps.

The function samples fresh proposal draws from each stored single-path
`fit_distribution`, then optionally performs the same PSIS importance
resampling step used internally by `Pathfinder.multipathfinder`.

# Arguments

- `result::Pathfinder.MultiPathfinderResult`: Existing result returned by
  `Pathfinder.multipathfinder` or the PEtabBayes multipathfinder wrapper.

- `ndraws::Integer`: Number of final draws to return after optional resampling.

# Keyword arguments

- `rng::Random.AbstractRNG = result.rng`: Random number generator used for
  sampling proposal draws and for resampling.

- `ndraws_per_run::Int`:
  Number of fresh proposal draws to generate from each single-path Pathfinder
  approximation. The total number of proposal draws before resampling is
  `ndraws_per_run * length(result.pathfinder_results)`.

- `importance::Bool = result.psis_result !== nothing`: Whether to perform
  PSIS importance resampling using target and proposal log densities. If
  `false`, draws are sampled uniformly from the fresh proposal pool.

# Returns

A named tuple with the fields:

- `draws`: Matrix of final draws with shape `dimension × ndraws`.
- `draw_component_ids`: Component/run index for each returned draw.
- `psis_result`: PSIS result returned by `Pathfinder.resample` when
  `importance = true`; otherwise `nothing`.
- `proposal_draws`: Matrix containing all fresh proposal draws before
  resampling.
- `proposal_component_ids`: Component/run index for each proposal draw.

# Notes

This function does not rerun the Pathfinder optimization. It only samples from
the already fitted approximations stored in `result.pathfinder_results`.

When `result` was produced for a `PEtabBayesLogDensity`, returned `draws` and
`proposal_draws` are transformed back to the PEtab parameter scale.
"""
function sample_pathfinder_result end
