import Pathfinder
import Optim
import LineSearches
import Random
import Distributions

function _petab_prior_sampler(
        rng::Random.AbstractRNG,
        x,
        log_target::PEtabBayes.PEtabBayesLogDensity,
    )
    petab_prob = log_target.prob
    inference_info = log_target.inference_info

    new_guess = PEtab.get_startguesses(petab_prob, 1)
    _x = PEtabBayes.to_prior_scale(new_guess, log_target)

    copyto!(x, inference_info.bijectors(_x))

    return x
end

"""
    multipathfinder(
        log_target::PEtabBayesLogDensity, ndraws::Int; kwargs...
    )

Run multipathfinder from Pathfinder.jl to fit a multivariate normal mixture model to
the target distribution defined by `log_target`. Draw `ndraws` approximate samples from the
target distribution.

This is a wrapper around `multipathfinder` from
[Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl) and supports the same
keyword arguments.

# Arguments
- `log_target`: The log-density of the target distribution.
- `ndraws`: Number of approximate draws to return.
- `init`: Optional initial points for the optimization. If not provided, the initial points
  will be sampled from the prior distribution of the PEtab problem.
- `optimizer`: An optimizer from Optim.jl to use for the optimization. Defaults to LBFGS with
  BackTracking linesearch instead of the default linesearch in Pathfinder.jl.
# Keyword arguments
Keyword arguments are passed to `multipathfinder`; see
[this page](https://mlcolab.github.io/Pathfinder.jl/stable/lib/public/#Multi-path-Pathfinder).

# References
1. Zhang, Lu, Bob Carpenter, Andrew Gelman, and Aki Vehtari. "Pathfinder: Parallel quasi-Newton variational inference." *Journal of Machine Learning Research* 23.306 (2022): 1–49.
"""

function multipathfinder(
        log_target::PEtabBayes.PEtabBayesLogDensity,
        ndraws::Int,
        init = nothing,
        optimizer = Optim.LBFGS(
            m = Pathfinder.DEFAULT_HISTORY_LENGTH,
            linesearch = LineSearches.BackTracking(),
            alphaguess = LineSearches.InitialHagerZhang(),
        );
        kwargs...
    )::Pathfinder.MultiPathfinderResult

    if (!isnothing(init))
        # If init is provided, transform to prior/inference scale
        inference_info = log_target.inference_info
        for i in eachindex(init)
            _x = PEtabBayes.to_prior_scale(init[i], log_target)
            init[i] .= inference_info.bijectors(_x)
        end
    end

    # Define the initial sampler for Pathfinder, which samples from the prior distribution of the PEtab problem
    init_sampler = (rng, x) -> _petab_prior_sampler(rng, x, log_target)

    multi_pathfinder_result = Pathfinder.multipathfinder(
        log_target,
        ndraws;
        init = init,
        init_sampler = init_sampler,
        optimizer = optimizer,
        kwargs...
    )

    return multi_pathfinder_result
end

function _logpdf_eachcol(dist, draws::AbstractMatrix)
    return [Distributions.logpdf(dist, @view draws[:, j]) for j in axes(draws, 2)]
end

function _rand_matrix(
        rng::Random.AbstractRNG,
        dist,
        ndraws::Int,
    )
    draws = rand(rng, dist, ndraws)

    # For a 1-dimensional target, rand may return a vector.
    # Normalize to the Pathfinder convention: dim x ndraws.
    if draws isa AbstractVector
        return reshape(draws, 1, :)
    else
        return draws
    end
end

"""
    sample_new_multipathfinder_draws(result, ndraws; rng, ndraws_per_run, importance)

Generate new draws from an existing `Pathfinder.MultiPathfinderResult`
without rerunning the single-path Pathfinder optimization steps.

The function samples fresh proposal draws from each stored single-path
`fit_distribution`, then optionally performs the same PSIS importance
resampling step used internally by `Pathfinder.multipathfinder`.

# Arguments

- `result::Pathfinder.MultiPathfinderResult`: Existing result returned by
  `Pathfinder.multipathfinder` or the PEtabBayes multipathfinder wrapper.

- `ndraws::Int`: Number of final draws to return after optional resampling.

# Keyword arguments

- `rng::Random.AbstractRNG = result.rng`: Random number generator used for
  sampling proposal draws and for resampling.

- `ndraws_per_run::Int = max(Pathfinder.DEFAULT_NDRAWS_ELBO, cld(ndraws, max(length(result.pathfinder_results), 1)))`:
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
"""
function sample_new_multipathfinder_draws(
        result::Pathfinder.MultiPathfinderResult,
        ndraws::Int;
        rng::Random.AbstractRNG = result.rng,
        ndraws_per_run::Int = max(
            Pathfinder.DEFAULT_NDRAWS_ELBO,
            cld(ndraws, max(length(result.pathfinder_results), 1)),
        ),
        importance::Bool = result.psis_result !== nothing,
    )

    ndraws > 0 || throw(ArgumentError("`ndraws` must be positive."))
    ndraws_per_run > 0 || throw(ArgumentError("`ndraws_per_run` must be positive."))

    pathfinder_results = result.pathfinder_results
    nruns = length(pathfinder_results)

    nruns > 0 || throw(ArgumentError("`result.pathfinder_results` is empty."))

    fit_distributions = map(r -> r.fit_distribution, pathfinder_results)

    # Draw fresh proposal samples from each single-path approximation q_k.
    proposal_draws_by_component = map(fit_distributions) do q
        _rand_matrix(rng, q, ndraws_per_run)
    end

    proposal_draws = reduce(hcat, proposal_draws_by_component)

    proposal_component_ids = reduce(
        vcat, [
            fill(k, ndraws_per_run) for k in 1:nruns
        ]
    )

    inds = axes(proposal_draws, 2)

    sample_inds, psis_result = if importance
        log_densities_fit = reduce(
            vcat, map(
                _logpdf_eachcol,
                fit_distributions,
                proposal_draws_by_component,
            )
        )

        log_densities_target = [
            result.logp(@view proposal_draws[:, j])
                for j in axes(proposal_draws, 2)
        ]

        log_densities_ratios = log_densities_target .- log_densities_fit

        Pathfinder.resample(rng, inds, log_densities_ratios, ndraws)
    else
        Pathfinder.resample(rng, inds, ndraws), nothing
    end

    draws = proposal_draws[:, sample_inds]
    draw_component_ids = proposal_component_ids[sample_inds]

    return (
        draws = draws,
        draw_component_ids = draw_component_ids,
        psis_result = psis_result,
        proposal_draws = proposal_draws,
        proposal_component_ids = proposal_component_ids,
    )
end
