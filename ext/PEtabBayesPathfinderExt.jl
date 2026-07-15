module PEtabBayesPathfinderExt

import Pathfinder
import PEtabBayes
import Distributions
import Random
import StatsBase

const DEFAULT_OPTIMIZER = Pathfinder.Optim.LBFGS(
    m = Pathfinder.DEFAULT_HISTORY_LENGTH,
    linesearch = Pathfinder.Optim.LineSearches.BackTracking(),
    alphaguess = Pathfinder.Optim.LineSearches.InitialHagerZhang(),
)

function PEtabBayes._to_petab_scale(
        result::Pathfinder.PathfinderResult, inference_info::PEtabBayes.InferenceInfo,
    )::Pathfinder.PathfinderResult
    draws = PEtabBayes._to_petab_scale(result.draws, inference_info)

    return Pathfinder.PathfinderResult(
        result.input, result.optimizer, result.rng, result.optim_prob, result.logp,
        result.fit_distribution, draws, result.fit_distribution_transformed, draws,
        result.fit_iteration, result.num_tries, result.optim_solution, result.optim_trace,
        result.fit_distributions, result.elbo_estimates, result.num_bfgs_updates_rejected,
    )
end
function PEtabBayes._to_petab_scale(
        result::Pathfinder.MultiPathfinderResult, inference_info::PEtabBayes.InferenceInfo,
    )::Pathfinder.MultiPathfinderResult
    draws = PEtabBayes._to_petab_scale(result.draws, inference_info)
    pathfinder_results = map(result.pathfinder_results) do pathfinder_result
        PEtabBayes._to_petab_scale(pathfinder_result, inference_info)
    end

    return Pathfinder.MultiPathfinderResult(
        result.input, result.optimizer, result.rng, result.optim_fun, result.logp,
        result.fit_distribution, draws, result.draw_component_ids,
        result.fit_distribution_transformed, draws, pathfinder_results, result.psis_result,
    )
end

function PEtabBayes.multipathfinder(
        log_target::PEtabBayes.PEtabBayesLogDensity, ndraws::Int, init = nothing,
        optimizer = DEFAULT_OPTIMIZER; kwargs...
    )::Pathfinder.MultiPathfinderResult

    if (!isnothing(init))
        # If init is provided, transform to prior/inference scale
        inference_info = log_target.inference_info
        for i in eachindex(init)
            _x = PEtabBayes.to_prior_scale(init[i], log_target)
            init[i] .= inference_info.bijectors(_x)
        end
    end

    # Define the initial sampler for Pathfinder, which samples from the prior
    # distribution of the PEtab problem.
    init_sampler = (rng, x) -> PEtabBayes._sample_prior(rng, x, log_target)

    multi_pathfinder_result = Pathfinder.multipathfinder(
        log_target, ndraws; init = init, init_sampler = init_sampler, optimizer = optimizer,
        kwargs...
    )

    return PEtabBayes._to_petab_scale(multi_pathfinder_result, log_target.inference_info)
end

function _logpdf_eachcol(dist, draws::AbstractMatrix)
    return [Distributions.logpdf(dist, @view draws[:, j]) for j in axes(draws, 2)]
end

function _rand_matrix(rng::Random.AbstractRNG, dist, ndraws::Integer)
    draws = rand(rng, dist, ndraws)

    # For a 1-dimensional target, rand may return a vector.
    # Normalize to the Pathfinder convention: dim x ndraws.
    if draws isa AbstractVector
        draws = reshape(draws, 1, :)
    end
    return draws
end

function PEtabBayes.sample_pathfinder_result(
        result::Pathfinder.MultiPathfinderResult, ndraws::Integer;
        rng::Random.AbstractRNG = result.rng,
        ndraws_per_run::Int = max(
            Pathfinder.DEFAULT_NDRAWS_ELBO,
            cld(ndraws, max(length(result.pathfinder_results), 1)),
        ),
        importance::Bool = result.psis_result !== nothing,
    )
    if !(ndraws > 0)
        throw(ArgumentError("`ndraws` must be positive."))
    end
    if !(ndraws_per_run > 0)
        throw(ArgumentError("`ndraws_per_run` must be positive."))
    end

    pathfinder_results = result.pathfinder_results
    nruns = length(pathfinder_results)

    if !(nruns > 0)
        throw(ArgumentError("`result.pathfinder_results` is empty."))
    end

    fit_distributions = map(r -> r.fit_distribution, pathfinder_results)

    # Draw fresh proposal samples from each single-path approximation q_k.
    proposal_draws_by_component = map(fit_distributions) do q
        _rand_matrix(rng, q, ndraws_per_run)
    end

    proposal_draws = reduce(hcat, proposal_draws_by_component)

    proposal_component_ids = reduce(vcat, [fill(k, ndraws_per_run) for k in 1:nruns])

    inds = axes(proposal_draws, 2)

    sample_inds, psis_result = if importance
        log_densities_fit = reduce(
            vcat,
            map(_logpdf_eachcol, fit_distributions, proposal_draws_by_component),
        )

        log_densities_target = [
            result.logp(@view proposal_draws[:, j]) for j in axes(proposal_draws, 2)
        ]

        log_densities_ratios = log_densities_target .- log_densities_fit

        # PSIS-smoothed importance resampling. This previously delegated to an internal
        # `Pathfinder.resample(rng, inds, log_ratios, ndraws)` method, which has since been
        # refactored, so the equivalent step is done here directly with the public
        # `PSIS.psis` (reached through Pathfinder) and `StatsBase.sample`.
        psis = Pathfinder.PSIS.psis(log_densities_ratios)
        weights = StatsBase.ProbabilityWeights(psis.weights, one(eltype(psis.weights)))
        StatsBase.sample(rng, inds, weights, ndraws; replace = true), psis
    else
        StatsBase.sample(rng, inds, ndraws; replace = true), nothing
    end

    draws = proposal_draws[:, sample_inds]
    draw_component_ids = proposal_component_ids[sample_inds]

    if result.input isa PEtabBayes.PEtabBayesLogDensity
        inference_info = result.input.inference_info
        draws = PEtabBayes._to_petab_scale(draws, inference_info)
        proposal_draws = PEtabBayes._to_petab_scale(proposal_draws, inference_info)
    end

    return (
        draws = draws, draw_component_ids = draw_component_ids, psis_result = psis_result,
        proposal_draws = proposal_draws, proposal_component_ids = proposal_component_ids,
    )
end

end
