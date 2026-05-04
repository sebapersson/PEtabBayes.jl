import Pathfinder
import Optim
import LineSearches
import Random

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

function multipathfinder(
        log_target::PEtabBayes.PEtabBayesLogDensity,
        ndraws::Integer,
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
