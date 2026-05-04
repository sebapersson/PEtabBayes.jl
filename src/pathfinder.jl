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
