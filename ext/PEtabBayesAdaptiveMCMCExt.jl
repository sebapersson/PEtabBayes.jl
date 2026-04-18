module PEtabBayesAdaptiveMCMCExt

import AdaptiveMCMC
import Dates
import LogDensityProblems
import MCMCChains: Chains
import PEtabBayes

function PEtabBayes._sample(
        log_target::PEtabBayes.PEtabBayesLogDensity, x0::PEtabBayes.InputVector,
        n_samples::Integer, alg::AdaptiveMCMC.AdaptState; kwargs...
    )::Chains
    if haskey(kwargs, :algorithm)
        throw(ArgumentError("The keyword argument `algorithm` is not supported. \
            For `sample`, the sampler algorithm must be passed as an \
            `AdaptiveMCMC.AdaptState` argument, e.g. \
            `sample(target, x0, n_samples, RobustAdaptiveMetropolis(x0))`, rather than \
            `sample(target, x0, n_samples; algorithm = :rwm)`."))
    end
    # alg must be a Vector{AdaptiveState} with length equal the number of tempering steps
    if !haskey(kwargs, :L)
        L = 1
    end
    _alg = [deepcopy(alg) for _ in 1:L]

    # Ensure inference is performed on the inference un-bounded scale
    x0_prior_scale = PEtabBayes.to_prior_scale(x0, log_target)
    x0_inference_scale = log_target.inference_info.bijectors(x0_prior_scale)

    # AdaptiveMCMC needs a function on the form; log_target(x)
    _log_target = let p = log_target
        x -> LogDensityProblems.logdensity(p, x)
    end

    start_time = Dates.now()
    res = AdaptiveMCMC.adaptive_rwm(
        x0_inference_scale, _log_target, 200000; algorithm = _alg, kwargs...
    )
    end_time = Dates.now()

    return PEtabBayes._to_chains_adaptive_mcmc(
        res, log_target; start_time = start_time, end_time = end_time
    )
end

end
