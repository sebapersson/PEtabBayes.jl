module PEtabBayesAdvancedHMCExt

import AdvancedHMC
import ArgCheck: @argcheck
import Dates
import MCMCChains: Chains
import PEtabBayes

const HMCSampler = Union{AdvancedHMC.HMC, AdvancedHMC.NUTS, AdvancedHMC.HMCDA}

function PEtabBayes._sample(
        log_target::PEtabBayes.PEtabBayesLogDensity, alg::HMCSampler,
        n_samples::Integer, x0::PEtabBayes.InputVector; drop_warmup::Bool = true,
        n_adapts::Union{Nothing, Integer} = nothing, verbose::Bool = false,
        progress::Bool = false
    )
    if !isnothing(n_adapts)
        @argcheck n_adapts > 0
    else
        n_adapts = min(div(n_samples, 10), 1_000)
    end

    # Ensure inference is performed on the inference un-bounded scale
    x0_prior_scale = PEtabBayes.to_prior_scale(x0, log_target)
    x0_inference_scale = log_target.inference_info.bijectors(x0_prior_scale)

    start_time = Dates.now()
    res = AdvancedHMC.sample(
        log_target, alg, n_samples; initial_params = x0_inference_scale,
        n_adapts = n_adapts, verbose = verbose, progress = progress,
        drop_warmup = drop_warmup
    )
    end_time = Dates.now()

    return PEtabBayes._to_chains_advanced_hmc(
        res, log_target; start_time = start_time, end_time = end_time
    )
end

end
