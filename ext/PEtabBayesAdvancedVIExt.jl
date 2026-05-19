module PEtabBayesAdvancedVIExt

import AdvancedVI
import PEtabBayes
import Random

function PEtabBayes._vi_initialization_to_inference_scale(q_init,
        prob::PEtabBayes.PEtabBayesLogDensity,
    )
    return q_init
end

function PEtabBayes._vi_initialization_to_inference_scale(
        q_init::AdvancedVI.MvLocationScale,
        prob::PEtabBayes.PEtabBayesLogDensity,
    )
    location = PEtabBayes._to_inference_scale(q_init.location, prob)
    return AdvancedVI.MvLocationScale(location, q_init.scale, q_init.dist)
end

function PEtabBayes._vi_initialization_to_inference_scale(
        q_init::AdvancedVI.MvLocationScaleLowRank,
        prob::PEtabBayes.PEtabBayesLogDensity,
    )
    location = PEtabBayes._to_inference_scale(q_init.location, prob)
    return AdvancedVI.MvLocationScaleLowRank(
        location, q_init.scale_diag, q_init.scale_factors, q_init.dist
    )
end

function PEtabBayes._optimize(algorithm, max_iter::Integer,
        prob::PEtabBayes.PEtabBayesLogDensity, q_init, args...; kwargs...
    )
    q_init_inference_scale = PEtabBayes._vi_initialization_to_inference_scale(
        q_init, prob
    )
    return AdvancedVI.optimize(
        algorithm, Int(max_iter), prob, q_init_inference_scale, args...; kwargs...
    )
end

function PEtabBayes._optimize(rng::Random.AbstractRNG, algorithm, max_iter::Integer,
        prob::PEtabBayes.PEtabBayesLogDensity, q_init, args...; kwargs...
    )
    q_init_inference_scale = PEtabBayes._vi_initialization_to_inference_scale(
        q_init, prob
    )
    return AdvancedVI.optimize(
        rng, algorithm, Int(max_iter), prob, q_init_inference_scale, args...; kwargs...
    )
end

end
