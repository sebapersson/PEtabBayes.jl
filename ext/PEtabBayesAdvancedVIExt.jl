module PEtabBayesAdvancedVIExt

import AdvancedVI
import PEtabBayes
import Random

function PEtabBayes._optimize(
        algorithm,
        max_iter::Integer,
        prob::PEtabBayes.PEtabBayesLogDensity,
        q_init,
        args...;
        kwargs...
    )
    return AdvancedVI.optimize(algorithm, Int(max_iter), prob, q_init, args...; kwargs...)
end

function PEtabBayes._optimize(
        rng::Random.AbstractRNG,
        algorithm,
        max_iter::Integer,
        prob::PEtabBayes.PEtabBayesLogDensity,
        q_init,
        args...;
        kwargs...
    )
    return AdvancedVI.optimize(
        rng, algorithm, Int(max_iter), prob, q_init, args...; kwargs...
    )
end

end
