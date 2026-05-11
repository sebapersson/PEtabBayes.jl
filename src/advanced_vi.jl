"""
    optimize(algorithm, max_iter, prob::PEtabBayesLogDensity, q_init, args...; kwargs...)
    optimize(rng, algorithm, max_iter, prob::PEtabBayesLogDensity, q_init, args...; kwargs...)

Run variational inference on the posterior defined by `prob`, starting from
the initial variational approximation `q_init`, using an algorithm from AdvancedVI.jl.

This is a wrapper around `AdvancedVI.optimize` from
[AdvancedVI.jl](https://turinglang.org/AdvancedVI.jl/stable/). The `PEtabBayesLogDensity`
target is already defined on the unconstrained inference scale, so variational
families with support on `R^d`, such as `AdvancedVI.FullRankGaussian` and
`AdvancedVI.MeanFieldGaussian`, can be passed directly.

# Arguments
- `rng`: Optional random number generator.
- `algorithm`: AdvancedVI.jl variational inference algorithm.
- `max_iter`: Maximum number of optimization iterations.
- `prob`: Log-posterior density to approximate.
- `q_init`: Initial variational approximation.
- `args...`: Additional positional arguments passed to `AdvancedVI.optimize`.

# Keyword arguments
Keyword arguments are passed to `AdvancedVI.optimize`; see
[the AdvancedVI optimize documentation](https://turinglang.org/AdvancedVI.jl/stable/general/#Running-Variational-Inference).
"""
function optimize(
        algorithm, max_iter::Integer, prob::PEtabBayesLogDensity, q_init, args...;
        kwargs...
    )
    @argcheck max_iter > 0
    return _optimize(algorithm, max_iter, prob, q_init, args...; kwargs...)
end

function optimize(
        rng::Random.AbstractRNG, algorithm, max_iter::Integer,
        prob::PEtabBayesLogDensity, q_init,
        args...; kwargs...
    )
    @argcheck max_iter > 0
    return _optimize(rng, algorithm, max_iter, prob, q_init, args...; kwargs...)
end

function _optimize end
