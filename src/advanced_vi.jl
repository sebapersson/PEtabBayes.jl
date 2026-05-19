"""
    optimize(algorithm, max_iter, prob::PEtabBayesLogDensity, q_init, args...; kwargs...)
    optimize(rng, algorithm, max_iter, prob::PEtabBayesLogDensity, q_init, args...; kwargs...)

Run variational inference on the posterior defined by `prob`, starting from
the initial variational approximation `q_init`, using an algorithm from AdvancedVI.jl.

This is a wrapper around `AdvancedVI.optimize` from
[AdvancedVI.jl](https://turinglang.org/AdvancedVI.jl/stable/). The `PEtabBayesLogDensity`
target is defined on the unconstrained inference scale, but the location of
location-scale initial variational approximations such as
`AdvancedVI.FullRankGaussian` and `AdvancedVI.MeanFieldGaussian` should be
provided on the PEtab parameter scale. The wrapper converts the initial location
to the inference scale before calling `AdvancedVI.optimize`.

# Arguments
- `rng`: Optional random number generator.
- `algorithm`: AdvancedVI.jl variational inference algorithm.
- `max_iter`: Maximum number of optimization iterations.
- `prob`: Log-posterior density to approximate.
- `q_init`: Initial variational approximation. For AdvancedVI location-scale
  families, the location is expected on the PEtab parameter scale.
- `args...`: Additional positional arguments passed to `AdvancedVI.optimize`.

# Keyword arguments
Keyword arguments are passed to `AdvancedVI.optimize`; see
[the AdvancedVI optimize documentation]
(https://turinglang.org/AdvancedVI.jl/stable/general/#Running-Variational-Inference).
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
function _vi_initialization_to_inference_scale end
