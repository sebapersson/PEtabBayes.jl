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
to the inference scale before calling `AdvancedVI.optimize`. The optimized
variational distribution is returned in a `ParameterScaleVariationalDistribution`
wrapper, so samples drawn from the output are on the PEtab parameter scale.

# Arguments
- `rng`: Optional random number generator.
- `algorithm`: AdvancedVI.jl variational inference algorithm.
- `max_iter`: Maximum number of optimization iterations.
- `prob`: Log-posterior density to approximate.
- `q_init`: Initial variational approximation. For AdvancedVI location-scale
  families, the location is expected on the PEtab parameter scale.
- `args...`: Additional positional arguments passed to `AdvancedVI.optimize`.

# Returns
Returns `(q, info, state)`, where `q` samples on the PEtab parameter scale. The
fitted AdvancedVI distribution on the inference scale is available as
`q.inference_scale_distribution`.

# Keyword arguments
Keyword arguments are passed to `AdvancedVI.optimize`; see
[the AdvancedVI optimize documentation]
(https://turinglang.org/AdvancedVI.jl/stable/general/#Running-Variational-Inference).
"""

struct ParameterScaleVariationalDistribution{Q, I <: InferenceInfo}
    inference_scale_distribution::Q
    inference_info::I
end

function Random.rand(
        rng::Random.AbstractRNG,
        q::ParameterScaleVariationalDistribution,
    )
    draw = rand(rng, q.inference_scale_distribution)
    return _to_petab_scale(draw, q.inference_info)
end

function Random.rand(q::ParameterScaleVariationalDistribution)
    return rand(Random.default_rng(), q)
end

function Random.rand(
        rng::Random.AbstractRNG,
        q::ParameterScaleVariationalDistribution,
        n_samples::Integer,
    )
    draws = rand(rng, q.inference_scale_distribution, n_samples)
    draws_petab_scale = similar(draws)

    for j in axes(draws, 2)
        draws_petab_scale[:, j] .= _to_petab_scale(@view(draws[:, j]), q.inference_info)
    end

    return draws_petab_scale
end

function Random.rand(q::ParameterScaleVariationalDistribution, n_samples::Integer)
    return rand(Random.default_rng(), q, n_samples)
end

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
