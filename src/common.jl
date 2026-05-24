function _to_petab_scale(
        x_inference::AbstractVector, inference_info::InferenceInfo
    )::AbstractVector

    # Transform x into θ - the scale for the priors
    @unpack inv_bijectors, priors_scale, parameters_scale = inference_info
    x_petab_scale = inference_info.inv_bijectors(x_inference)

    for i in eachindex(x_petab_scale)
        priors_scale[i] === :parameter_scale && continue
        x_petab_scale[i] = PEtab.transform_x(
            x_petab_scale[i], parameters_scale[i], to_xscale = true
        )
    end
    return x_petab_scale
end
function _to_petab_scale(
        draws::T, inference_info::PEtabBayes.InferenceInfo
    )::T where T <: AbstractMatrix{<:Real}
    draws_petab_scale = similar(draws)

    for j in axes(draws, 2)
        draws_petab_scale[:, j] .= _to_petab_scale(@view(draws[:, j]), inference_info)
    end

    return draws_petab_scale
end

"""
    to_prior_scale(x_petab_scale, target::PEtabLogDensity)

Transforms parameter `x` from the PEtab problem scale to the prior scale.

This conversion is needed for Bayesian inference, as in PEtab.jl Bayesian inference is
performed on the prior scale.
"""
function to_prior_scale(
        x_petab_scale::T, target::PEtabBayesLogDensity
    )::T where {T <: AbstractVector}
    @unpack parameters_scale, priors_scale = target.inference_info

    x_prior_scale = similar(x_petab_scale)
    for (i, x) in pairs(x_petab_scale)
        if priors_scale[i] == :parameter_scale
            x_prior_scale[i] = x
            continue
        end

        x_prior_scale[i] = PEtab.transform_x(x, parameters_scale[i]; to_xscale = false)
    end
    return x_prior_scale
end

function _gradient_to_inference_scale!(
        grad::T, x_inference::T, x_petab_scale::T, inference_info::InferenceInfo
    )::Nothing where {T <: AbstractVector}
    # Two-step procedure
    # 1 : From parameter to prior-scale
    # 2 : From prior to inference scale
    @unpack inv_bijectors, priors_scale, parameters_scale = inference_info
    for i in eachindex(grad)
        # 1 parameter to prior scale
        if priors_scale[i] != :parameter_scale
            if parameters_scale[i] === :log10
                grad[i] *= exp(Bijectors.logabsdetjac(log10, exp10(x_petab_scale[i])))
            elseif parameters_scale[i] === :log
                grad[i] *= exp(Bijectors.logabsdetjac(log, exp(x_petab_scale[i])))
            elseif parameters_scale[i] === :log2
                grad[i] *= exp(Bijectors.logabsdetjac(log2, exp2(x_petab_scale[i])))
            end
        end

        # 2 from prior to inference scale
        grad[i] *= exp(Bijectors.logabsdetjac(inv_bijectors.bs[i], x_inference[i]))
    end
    return nothing
end

function _sample_prior(
        rng::Random.AbstractRNG, x::T, log_target::PEtabBayes.PEtabBayesLogDensity,
    )::T where T <: AbstractVector{<:Real}
    petab_prob = log_target.prob
    inference_info = log_target.inference_info

    new_guess = PEtab.get_startguesses(rng, petab_prob, 1)
    _x = to_prior_scale(new_guess, log_target)

    copyto!(x, inference_info.bijectors(_x))
    return x
end
