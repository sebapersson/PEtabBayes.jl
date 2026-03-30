function _to_petab_scale(
        x_inference::T, inference_info::InferenceInfo
    )::T where {T <: AbstractVector}

    # Transform x into θ - the scale for the priors
    @unpack inv_bijectors, priors_scale, parameters_scale = inference_info
    x_petab_scale = inference_info.inv_bijectors(x_inference)

    for i in eachindex(x_petab_scale)
        priors_scale[i] === :parameters_scale && continue
        x_petab_scale[i] = _transform_x(x_petab_scale[i], parameters_scale[i])
    end
    return x_petab_scale
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

        x_prior_scale[i] = _transform_x(x, parameters_scale[i]; reverse = true)
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


function _transform_x(x::T, transform::Symbol; reverse::Bool = false)::T where {T <: Real}
    if transform == :log10
        return reverse ? exp10(x) : log10(x)
    elseif transform == :log
        return reverse ? exp(x) : log(x)
    elseif transform == :log2
        return reverse ? exp2(x) : log2(x)
    end
    return x
end
