function _to_petab_scale(
        x_inference::AbstractVector{<:Real}, inference_info::InferenceInfo
    )::AbstractVector{<:Real}

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
    )::T where {T <: AbstractMatrix{<:Real}}
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
    )::T where {T <: AbstractVector{<:Real}}
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

function _to_inference_scale(
        x_petab_scale::AbstractVector, target::PEtabBayesLogDensity
    )::AbstractVector
    return target.inference_info.bijectors(to_prior_scale(x_petab_scale, target))
end

function _gradient_to_inference_scale!(
        grad::T, x_inference::T, x_petab_scale::T, inference_info::InferenceInfo
    )::Nothing where {T <: AbstractVector{<:Real}}
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
    )::T where {T <: AbstractVector{<:Real}}
    petab_prob = log_target.prob
    inference_info = log_target.inference_info

    new_guess = PEtab.get_startguesses(rng, petab_prob, 1)
    _x = to_prior_scale(new_guess, log_target)

    copyto!(x, inference_info.bijectors(_x))
    return x
end

function _default_uniform_prior_bounds(scale::Symbol)::Tuple{Float64, Float64}
    lower_bound = 1.0e-3
    upper_bound = 1.0e3
    if scale === :log10
        return log10(lower_bound), log10(upper_bound)
    elseif scale === :log
        return log(lower_bound), log(upper_bound)
    elseif scale === :log2
        return log2(lower_bound), log2(upper_bound)
    end
    return lower_bound, upper_bound
end
function _default_uniform_prior(
        parameter_name::Symbol, parameter_scale::Symbol,
        lower_bound_parameter_scale::Float64, upper_bound_parameter_scale::Float64,
        lower_bound_linear_scale::Float64, upper_bound_linear_scale::Float64,
        petab_version::String, defined_in_julia::Bool
    )
    if petab_version == "2.0.0" || defined_in_julia == true
        prior_scale = :lin
        lower_bound = lower_bound_linear_scale
        upper_bound = upper_bound_linear_scale
        bounds_scale = "linear"
    else
        prior_scale = parameter_scale === :lin ? :lin : :parameter_scale
        lower_bound = lower_bound_parameter_scale
        upper_bound = upper_bound_parameter_scale
        bounds_scale = prior_scale === :parameter_scale ? "parameter" : "linear"
    end

    if isinf(lower_bound) || isinf(upper_bound)
        @warn "Lower or upper bound for parameter $(parameter_name) is -inf and/or inf \
            on the $(bounds_scale) scale. Assigning default Uniform prior with \
            linear-scale fallback bounds 1e-3 and 1e3"
        fallback_scale = prior_scale === :parameter_scale ? parameter_scale : :lin
        lower_bound, upper_bound = _default_uniform_prior_bounds(fallback_scale)
    end

    return Uniform(lower_bound, upper_bound), prior_scale
end
