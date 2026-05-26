struct InferenceInfo{
        d1 <: Vector{<:ContDistribution},
        d2 <: Vector{<:ContDistribution},
        b1,
        b2,
    }
    priors::d1
    tpriors::d2
    bijectors::b1
    inv_bijectors::b2
    priors_scale::Vector{Symbol}
    parameters_scale::Vector{Symbol}
    parameters_id::Vector{Symbol}
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
        parameter_name::Symbol,
        parameter_scale::Symbol,
        lower_bound_parameter_scale::Float64,
        upper_bound_parameter_scale::Float64,
        lower_bound_linear_scale::Float64,
        upper_bound_linear_scale::Float64,
        petab_version::String,
    )
    if petab_version == "2.0.0"
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

function InferenceInfo(petab_problem::PEtabODEProblem)::InferenceInfo
    @unpack model_info, lower_bounds, upper_bounds, xnominal = petab_problem
    @unpack priors, petab_parameters = model_info
    petab_version = PEtab._get_version(model_info)

    parameter_names = Symbol.(labels(xnominal))
    n_parameters = length(parameter_names)

    priors_dist = Vector{PEtab.ContDistribution}(undef, n_parameters)
    bijectors = Vector(undef, n_parameters)
    priors_scale = similar(parameter_names)
    parameters_scale = similar(parameter_names)

    for (ix, θ) in pairs(parameter_names)
        iθ = nothing

        # ML parameters are always on linear scale
        if ix in model_info.xindices.indices_est[:est_to_mech]
            iθ = findfirst(x -> x == θ, petab_parameters.parameter_id)
            parameters_scale[ix] = petab_parameters.parameter_scale[iθ]
        else
            parameters_scale[ix] = :lin
        end

        # In case the parameter lacks a defined prior we default to a Uniform
        # on parameter scale with lb and ub as bounds
        if !in(ix, priors.ix_prior)
            lower_bound_linear_scale = isnothing(iθ) ? lower_bounds[ix] :
                petab_parameters.lower_bounds[iθ]
            upper_bound_linear_scale = isnothing(iθ) ? upper_bounds[ix] :
                petab_parameters.upper_bounds[iθ]
            priors_dist[ix], priors_scale[ix] = _default_uniform_prior(
                parameter_names[ix],
                parameters_scale[ix],
                lower_bounds[ix],
                upper_bounds[ix],
                lower_bound_linear_scale,
                upper_bound_linear_scale,
                petab_version,
            )
        else
            jx = findfirst(x -> x == ix, priors.ix_prior)
            priors_dist[ix] = priors.distributions[jx]
            priors_scale[ix] = priors.priors_on_parameter_scale[jx] ? :parameter_scale : :lin
        end
        bijectors[ix] = Bijectors.bijector(priors_dist[ix])
    end

    inv_bijectors = Bijectors.Stacked(Bijectors.inverse.(bijectors))
    bijectors = Bijectors.Stacked(bijectors)
    tpriors = Bijectors.transformed.(priors_dist)

    return InferenceInfo(
        priors_dist, tpriors, bijectors, inv_bijectors, priors_scale, parameters_scale,
        parameter_names
    )
end

struct PriorCorrection{T <: InferenceInfo}
    inference_info::T
    prior_correction_grad::Vector{Float64}
end
function (prior_correction::PriorCorrection)(
        x_inference::AbstractVector{T}
    )::T where {T <: Real}
    log_prior = _log_prior(x_inference, prior_correction.inference_info)
    correction = Bijectors.logabsdetjac(
        prior_correction.inference_info.inv_bijectors, x_inference
    )
    return log_prior + correction
end

"""
PEtabBayesLogDensity(prob::PEtabODEProblem)

Create a `LogDensityProblem` using the posterior and gradient functions from `prob`.

This [`LogDensityProblem` interface](https://github.com/tpapp/LogDensityProblems.jl)
defines everything needed to perform Bayesian inference with packages such as
`AdvancedHMC.jl` (which includes algorithms like NUTS, used by `Turing.jl`), and
`AdaptiveMCMC.jl`.
"""
struct PEtabBayesLogDensity{
        T <: InferenceInfo,
        I <: Integer,
        P <: PriorCorrection,
    }
    inference_info::T
    dim::I
    f_prior_correction::P
    prob::PEtabODEProblem
end
function PEtabBayesLogDensity(petab_problem::PEtabODEProblem)::PEtabBayesLogDensity
    @unpack nparameters_estimate = petab_problem
    inference_info = InferenceInfo(petab_problem)
    prior_correction = PriorCorrection(
        inference_info, zeros(Float64, nparameters_estimate)
    )
    return PEtabBayesLogDensity(
        inference_info, nparameters_estimate, prior_correction, petab_problem
    )
end

function (logpotential::PEtabBayesLogDensity)(x)
    return logpotential.logtarget(x)
end
