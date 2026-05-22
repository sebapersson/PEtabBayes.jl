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

function InferenceInfo(petab_problem::PEtabODEProblem)::InferenceInfo
    @unpack model_info, lower_bounds, upper_bounds, xnominal = petab_problem
    @unpack priors, petab_parameters = model_info

    parameter_names = Symbol.(labels(xnominal))
    n_parameters = length(parameter_names)

    priors_dist = Vector{PEtab.ContDistribution}(undef, n_parameters)
    bijectors = Vector(undef, n_parameters)
    priors_scale = similar(parameter_names)
    parameters_scale = similar(parameter_names)

    for (ix, θ) in pairs(parameter_names)
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
            if isinf(lower_bounds[ix]) || isinf(upper_bounds[ix])
                @warn "Lower or upper bounds for parameter $(parameter_names[ix]) is \
                    -inf and/or inf. Assigning default Uniform prior with linear-scale \
                    fallback bounds 1e-3 and 1e3"
                priors_dist[ix] = Uniform(_default_uniform_prior_bounds(parameters_scale[ix])...)
            else
                priors_dist[ix] = Uniform(lower_bounds[ix], upper_bounds[ix])
            end
            priors_scale[ix] = parameters_scale[ix] === :lin ? :lin : :parameter_scale
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
