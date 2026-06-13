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
            lower_bound_linear_scale = isnothing(iθ) ? lower_bounds[ix] : petab_parameters.lower_bounds[iθ]
            upper_bound_linear_scale = isnothing(iθ) ? upper_bounds[ix] : petab_parameters.upper_bounds[iθ]
            priors_dist[ix], priors_scale[ix] = _default_uniform_prior(
                parameter_names[ix], parameters_scale[ix], lower_bounds[ix],
                upper_bounds[ix], lower_bound_linear_scale, upper_bound_linear_scale,
                petab_version, model_info.model.defined_in_julia
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

"""
    PredictiveObservable

Prior or posterior predictive trajectories for a single observable within one simulation
condition.

Instances are normally created by [`predictive_check`](@ref) and consumed by the
`PEtabPredictiveCheck` plotting recipe; they are rarely constructed by hand.

# Fields
- `observable_id::Symbol`: Identifier of the `observableId` in the PEtab observables table.
- `observable_formula::String`: The observable formula, for use as a plot label.
- `t_model::Vector{Float64}`: Time points of the latent model trajectories. A dense grid
   when the observable has no observable parameters; otherwise the measured time points.
- `h::Matrix{Float64}`: Latent model output for the observable, of size
  `(length(t_model), n_draws)`.
- `t_obs::Vector{Float64}`: Time points of the measured data for this observable.
- `y_obs::Vector{Float64}`: Measured data values at `t_obs`.
- `y_rep::Matrix{Float64}`: Data-level predictive draws at `t_obs`, of size
  `(length(t_obs), n_draws)`, obtained by sampling the measurement error model around the
  latent output. Empty (`0×0`) when the predictive check is performed on the `:fit` level.

See also [`PEtabPredictiveCheck`](@ref).
"""
struct PredictiveObservable
    observable_id::Symbol
    observable_formula::String
    t_model::Vector{Float64}
    h::Matrix{Float64}
    t_obs::Vector{Float64}
    y_obs::Vector{Float64}
    y_rep::Matrix{Float64}
end

"""
    PEtabPredictiveCheck

Prior or posterior predictive check for a single simulation condition.

Collects the predictive trajectories of every observable measured in one simulation
condition (or, for PEtab v2.0.0, one experiment), from simulating the model for a set of
parameter vectors drawn from either the prior or the posterior.

# Fields
- `simulation_id::Symbol`: Identifier of the simulation condition.
- `pre_equilibration_id::Union{Symbol, Nothing}`: Pre-equilibration (steady-state)
  condition applied before `simulation_id`, or `nothing` if there is none.
- `experiment_id::Union{Symbol, Nothing}`: Experiment identifier for PEtab v2.0.0 problems,
  or `nothing` for earlier PEtab versions.
- `source::Symbol`: Origin of the parameter draws, either `:prior` or `:posterior`.
- `level::Symbol`: Level of the predictive check. `:fit` stores only the latent model
  trajectories; `:data` additionally stores data-level draws (`y_rep`) sampled from the
  measurement error model.
- `n_draws::Int`: Number of parameter draws used, i.e. the number of columns in each `h`
  (and `y_rep`).
- `observables::Vector{PredictiveObservable}`: Predictive trajectories for each observable
  measured in this condition, in plotting order.

# Indexing

Individual observables can be retrieved by their `observable_id`, and the available ids
queried with [`observable_ids`](@ref):

```julia
pc[:obs_id]            # the PredictiveObservable for :obs_id
observable_ids(pc)     # all observable ids, in plotting order
length(pc)             # number of observables
```

See also [`predictive_check`](@ref) and [`PredictiveObservable`](@ref).
"""
struct PEtabPredictiveCheck
    simulation_id::Symbol
    pre_equilibration_id::Union{Symbol, Nothing}
    experiment_id::Union{Symbol, Nothing}
    source::Symbol
    level::Vector{Symbol}
    n_draws::Int
    observables::Vector{PredictiveObservable}
end

"""
    observable_ids(pc::PEtabPredictiveCheck) -> Vector{Symbol}

Return the `observable_id` of every observable stored in `pc`, in plotting order.
"""
function observable_ids(pc::PEtabPredictiveCheck)::Vector{Symbol}
    return [obs.observable_id for obs in pc.observables]
end

Base.length(pc::PEtabPredictiveCheck)::Int = length(pc.observables)

function Base.getindex(
        pc::PEtabPredictiveCheck, observable_id::Symbol
    )::PredictiveObservable
    i = findfirst(obs -> obs.observable_id == observable_id, pc.observables)
    isnothing(i) && throw(KeyError(observable_id))
    return pc.observables[i]
end

"""
    PEtabPrior

Sample from the prior in the PEtabBayesLogDensity.
"""
struct PEtabPrior end
