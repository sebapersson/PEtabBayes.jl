"""
    predictive_check(
        [rng::AbstractRNG, ] chains::Chains, log_target::PEtabBayesLogDensity; condition = nothing,
        experiment = nothing, observable_ids = nothing, n_draws = nothing, n_tsave = 50,
        model_fit = true, data_fit = true
    )

Compute a prior or posterior predictive check for a single simulation condition.

For all `observable_ids` in the condition, the model is simulated across the parameter
draws in `chains`. Two views can be produced. The model fit (`model_fit`) summarizes the
model's predicted observable, reflecting parameter uncertainty in the ODE model alone. The
data fit (`data_fit`) samples the measurement error model around each prediction, so it
reflects both parameter uncertainty and observation noise, and is the view compared directly
against the data.

# Arguments
- `chains`: Prior or posterior draws, as returned by [`sample`](@ref).
- `log_target`: Log-posterior density `chains` was produced from.

# Keyword arguments
- `condition` / `experiment`: Select the condition, resolved as in `PEtab.get_odesol`; a
  `pre_eq => sim` `Pair` for PEtab v1 pre-equilibration, or `experiment` for PEtab v2.0.0.
  When neither is given the model's default condition is used.
- `observable_ids`: Restrict to a subset of observables. `nothing` uses every observable
  measured in the condition.
- `n_draws`: Number of parameter draws to use, thinned evenly from the chain. Defaults to
  `min(5000, n_samples)`.
- `n_tsave`: Number of time points in the dense grid used for the model-fit trajectories.
- `model_fit`: Whether to compute the latent model-fit trajectories.
- `data_fit`: Whether to compute the data-level predictive replicates.

# Returns

A [`PEtabPredictiveCheck`](@ref), which is plotted with  `plot(pc, :model_fit)` or
`plot(pc, :data_fit)`.
"""
function predictive_check(
        chains::Chains, log_target::PEtabBayesLogDensity; kwargs...
    )::PEtabPredictiveCheck
    rng = Random.default_rng()
    return predictive_check(rng, chains, log_target; kwargs...)
end
function predictive_check(
        rng::Random.AbstractRNG, chains::Chains, log_target::PEtabBayesLogDensity;
        condition::Union{PEtab.ConditionExp, Nothing} = nothing,
        experiment::Union{PEtab.ConditionExp, Nothing} = nothing,
        observable_ids::Union{Vector{Symbol}, Nothing} = nothing,
        n_draws::Union{Nothing, Integer} = nothing, source::Symbol = _infer_source(chains),
        n_tsave::Integer = 50, model_fit::Bool = true, data_fit::Bool = true
    )::PEtabPredictiveCheck
    @argcheck model_fit || data_fit
    @argcheck source in (:prior, :posterior)

    level = Symbol[]
    model_fit && push!(level, :model_fit)
    data_fit && push!(level, :data_fit)

    # Check user provided ids
    model_info = log_target.prob.model_info
    PEtab._check_experiment_id(condition, experiment, model_info)
    simulation_id = PEtab._get_simulation_id(condition, experiment, model_info)
    pre_equilibration_id = PEtab._get_pre_equilibration_id(
        condition, experiment, model_info
    )
    PEtab._check_condition_ids(simulation_id, pre_equilibration_id, model_info)

    if PEtab._get_version(model_info) == "2.0.0" && isnothing(experiment)
        experiment_id = Symbol(first(split("$simulation_id", '_')))
    else
        experiment_id = Symbol(experiment)
    end

    observables_df = model_info.model.petab_tables[:observables]
    if isnothing(observable_ids)
        observable_ids = observables_df[!, :observableId]
    else
        observable_ids = string.(observable_ids)
    end

    sample_values = _get_samples(chains, n_draws)
    n_draws = size(sample_values, 1)
    predictive_observables = PredictiveObservable[]
    for observable_id in observable_ids
        idata = PEtab._get_index_data(condition, experiment, observable_id, model_info)
        isempty(idata) && continue
        predictive_observable = PredictiveObservable(
            rng, log_target, sample_values, observable_id, condition, experiment, n_tsave,
            model_fit, data_fit
        )
        push!(predictive_observables, predictive_observable)
    end

    return PEtabPredictiveCheck(
        simulation_id, pre_equilibration_id, experiment_id, source, level, n_draws,
        predictive_observables
    )
end

# TODO: Optimize to only require on ODE-solve for both model and data fit
function PredictiveObservable(
        rng::Random.AbstractRNG, log_target::PEtabBayesLogDensity,
        sample_values::Matrix{Float64}, observable_id::String, condition, experiment,
        n_tsave::Integer, model_fit::Bool, data_fit::Bool
    )::PredictiveObservable
    # Observed values
    model_info = log_target.prob.model_info
    measurements_df = log_target.prob.model_info.model.petab_tables[:measurements]
    idata = PEtab._get_index_data(condition, experiment, observable_id, model_info)
    t_obs = measurements_df[idata, :time]
    h_obs = measurements_df[idata, :measurement]

    n_draws = size(sample_values, 1)
    if model_fit == true
        h_matrix = zeros(Float64, n_tsave, n_draws)
        cols_drop = Int64[]
        model_fit_ref = Any[]
        for row_idx in 1:n_draws
            x_prior_scale = sample_values[row_idx, :]
            x_petab_scale = _prior_to_petab_scale(x_prior_scale, log_target.inference_info)

            model_fit = PEtab._get_observable(
                x_petab_scale, log_target.prob, condition, experiment, observable_id;
                n_tsave = n_tsave
            )

            # Could not solve the ODE for provided parameter
            if isempty(model_fit.h_mod)
                push!(cols_drop, row_idx)
                continue
            end

            # Save to build the struct
            if isempty(model_fit_ref)
                push!(model_fit_ref, model_fit)
            end
            h_matrix[:, row_idx] .= model_fit.h_mod
            h_matrix[:, Not(cols_drop)]
        end
        t_mod = model_fit_ref[1].t_mod
    else
        h_matrix = zeros(Float64, 0, 0)
        t_mod = Float64[]
    end

    if data_fit == true
        y_rep = zeros(Float64, length(idata), n_draws)
        cols_drop = Int64[]

        # View to avoid allocating new memory every iteration
        petab_measurements = model_info.petab_measurements
        h_mod = @view petab_measurements.simulated_values[idata]
        sigma_mod = @view petab_measurements.sigma_values[idata]
        dist_mod = @view petab_measurements.noise_distributions[idata]

        for row_idx in 1:n_draws
            x_prior_scale = sample_values[row_idx, :]
            x_petab_scale = _prior_to_petab_scale(x_prior_scale, log_target.inference_info)

            nllh_val = log_target.prob.nllh(x_petab_scale)
            if isinf(nllh_val)
                push!(cols_drop, row_idx)
                continue
            end

            y_rep[:, row_idx] .= _get_y_rep(rng, h_mod, sigma_mod, dist_mod)
        end
        y_rep = y_rep[:, Not(cols_drop)]
    else
        y_rep = zeros(Float64, 0, 0)
    end

    observables_df = model_info.model.petab_tables[:observables]
    obs_idx = findfirst(x -> x == observable_id, observables_df.observableId)
    return PEtabBayes.PredictiveObservable(
        Symbol(observable_id), observables_df.observableFormula[obs_idx], t_mod, h_matrix,
        t_obs, h_obs, y_rep
    )
end

function _infer_source(chains::Chains)::Symbol
    haskey(chains.info, :source) || return :posterior
    return Symbol(chains.info.source)
end

function _get_samples(
        chains::Chains, n_draws::Union{Nothing, Integer}
    )::Matrix{Float64}
    sample_values = Array(chains)
    n_samples = size(sample_values, 1)
    n_draws = isnothing(n_draws) ? min(5000, n_samples) : n_draws
    @argcheck n_draws <= n_samples

    # Thin the chain if there n_draws < n_samples
    if n_samples == n_draws
        return sample_values
    end
    rows = unique(round.(Int, range(1, n_samples; length = n_draws)))
    return sample_values[rows, :]
end

function _get_y_rep(
        rng::Random.AbstractRNG, h::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
        noise_distributions::AbstractVector{Symbol};
    )::Vector{Float64}
    y_rep = zeros(Float64, length(h))
    for i in eachindex(h, sigma, noise_distributions)
        dist, transform = PEtab.NOISE_DISTRIBUTIONS[noise_distributions[i]]
        y_rep[i] = rand(rng, dist(transform(h[i]), sigma[i]))
    end
    return y_rep
end

function _prior_to_petab_scale(
        x_prior_scale::T, inference_info::InferenceInfo
    )::T where T <: AbstractVector{<:Real}
    x_petab_scale = similar(x_prior_scale)
    for i in eachindex(x_petab_scale)
        inference_info.priors_scale[i] === :parameter_scale && continue
        x_petab_scale[i] = PEtab.transform_x(
            x_prior_scale[i], inference_info.parameters_scale[i], to_xscale = true
        )
    end
    return x_petab_scale
end
