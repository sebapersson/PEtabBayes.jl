function predictive_check(
        chains::Chains, target::PEtabBayesLogDensity;
        condition::Union{PEtab.ConditionExp, Nothing} = nothing,
        experiment::Union{PEtab.ConditionExp, Nothing} = nothing,
        observable_ids::Union{Vector{Symbol}, Nothing} = nothing,
        n_draws::Union{Nothing, Integer} = nothing,
        source::Symbol = _infer_source(chains), n_tsave::Integer = 50,
    )::PEtabPredictiveCheck
    # Check user provided ids
    model_info = target.prob.model_info
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
            target, sample_values, observable_id, condition, experiment, n_tsave
        )
        push!(predictive_observables, predictive_observable)
    end

    return PEtabPredictiveCheck(
        simulation_id, pre_equilibration_id, experiment_id, source, :fit, n_draws,
        predictive_observables
    )
end

function PredictiveObservable(
        target::PEtabBayesLogDensity, sample_values::Matrix{Float64},
        observable_id::String, condition, experiment, n_tsave::Integer
    )::PredictiveObservable
    n_draws = size(sample_values, 1)
    h_matrix = zeros(Float64, n_tsave, n_draws)
    cols_drop = Int64[]
    model_fit_ref = Any[]

    for row_idx in 1:n_draws
        x_prior_scale = sample_values[row_idx, :]
        x_petab_scale = similar(x_prior_scale)
        for i in eachindex(x_petab_scale)
            target.inference_info.priors_scale[i] === :parameter_scale && continue
            x_petab_scale[i] = PEtab.transform_x(
                x_prior_scale[i], target.inference_info.parameters_scale[i],
                to_xscale = true
            )
        end

        model_fit = PEtab._get_observable(
            x_petab_scale, target.prob, condition, experiment, observable_id;
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
    end

    observables_df = target.prob.model_info.model.petab_tables[:observables]
    obs_idx = findfirst(x -> x == observable_id, observables_df.observableId)
    return PEtabBayes.PredictiveObservable(
        Symbol(observable_id), observables_df.observableFormula[obs_idx],
        model_fit_ref[1].t_mod, h_matrix[:, Not(cols_drop)], model_fit_ref[1].t_obs,
        model_fit_ref[1].h_obs, zeros(0, 0)
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
