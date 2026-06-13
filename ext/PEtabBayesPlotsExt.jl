module PEtabBayesPlotsExt

import Plots
import Statistics
import PEtabBayes: PEtabPredictiveCheck, PredictiveObservable

# Plot a prior/posterior predictive check. All requested observables are drawn on the same
# axes, one colour each; use `observable_ids` to control which are shown. Per observable:
# the measured data, plus the model fit summarised across draws as a central line
# (`summary`) and/or a credible band (`quantiles`). Each series is labelled so the legend
# states what the line and ribbon represent, including whether they are prior or posterior.
# Operates on the latent trajectories `h` (the `:model_fit` level); data-level `y_rep`
# intervals are a later addition.
Plots.@recipe function f(
        pc::PEtabPredictiveCheck; observable_ids = nothing, summary = :median,
        quantiles = (0.05, 0.95), observable_id_label = false
    )
    observables = _select_observables(pc, observable_ids)
    src = string(pc.source)
    pct = isnothing(quantiles) ? 0 : round(Int, 100 * (last(quantiles) - first(quantiles)))

    title --> _predictive_title(pc)
    xguide --> "Time"
    yguide --> "Value"
    legend --> :topright

    for (i, o) in enumerate(observables)
        label_obs = observable_id_label ? string(o.observable_id) : o.observable_formula

        central = isnothing(summary) ? nothing : _central(o.h, summary)
        band = isnothing(quantiles) ? nothing : _band(o.h, quantiles)

        # Credible band (drawn first so it sits behind the line and points).
        if !isnothing(band)
            Plots.@series begin
                seriestype := :path
                linealpha := 0
                fillrange := band.lower
                fillalpha := 0.2
                fillcolor := i
                label := "$(label_obs) ($(src) $(pct)% CI)"
                o.t_model, band.upper
            end
        end

        # Central summary line, sharing the observable's colour.
        if !isnothing(central)
            Plots.@series begin
                seriestype := :path
                linecolor := i
                linewidth := 2
                label := "$(label_obs) ($(src) $(summary))"
                o.t_model, central
            end
        end

        # Measured data on top.
        Plots.@series begin
            seriestype := :scatter
            markercolor := i
            markersize := 3
            label := "$(label_obs) (data)"
            o.t_obs, o.y_obs
        end
    end
end

function _select_observables(
        pc::PEtabPredictiveCheck, observable_ids::Union{Vector{Symbol}, Nothing}
    )::Vector{PredictiveObservable}
    isnothing(observable_ids) && return pc.observables
    return [pc[observable_id] for observable_id in observable_ids]
end

# Per-time-point central summary across the draws (columns of `h`).
function _central(h::AbstractMatrix, summary)::Vector{Float64}
    f = if summary === :median
        Statistics.median
    elseif summary === :mean
        Statistics.mean
    elseif summary isa Function
        summary
    else
        error("`summary` must be :median, :mean, a function, or nothing (got $summary)")
    end
    return [f(@view h[k, :]) for k in axes(h, 1)]
end

# Per-time-point lower/upper quantile band across the draws.
function _band(h::AbstractMatrix, quantiles)::NamedTuple
    lo, hi = first(quantiles), last(quantiles)
    lower = [Statistics.quantile(@view(h[k, :]), lo) for k in axes(h, 1)]
    upper = [Statistics.quantile(@view(h[k, :]), hi) for k in axes(h, 1)]
    return (lower = lower, upper = upper)
end

# Title logic matching PEtab.jl's optimised-solution recipe.
function _predictive_title(pc::PEtabPredictiveCheck)::String
    if !(isnothing(pc.experiment_id) || pc.experiment_id != :nothing)
        return "Experiment: $(pc.experiment_id)"
    elseif isnothing(pc.pre_equilibration_id)
        return "Condition: $(pc.simulation_id)"
    end
    return "Condition: $(pc.pre_equilibration_id) => $(pc.simulation_id)"
end

end
