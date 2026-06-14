module PEtabBayesPlotsExt

import ArgCheck: @argcheck
import Plots
import Statistics
import PEtabBayes: PEtabPredictiveCheck, PredictiveObservable

# Plot a prior/posterior predictive check. All requested observables are drawn on the same
# axes, one colour each; use `observable_ids` to control which are shown. `plot_type`
# selects the view:
#   :model_fit  -> the latent model trajectories `h` on the dense grid `t_model`
#   :data_fit   -> the data-level replicates `y_rep`, which exist only at the measurement
#                  times `t_obs`
# Either view shows the measured data plus a central line (`summary`) and/or a band
# (`quantiles`). The legend states what the line and band represent, including whether they
# are prior or posterior and whether the band is a credible (model fit) or predictive (data
# fit) interval.
Plots.@recipe function f(
        pc::PEtabPredictiveCheck, plot_type = :data_fit; observable_ids = nothing,
        summary = :median, quantiles = (0.05, 0.95), observable_id_label = false
    )
    @argcheck plot_type in (:model_fit, :data_fit)
    if !in(plot_type, pc.level)
        throw(
            ArgumentError(
                "The predictive check holds no $plot_type values; recompute \
                 predictive_check with \
                 $(plot_type === :data_fit ? "data_fit = true" : "model_fit = true"), or \
                 pass plot_type = $(first(pc.level))"
            )
        )
    end

    observables = _select_observables(pc, observable_ids)
    src = string(pc.source)
    pct = isnothing(quantiles) ? 0 : round(Int, 100 * (last(quantiles) - first(quantiles)))
    # Credible interval for the latent model output; predictive interval for the data.
    interval = plot_type === :model_fit ? "CI" : "PI"

    title --> _predictive_title(pc)
    xguide --> "Time"
    yguide --> "Value"
    legend --> :topright

    for (i, o) in enumerate(observables)
        label_obs = observable_id_label ? string(o.observable_id) : o.observable_formula

        # Source the summaries from the matrix/time grid matching the requested view.
        values, t = plot_type === :model_fit ? (o.h, o.t_model) : (o.y_rep, o.t_obs)
        # `t_obs` may be unsorted or repeated, so order before drawing path/band series.
        perm = sortperm(t)
        ts = t[perm]

        central = isnothing(summary) ? nothing : _central(values, summary)[perm]
        band = isnothing(quantiles) ? nothing : _band(values, quantiles)

        # Band (drawn first so it sits behind the line and points).
        if !isnothing(band)
            Plots.@series begin
                seriestype := :path
                linealpha := 0
                fillrange := band.lower[perm]
                fillalpha := 0.2
                fillcolor := i
                label := "$(label_obs) ($(src) $(pct)% $(interval))"
                ts, band.upper[perm]
            end
        end

        # Central summary line, sharing the observable's colour.
        if !isnothing(central)
            Plots.@series begin
                seriestype := :path
                linecolor := i
                linewidth := 2
                label := "$(label_obs) ($(src) $(summary))"
                ts, central
            end
        end

        # Measured data on top (always shown, on its own measurement times).
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

# Per-time-point central summary across the draws (columns of the value matrix).
function _central(values::AbstractMatrix, summary)::Vector{Float64}
    f = if summary === :median
        Statistics.median
    elseif summary === :mean
        Statistics.mean
    elseif summary isa Function
        summary
    else
        error("`summary` must be :median, :mean, a function, or nothing (got $summary)")
    end
    return [f(@view values[k, :]) for k in axes(values, 1)]
end

# Per-time-point lower/upper quantile band across the draws.
function _band(values::AbstractMatrix, quantiles)::NamedTuple
    lo, hi = first(quantiles), last(quantiles)
    lower = [Statistics.quantile(@view(values[k, :]), lo) for k in axes(values, 1)]
    upper = [Statistics.quantile(@view(values[k, :]), hi) for k in axes(values, 1)]
    return (lower = lower, upper = upper)
end

# Title logic matching PEtab.jl's optimised-solution recipe. `experiment_id` is the Symbol
# `:nothing` for non-v2.0.0 problems, so treat that (and a literal `nothing`) as "no
# experiment".
function _predictive_title(pc::PEtabPredictiveCheck)::String
    if !isnothing(pc.experiment_id) && pc.experiment_id !== :nothing
        return "Experiment: $(pc.experiment_id)"
    elseif isnothing(pc.pre_equilibration_id)
        return "Condition: $(pc.simulation_id)"
    end
    return "Condition: $(pc.pre_equilibration_id) => $(pc.simulation_id)"
end

end
