#=
    Functions for better printing of relevant PEtabBayes-structs which are
    exported to the user.
=#
import Base.show
import StatsBase

StyledStrings.addface!(:PURPLE => StyledStrings.Face(foreground = 0x008f4093))

function show(io::IO, log_target::PEtabBayesLogDensity)
    @unpack prob, dim = log_target
    name = prob.model_info.model.name
    nest = @sprintf("%d", dim)
    header = StyledStrings.StyledMarkup.styled"{PURPLE:{bold:PEtabBayesLogDensity}} {emphasis:$(name)}: $nest parameters \
        to infer\n(for more statistics, call `describe(log_target)`)\n"
    return print(io, StyledStrings.StyledMarkup.styled"$(header)")
end
function Base.show(io::IO, pc::PEtabPredictiveCheck)
    cond = pc.experiment_id in (nothing, :nothing) ? pc.simulation_id : pc.experiment_id
    cond_str = cond === :__c0__ ? "the default condition" : "condition $cond"
    observables = join((o.observable_id for o in pc.observables), ", ")
    views = join(pc.level, ", ")

    header = StyledStrings.StyledMarkup.styled"{PURPLE:{bold:PEtabPredictiveCheck}} {emphasis:$(pc.source)} for $cond_str"
    body = "\n  observables: $observables\n  $(pc.n_draws) draws ($views)"
    print(io, StyledStrings.StyledMarkup.styled"$header$body")
end

"""
    describe(log_target::PEtabBayesLogDensity)

Print summary and configuration statistics for `log_target`
"""
function StatsBase.describe(log_target::PEtabBayesLogDensity)
    return print(_describe(log_target))
end

function _describe(log_target::PEtabBayesLogDensity; styled::Bool = true)
    # Get problem statistics
    @unpack inference_info, dim, f_prior_correction, prob = log_target
    model = prob.model_info.model
    name = model.name
    nstates = @sprintf("%d", length(PEtab._get_state_ids(model.sys_mutated)))
    nparameters = @sprintf("%d", PEtab._get_n_parameters_sys(model.sys_mutated))
    nest = @sprintf("%d", dim)
    n_observables = length(unique(model.petab_tables[:measurements].observableId))
    n_conditions = length(prob.model_info.simulation_info.conditionids[:experiment])

    header = StyledStrings.StyledMarkup.styled"{PURPLE:{bold:PEtabBayesLogDensity}} {emphasis:$(name)}\n"
    opt_head = StyledStrings.StyledMarkup.styled"{underline:Problem statistics}\n"
    opt1 = "  Parameters to estimate: $nest\n"
    opt2 = "  ODE: $nstates states, $nparameters parameters\n"
    opt3 = "  Observables: $(n_observables)\n"
    opt4 = "  Simulation conditions: $(n_conditions)\n"
    model_stat = StyledStrings.StyledMarkup.styled"$(opt_head)$(opt1)$(opt2)$(opt3)$(opt4)\n"

    opt_head = StyledStrings.StyledMarkup.styled"{underline:Inference setup}\n"
    opt1 = "  Parameters scale : $(inference_info.parameters_scale)\n"
    opt2 = "  Inference dimension: $nest\n"
    comp_stat = StyledStrings.StyledMarkup.styled"$(opt_head)$(opt1)$(opt2)"

    if styled
        return StyledStrings.StyledMarkup.styled"$(header)$(model_stat)$(comp_stat)"
    else
        return "$(header)$(model_stat)$(comp_stat)"
    end
end

function parameters(log_target::PEtabBayesLogDensity)
    @unpack inference_info, dim, f_prior_correction, prob = log_target
    function _format_prior(prior)
        name = Base.typename(typeof(prior)).name
        param_names = fieldnames(typeof(prior))
        prior_params = params(prior)
        param_str = join(["$n=$v" for (n, v) in zip(param_names, prior_params)], ", ")
        return "$name($param_str)"
    end
    priors_formatted = join(["    $(inference_info.parameters_id[i]): $(_format_prior(inference_info.priors[i])) :: $(inference_info.priors_scale[i])" for i in eachindex(inference_info.priors)], "\n")
    priors_stat = "  Priors :\n$(priors_formatted)\n"

    return StyledStrings.StyledMarkup.styled"$(priors_stat)"
end
