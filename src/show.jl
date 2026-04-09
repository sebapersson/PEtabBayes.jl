#=
    Functions for better printing of relevant PEtabBayes-structs which are
    exported to the user.
=#
import Base.show

StyledStrings.addface!(:PURPLE => StyledStrings.Face(foreground=0x008f4093))

function show(io::IO, logdensity::PEtabBayesLogDensity)
    @unpack prob, dim = logdensity
    name = prob.model_info.model.name
    nest = @sprintf("%d", dim)
    header = styled"{PURPLE:{bold:PEtabBayesLogDensity}} {emphasis:$(name)}: $nest parameters \
        to estimate\n(for more statistics, call `describe(logdensity)`)"
    return print(io, styled"$(header)")
end

"""
    describe(logdensity::PEtabBayesLogDensity)

Print summary and configuration statistics for `logdensity`
"""
function describe(logdensity::PEtabBayesLogDensity)
    return print(_describe(logdensity))
end

function _describe(logdensity::PEtabBayesLogDensity; styled::Bool=true)
    # Get problem statistics
    @unpack inference_info, dim, f_prior_correction, prob = logdensity
    model = prob.model_info.model
    name = model.name
    nstates = @sprintf("%d", length(PEtab._get_state_ids(model.sys_mutated)))
    nparameters = @sprintf("%d", PEtab._get_n_parameters_sys(model.sys_mutated))
    nest = @sprintf("%d", dim)
    n_observables = length(unique(model.petab_tables[:measurements].observableId))
    n_conditions = length(prob.model_info.simulation_info.conditionids[:experiment])

    header = styled"{PURPLE:{bold:PEtabBayesLogDensity}} {emphasis:$(name)}\n"
    opt_head = styled"{underline:Problem statistics}\n"
    opt1 = "  Parameters to estimate: $nest\n"
    opt2 = "  ODE: $nstates states, $nparameters parameters\n"
    opt3 = "  Observables: $(n_observables)\n"
    opt4 = "  Simulation conditions: $(n_conditions)\n"
    model_stat = styled"$(opt_head)$(opt1)$(opt2)$(opt3)$(opt4)\n"

    opt_head = styled"{underline:Inference setup}\n"
    function _format_prior(prior)
        name = Base.typename(typeof(prior)).name
        param_names = fieldnames(typeof(prior))
        prior_params = params(prior)
        param_str = join(["$n=$v" for (n, v) in zip(param_names, prior_params)], ", ")
        return "$name($param_str)"
    end
    priors_formatted = join(["    $(inference_info.parameters_id[i]): $(_format_prior(inference_info.priors[i])) :: $(inference_info.priors_scale[i])" for i in eachindex(inference_info.priors)], "\n")
    opt1 = "  Priors :\n$(priors_formatted)\n"
    opt2 = "  Parameters scale : $(inference_info.parameters_scale)\n"
    opt3 = "  Inference dimension: $nest\n"
    comp_stat = styled"$(opt_head)$(opt1)$(opt2)$(opt3)"

    if styled
        return styled"$(header)$(model_stat)$(comp_stat)"
    else
        return "$(header)$(model_stat)$(comp_stat)"
    end
end
