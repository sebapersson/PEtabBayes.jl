"""
    to_chains(res, target::PEtabLogDensity; kwargs...)::MCMCChains

Converts Bayesian inference results obtained with `PEtabLogDensity` into an `MCMCChains`.

`res` can be the inference results from AdvancedHMC.jl or AdaptiveMCMC.jl. The returned
chain has the parameters on the prior scale.

# Keyword Arguments
- `start_time`: Optional starting time for the inference, obtained with `now()`.
- `end_time`: Optional ending time for the inference, obtained with `now()`.

!!! note
    To use this function, the MCMCChains package must be loaded: `using MCMCChains`
"""

function to_chains(
        res, target::PEtabBayesLogDensity; start_time = nothing, end_time = nothing
    )
    # Dependent on method
    n_samples = length(res)
    inference_info = target.inference_info

    out = Array{Float64, 3}(undef, (n_samples, length(inference_info.parameters_id), 1))
    for i in 1:n_samples
        out[i, :, 1] .= inference_info.inv_bijectors(res[i].z.θ)
    end

    if isnothing(start_time) || isnothing(end_time)
        return Chains(out, inference_info.parameters_id)
    else
        _chain = Chains(out, inference_info.parameters_id)
        return setinfo(_chain, (start_time = start_time, stop_time = end_time))
    end
end


function to_chains(
        res::NamedTuple, target::PEtabBayesLogDensity; start_time = nothing,
        end_time = nothing
    )
    # Dependent on method
    n_samples = size(res.X)[2]
    inference_info = target.inference_info

    out = Array{Float64, 3}(undef, (n_samples, length(inference_info.parameters_id), 1))
    for i in 1:n_samples
        out[i, :, 1] .= inference_info.inv_bijectors(res.X[:, i])
    end

    if isnothing(start_time) || isnothing(end_time)
        return Chains(out, inference_info.parameters_id)
    else
        _chain = Chains(out, inference_info.parameters_id)
        return setinfo(_chain, (start_time = start_time, stop_time = end_time))
    end
end
