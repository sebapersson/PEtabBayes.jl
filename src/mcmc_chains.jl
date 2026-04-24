function _to_chains_advanced_hmc(
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

function _to_chains_adaptive_mcmc(
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
