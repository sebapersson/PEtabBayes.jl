function _to_chains_advanced_hmc(
        res, target::PEtabBayesLogDensity; start_time = nothing, end_time = nothing
    )
    # Dependent on method
    n_samples = length(res)
    inference_info = target.inference_info

    out = Array{Float64, 3}(undef, (n_samples, length(inference_info.parameters_id), 1))
    for i in 1:n_samples
        out[i, :, 1] .= _to_petab_scale(res[i].z.θ, inference_info)
    end

    return _to_chain(
        out, inference_info.parameters_id; start_time = start_time, end_time = end_time
    )
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
        out[i, :, 1] .= _to_petab_scale(res.X[:, i], inference_info)
    end

    return _to_chain(
        out, inference_info.parameters_id; start_time = start_time, end_time = end_time
    )
end

# Keep the sampler buffers in (iteration, parameter, chain) order; FlexiChains uses
# (iteration, chain, parameter). Extras are excluded from Array(chain) by default.
function _to_chain(
        draws, parameters_id; source = :posterior, start_time = nothing, end_time = nothing
    )::SymChain
    n_samples, _, n_chains = size(draws)
    sampling_time = isnothing(start_time) || isnothing(end_time) ?
        fill(missing, n_chains) : fill(Dates.value(end_time - start_time) / 1000, n_chains)
    chain = SymChain(
        permutedims(draws, (1, 3, 2)), Tuple(Parameter.(parameters_id));
        sampling_time = sampling_time
    )
    provenance = SymChain(
        n_samples, n_chains, Dict(Extra(:is_prior) => fill(source == :prior, n_samples, n_chains))
    )
    return merge(provenance, chain)
end
