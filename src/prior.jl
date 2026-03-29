function _log_prior(
        x_inference::AbstractVector{T}, inference_info::InferenceInfo
    )::T where {T <: Real}
    x_prior = inference_info.inv_bijectors(x_inference)

    log_prior = 0.0
    for (i, prior) in pairs(inference_info.priors)
        log_prior += logpdf(prior, x_prior[i])
    end
    return log_prior
end
