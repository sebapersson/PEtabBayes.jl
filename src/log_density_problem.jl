LogDensityProblems.logdensity(p::PEtabBayesLogDensity, x) = _log_target(x, p)

LogDensityProblems.dimension(p::PEtabBayesLogDensity) = p.dim

LogDensityProblems.capabilities(::PEtabBayesLogDensity) = LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.logdensity_and_gradient(p::PEtabBayesLogDensity, x) = _log_target_gradient(x, p)

function _log_target(
        x_inference::AbstractVector{T}, log_density::PEtabBayesLogDensity
    )::T where {T <: Real}
    @unpack inference_info, prob = log_density

    # Log-posterior
    x_petab_scale = _to_petab_scale(x_inference, inference_info)
    log_target = _log_likelihood(x_petab_scale, prob)

    # Prior + Jacobian correction for transformed parameters
    log_target += log_density.f_prior_correction(x_inference)
    return log_target
end

function _log_target_gradient(
        x_inference::AbstractVector{T}, log_density::PEtabBayesLogDensity
    )::Tuple{T, Vector{T}} where {T <: Real}
    @unpack inference_info, prob = log_density

    x_petab_scale = _to_petab_scale(x_inference, inference_info)
    log_target, log_target_grad = _log_likelihood_gradient(x_petab_scale, prob)

    # Log-posterior with Jacobian correction for transformed parameters
    log_target += log_density.f_prior_correction(x_inference)

    # Gradient with transformation correction
    @unpack prior_correction_grad = log_density.f_prior_correction
    ForwardDiff.gradient!(
        prior_correction_grad, log_density.f_prior_correction, x_inference
    )
    _gradient_to_inference_scale!(
        log_target_grad, x_inference, x_petab_scale, inference_info
    )
    log_target_grad .+= prior_correction_grad

    return log_target, log_target_grad
end
