function _log_likelihood(
        x_petab_scale::AbstractVector{T}, prob::PEtabODEProblem
    )::T where {T <: Real}
    return prob.nllh(x_petab_scale; prior = false) * -1
end

function _log_likelihood_gradient(
        x_petab_scale::AbstractVector{T}, prob::PEtabODEProblem
    )::Tuple{T, Vector{T}} where {T <: Real}
    nllh, nllh_grad = prob.nllh_grad(x_petab_scale; prior = false)
    nllh_grad .*= -1
    return nllh * -1, nllh_grad
end
