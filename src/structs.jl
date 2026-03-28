struct InferenceInfo{
        d1 <: Vector{<:ContDistribution},
        d2 <: Vector{<:ContDistribution},
        b1,
        b2,
    }
    priors::d1
    tpriors::d2
    bijectors::b1
    inv_bijectors::b2
    priors_scale::Vector{Symbol}
    parameters_scale::Vector{Symbol}
    parameters_id::Vector{Symbol}
end

"""
PEtabBayesLogDensity(prob::PEtabODEProblem)

Create a `LogDensityProblem` using the posterior and gradient functions from `prob`.

This [`LogDensityProblem` interface](https://github.com/tpapp/LogDensityProblems.jl)
defines everything needed to perform Bayesian inference with packages such as
`AdvancedHMC.jl` (which includes algorithms like NUTS, used by `Turing.jl`), and
`AdaptiveMCMC.jl`.
"""
struct PEtabBayesLogDensity{
        T <: InferenceInfo,
        I <: Integer,
        T2 <: AbstractFloat,
    }
    inference_info::T
    logtarget::Any
    logtarget_gradient::Any
    initial_value::Vector{T2}
    dim::I
end
function (logpotential::PEtabBayesLogDensity)(x)
    return logpotential.logtarget(x)
end
