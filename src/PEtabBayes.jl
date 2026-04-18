module PEtabBayes

import ArgCheck: @argcheck
import Bijectors
using ComponentArrays: ComponentArray, labels
import Dates
using Distributions: Distribution, Univariate, Continuous, Uniform, logpdf
using LogDensityProblems: LogDensityProblems, LogDensityOrder
import ForwardDiff
using MCMCChains: Chains, setinfo
using PEtab: PEtab, PEtabODEProblem
using SimpleUnPack: @unpack

const ContDistribution = Distribution{Univariate, Continuous}
const InputVector = Union{Vector{<:Real}, ComponentArray{<:Real}}

include("structs.jl")

include("common.jl")
include("likelihood.jl")
include("log_density_problem.jl")
include("mcmc_chains.jl")
include("prior.jl")
include("sample.jl")

export PEtabBayesLogDensity, to_prior_scale, to_chains, sample

end
