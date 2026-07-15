module PEtabBayes

import ArgCheck: @argcheck
import Bijectors
using ComponentArrays: ComponentArray, labels
import Distributed
using Distributions: Distribution, Univariate, Continuous, Uniform, logpdf, params
using LogDensityProblems: LogDensityProblems, LogDensityOrder
import ForwardDiff
using MCMCChains: Chains, setinfo, chainscat
using PEtab: PEtab, PEtabODEProblem
import Random
using SimpleUnPack: @unpack
import StyledStrings
using Printf: @sprintf
import InvertedIndices: Not

const ContDistribution = Distribution{Univariate, Continuous}
const InputVector = Union{Vector{<:Real}, ComponentArray{<:Real}}

include("structs.jl")

include("common.jl")
include("likelihood.jl")
include("log_density_problem.jl")
include("mcmc_chains.jl")
include("prior.jl")
include("sample.jl")
include("sample_chains.jl")
include("show.jl")
include("pathfinder.jl")
include("advanced_vi.jl")
include("predictive_check.jl")

export PEtabBayesLogDensity, PEtabPrior, to_prior_scale, sample, parameters,
    multipathfinder, optimize, predictive_check, SerialSampling, ThreadedSampling,
    DistributedSampling

end
