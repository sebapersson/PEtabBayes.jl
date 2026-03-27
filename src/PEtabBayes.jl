module PEtabBayes

using Bijectors
using ComponentArrays: ComponentArray, labels
using Distributions
using LogDensityProblems: LogDensityProblems, LogDensityOrder
using ForwardDiff: gradient
using MCMCChains: Chains, setinfo
using PEtab: PEtab, PEtabODEProblem, PEtabParameter, PEtabObservable, PEtabCondition
using ModelingToolkitBase: @unpack

const ContDistribution = Distribution{Univariate, Continuous}
include(joinpath("structs", "inference.jl"))

# Functions that only appear in extension
function compute_llh end
function compute_prior end
function get_correction end
function correct_gradient! end

export PEtabBayesLogDensity, to_prior_scale, to_chains #, InferenceInfo

function to_prior_scale end
function to_chains end

include(joinpath(@__DIR__, "init_structs.jl"))
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "likelihood.jl"))
include(joinpath(@__DIR__, "log_density_problem.jl"))
include(joinpath(@__DIR__, "prior.jl"))
include(joinpath(@__DIR__, "mcmc_chains.jl"))

end # end of module PEtabBayes
