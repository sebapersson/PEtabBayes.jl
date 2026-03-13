module PEtabBayes

import Bijectors
import ComponentArrays
using Distributions
using LogDensityProblems
using LogDensityProblemsAD
import ForwardDiff
using ModelingToolkitBase
using PEtab

const ContDistribution = Distribution{Univariate, Continuous}
include(joinpath("structs", "inference.jl"))

# Functions that only appear in extension
function compute_llh end
function compute_prior end
function get_correction end
function correct_gradient! end

export PEtabBayesLogDensity, to_prior_scale, to_chains #, InferenceInfo, to_prior_scale, to_chains

"""
    to_prior_scale(xpetab, target::PEtabLogDensity)

Transforms parameter `x` from the PEtab problem scale to the prior scale.

This conversion is needed for Bayesian inference, as in PEtab.jl Bayesian inference is
performed on the prior scale.

!!! note
    To use this function, the Bijectors, LogDensityProblems, and LogDensityProblemsAD
    packages must be loaded: `using Bijectors, LogDensityProblems, LogDensityProblemsAD`
"""
function to_prior_scale end

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
function to_chains end

include(joinpath(@__DIR__, "PEtabBayesLogDensityProblems", "Init_structs.jl"))
include(joinpath(@__DIR__, "PEtabBayesLogDensityProblems", "Common.jl"))
include(joinpath(@__DIR__, "PEtabBayesLogDensityProblems", "Likelihood.jl"))
include(joinpath(@__DIR__, "PEtabBayesLogDensityProblems", "LogDensityProblem.jl"))
include(joinpath(@__DIR__, "PEtabBayesLogDensityProblems", "Prior.jl"))

include(joinpath(@__DIR__, "PEtabBayesMCMCChains.jl"))

end # end of module PEtabBayes
