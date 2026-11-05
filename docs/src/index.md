# PEtabBayes.jl

PEtabBayes.jl provides Bayesian inference for ODE models defined with PEtab.jl.
Start with a `PEtabODEProblem`, construct a [`PEtabBayesLogDensity`](@ref), and sample or
approximate its posterior distribution.

## Features

- MCMC posterior sampling with [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl),
  including NUTS as in Stan, and [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl).
- Serial, threaded, and distributed sampling of multiple chains.
- Variational inference with [AdvancedVI.jl](https://github.com/TuringLang/AdvancedVI.jl)
  and [Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl).
- Independent prior sampling and prior/posterior predictive checks.
- A [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface
  with parameter transformations and prior corrections.

Load the corresponding inference package to enable its PEtabBayes integration.

## Installation

PEtabBayes.jl requires Julia 1.10 or later. Install it using the Julia package manager:

```julia
using Pkg
Pkg.add("PEtabBayes")
```

Install the packages used by the [getting started tutorial](@ref tutorial) with:

```julia
Pkg.add(["PEtab", "AdvancedHMC", "Pathfinder", "DataFrames", "Distributions",
    "ModelingToolkitBase", "OrdinaryDiffEqRosenbrock", "Plots"])
```

## Getting started and help

Follow the [getting started tutorial](@ref tutorial) or consult the [API reference](@ref API).
For model construction and simulation settings, consult the PEtab.jl documentation.
Report bugs or suggest improvements through the
[PEtabBayes.jl issue tracker](https://github.com/sebapersson/PEtabBayes.jl/issues).
See [Contributing](@ref contribute) for contribution guidelines.
