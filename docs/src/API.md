```@meta
CollapsedDocStrings=true
```

# [PEtabBayes.jl API](@id API)

PEtabBayes operates on `PEtabODEProblem`s constructed with PEtab.jl. Inference backends
are enabled by loading AdvancedHMC.jl, AdaptiveMCMC.jl, AdvancedVI.jl, or Pathfinder.jl.
Load Plots.jl to enable predictive-check plotting.

```@autodocs
Modules = [PEtabBayes]
```
