# [Getting started](@id tutorial)

PEtabBayes.jl performs Bayesian inference for a `PEtabODEProblem` defined with PEtab.jl.
This example defines a growth model and its priors, then samples its posterior.
Code blocks run in order in a shared Julia session when the documentation is built.

## Creating a Bayesian inference problem

For PEtab problems in the PEtab standard format, priors are defined in the
[parameter table](https://petab.readthedocs.io/en/latest/documentation_data_format.html#parameter-table).
Here, we instead consider a model defined directly in Julia: a simple saturated growth
model. First, let’s define the model and simulate data:

```@example tutorial
using DataFrames, Distributions, OrdinaryDiffEqRosenbrock,
    ModelingToolkitBase, PEtab
using ModelingToolkitBase: t_nounits as t, D_nounits as D

ps = @parameters b1 b2
sps = @variables x(t) = 0.0 obs_x(t)
eqs = [
    # Dynamics
    D(x) ~ b2 * (b1 - x)
    # Observables
    obs_x ~ x
]
@named sys_model = System(eqs, t, sps, ps)
sys = mtkcompile(sys_model)

# Simulate data with Normal measurement noise (σ = 0.03)
import Random
Random.seed!(1234)
oprob = ODEProblem(sys, [0.0], (0.0, 2.5), [1.0, 0.2])
tsave = range(0.0, 2.5, 101)
sol = solve(oprob, Rodas5P(); abstol=1e-12, reltol=1e-12, saveat=tsave)
obs = sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))
measurements = DataFrame(obs_id="obs_x", time=sol.t, measurement=obs)

# Observable
@parameters sigma
observables = PEtabObservable(:obs_x, :obs_x, sigma)
nothing # hide
```

Priors are assigned via `PEtabParameter` using any continuous distribution from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl). For example:

```@example tutorial
pest = [
    PEtabParameter(:b1, value=1.0, scale=:lin, prior=Uniform(0.0, 5.0)),
    PEtabParameter(:b2, value=0.2, prior=LogNormal(1.0, 1.0)),
    PEtabParameter(:sigma, value=0.03, scale=:lin, prior=Gamma(1.0, 1.0))
]
nothing # hide
```

Priors are evaluated on the parameter’s **linear** (non-transformed) scale. For example,
while `b2` above is estimated on the `log10` scale, the prior applies to `b2` (not
`log10(b2)`). If no prior is provided, the default is a `Uniform` over the parameter bounds.
Finally, build the problem as usual:

```@example tutorial
model = PEtabModel(sys, observables, measurements, pest)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

## Sampling the posterior

Construct the inference target and obtain an initial point on the PEtab estimation scale:

```@example tutorial
using PEtabBayes

target = PEtabBayesLogDensity(petab_prob)
x0 = PEtab.get_x(petab_prob)
nothing # hide
```

Load AdvancedHMC.jl to enable Hamiltonian Monte Carlo sampling, then run NUTS:

```@example tutorial
using AdvancedHMC

chain = PEtabBayes.sample(target, NUTS(0.8), 2000, x0; n_adapts = 1000)
nothing # hide
```

PEtabBayes transforms the initial point to the unconstrained inference scale internally.
The returned `FlexiChains.SymChain` contains samples on the PEtab estimation scale.
For parameters estimated on a log scale, the returned values remain on that log scale.

To run several chains, pass a vector of initial points (one per chain). The
[`SerialSampling`](@ref), [`ThreadedSampling`](@ref), and [`DistributedSampling`](@ref)
backends control how the chains are run.

## Sampling the prior

Draw independent prior samples without an initial point:

```@example tutorial
prior_chain = PEtabBayes.sample(target, PEtabPrior(), 1000)
nothing # hide
```

Prior samples are also returned on the PEtab estimation scale.

## Predictive checks

Use prior or posterior samples to compare predictions with measurements:

```@example tutorial
using Plots

check = predictive_check(chain, target)
plot(check)
```

See the [API reference](@ref API) for sampling, variational inference, and predictive
check options. Model construction and ODE solver configuration belong to PEtab.jl.
