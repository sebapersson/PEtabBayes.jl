# [Getting started](@id tutorial)

PEtabBayes.jl performs Bayesian inference for a `PEtabODEProblem` defined with PEtab.jl.
This example defines a growth model and its priors, then samples its posterior.

## Input problem

We consider a simple saturated growth model with state ``x(t)`` and observations ``y_i``:

```math
\frac{dx}{dt} = b_2(b_1 - x), \qquad x(0) = 0,
\qquad y_i \mid b_1,b_2,\sigma \sim \mathcal{N}(x(t_i),\sigma^2).
```

The parameters to infer from data are the saturation level ``b_1``, the growth rate ``b_2``,
and the measurement noise standard deviation ``\sigma``.

The synthetic data below use ``b_1 = 1``, ``b_2 = 0.2``, and ``\sigma = 0.03``,
at 101 evenly spaced times from 0 to 2.5. 

For more on creating a parameter estimation
problem, see the [PEtab.jl getting started tutorial](https://sebapersson.github.io/PEtab.jl/stable/tutorial).

## Creating a Bayesian inference problem

In a Bayesian inference problem we need to define prior distributions, encoding our prior
beliefs about the parameters of interest. In this example we set independent priors:
```math
b_1 \sim \operatorname{Uniform}(0,5), \qquad
b_2 \sim \operatorname{LogNormal}(1,1), \qquad
\sigma \sim \operatorname{Gamma}(1,1).
```

For PEtab problems in the PEtab standard format, priors are defined in the
[parameter table](https://petab.readthedocs.io/en/latest/documentation_data_format.html#parameter-table).
Here, we instead consider a model defined directly in Julia: a simple saturated growth
model. First, we define the model and simulate data:

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
oprob = ODEProblem(sys, [x => 0.0, b1 => 1.0, b2 => 0.2], (0.0, 2.5))
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
    PEtabParameter(:b2, value=0.2, scale=:log10, prior=LogNormal(1.0, 1.0)),
    PEtabParameter(:sigma, value=0.03, scale=:lin, prior=Gamma(1.0, 1.0))
]
nothing # hide
```

Priors are evaluated on the parameter’s **linear** (non-transformed) scale. For example,
while `b2` above is estimated on the `log10` scale, the prior applies to `b2` (not
`log10(b2)`). If no prior is provided, the default is a `Uniform` over the parameter bounds.
Finally, build the problem:

```@example tutorial
model = PEtabModel(sys, observables, measurements, pest)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

## Bayesian inference (general setup)

To run Bayesian inference, first construct a `PEtabLogDensity`. This object implements the
[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface and can be
used as the target density for MCMC:

```@example tutorial
using PEtabBayes

target = PEtabBayesLogDensity(petab_prob)
```

## Prior predictive check

Construct the inference target and draw 1000 independent samples from the prior. No initial
point is needed, and the samples are returned on the PEtab estimation scale:

```@example tutorial
prior_chain = PEtabBayes.sample(target, PEtabPrior(), 1000)
nothing # hide
```

Before fitting the model, use these samples to check whether the priors imply plausible
measurements on the scale of the observed data:

```@example tutorial
using Plots

prior_check = predictive_check(prior_chain, target; n_draws = 500)
plot(prior_check, :data_fit)
```

The points are the measurements, and the line and shaded band summarize replicated
measurements generated from the prior, including measurement noise. Predictions that are
implausible for the problem can indicate that the priors need revision. Use
`plot(prior_check, :model_fit)` to inspect the predicted observable without measurement
noise.

## Find initial value

For MCMC, a key choice is the initial value of the Markov chain. For simplicity, one can
start from the nominal parameter vector, but in practice inference should be run with 
multiple chains and dispersed starting points [gelman2020bayesian](@cite):

```@example tutorial
x0_nominal = PEtab.get_x(petab_prob)
nothing # hide
```

Alternatively, use multipathfinder to draw approximate posterior samples and take one as
an initial value. By default, its optimization paths start from prior samples:

```@example tutorial
using Pathfinder
using Logging # hide

with_logger(ConsoleLogger(stderr, Logging.Error)) do # hide
    global pathfinder_result # hide
pathfinder_result = PEtabBayes.multipathfinder(target, 100; nruns = 4)
end # hide
x0 = pathfinder_result.draws[:, 1]
nothing # hide
```

The draws are on the PEtab estimation scale, so they can be passed directly to the sampler.
They are approximate posterior draws; use multiple dispersed starting points for multiple
chains.

## Sampling the posterior

Load AdvancedHMC.jl to enable Hamiltonian Monte Carlo sampling, then run NUTS:

```@example tutorial
using AdvancedHMC

chain = PEtabBayes.sample(target, NUTS(0.8), 2000, x0; n_adapts = 1000)
nothing # hide
```

PEtabBayes internally transforms the initial point to the unconstrained inference scale 
used in NUTS. The returned `FlexiChains.SymChain` contains samples on the PEtab 
estimation scale. For parameters estimated on a log scale, the returned values remain on 
that log scale. The chain can be plotted with:

```@example tutorial
using StatsPlots
plot(chain)
```

To run several chains, pass a vector of initial points (one per chain). The
[`SerialSampling`](@ref), [`ThreadedSampling`](@ref), and [`DistributedSampling`](@ref)
backends control how the chains are run.



## Posterior predictive check

Use the posterior samples in `chain` to generate replicated measurements and compare
them with the observed data. The data-fit plot includes both uncertainty in the
parameters and measurement noise:

```@example tutorial
check = predictive_check(chain, target; n_draws = 500)
plot(check, :data_fit)
```

The points are the measurements; the line and shaded band show the median and the
5th–95th percentile interval of the replicated measurements at each measurement time.
Look for systematic differences between the observations and the predictions, such as
measurements consistently falling outside the band or following a different trend.

To inspect uncertainty in the predicted observable without measurement noise, plot the
model fit instead:

```@example tutorial
plot(check, :model_fit)
```

The `n_draws` keyword controls how many samples are used from the chain.

See the [API reference](@ref API) for sampling, variational inference, and predictive
check options. Model construction and ODE solver configuration belong to PEtab.jl.

## References

```@bibliography
Pages = ["tutorial.md"]
```

