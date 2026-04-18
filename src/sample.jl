"""
    sample(
        log_target::PEtabLogDensity, alg::AdaptiveMCMC.AdaptState, n_samples, x0; kwargs...
    )

Draw `n_samples` from the posterior defined by `log_target`, starting from `x0` using the
adaptive MCMC sampler `alg` from AdaptiveMCMC.jl. Returns an `MCMCChains.Chains`.

This is a wrapper around `adaptive_rwm` from
[AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl) and supports the same
keyword arguments.

# Arguments
- `log_target`: Log-posterior density to sample from.
- `alg`: AdaptiveMCMC.jl sampler, provided as `Alg(x0; kwargs...)`. The following adaptive
  samplers are supported:
  - `RobustAdaptiveMetropolis` (recommended): Robust Adaptive Metropolis (RAM) [1].
  - `AdaptiveMetropolis`: Adaptive Metropolis (AM) [2].
  - `AdaptiveScalingMetropolis`: Adaptive Scaling Metropolis (ASM) [3].
  - `AdaptiveScalingWithinAdaptiveMetropolis`: Adaptive scaling within adaptive
    Metropolis [3].
- `n_samples`: Number of samples to draw, including burn-in.
- `x0`: Initial parameter vector on the PEtab estimation scale, for example the output from
  `get_x(petab_prob)`. Can be a `Vector` or `ComponentArray`.

# Keyword arguments
Keyword arguments are passed to `adaptive_rwm`; see
[this page](https://mvihola.github.io/docs/AdaptiveMCMC.jl/#AdaptiveMCMC.adaptive_rwm).

# References
1. Vihola, Matti. "Robust adaptive Metropolis algorithm with coerced acceptance rate." *Statistics and Computing* 22.5 (2012): 997-1008.
2. Haario, Heikki, Eero Saksman, and Johanna Tamminen. "An adaptive Metropolis algorithm." *Bernoulli* 7.2 (2001): 223-242.
3. Andrieu, Christophe, and Johannes Thoms. "A tutorial on adaptive MCMC." *Statistics and Computing* 18.4 (2008): 343-373.
"""
function sample(
        log_target::PEtabBayesLogDensity, alg, n_samples::Integer,
        x0::PEtabBayes.InputVector; kwargs...
    )
    @argcheck n_samples > 0
    return _sample(log_target, alg, n_samples, x0; kwargs...)
end
"""
    sample(
        log_target::PEtabLogDensity, alg::HMCSampler, n_samples, x0; kwargs...
    )

Draw `n_samples` from the posterior defined by `log_target`, starting from `x0`, using the
Hamiltonian Monte Carlo sampler `alg` from AdvancedHMC.jl. Returns an `MCMCChains.Chains`.

This is a wrapper around `AdvancedHMC.sample` from
[AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl).

`log_target`, `x0`, and `n_samples` have the same meaning as in
`sample(log_target, x0, n_samples, alg::AdaptiveMCMC.AdaptState; kwargs...)`.

# Arguments
- `alg`: AdvancedHMC.jl sampler. The following samplers are supported:
  - `NUTS` (recommended): No-U-Turn Sampler.
  - `HMC`: Hamiltonian Monte Carlo.
  - `HMCDA`: Hamiltonian Monte Carlo with dual averaging.

# Keyword arguments
Keyword arguments are passed to `AdvancedHMC.sample`. Supported keyword arguments are:

- `n_adapts::Int = min(div(n_samples, 10), 1_000)`: Number of adaptation steps.
- `drop_warmup::Bool = true`: Whether to drop warmup samples.
- `verbose::Bool = false`: Whether to print sampler output.
- `progress::Bool = false`: Whether to show a progress meter.
"""
function sample(log_target::PEtabBayesLogDensity; kwargs...)
end

function _sample end
