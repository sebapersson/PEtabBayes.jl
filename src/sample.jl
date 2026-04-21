"""
    sample(
        log_target::PEtabLogDensity, x0, n_samples, alg::AdaptiveMCMC.AdaptState; kwargs...
    )

Draw `n_samples` from the posterior defined by `log_target`, starting from `x0` using the
adaptive MCMC sampler `alg` from AdaptiveMCMC.jl. Returns an `MCMCChains.Chains`.

This is a wrapper around `adaptive_rwm` from
[AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl) and supports the same
keyword arguments.

# Arguments
- `log_target`: Log-posterior density to sample from.
- `x0`: Initial parameter vector on the PEtab estimation scale, for example the output from
  `get_x(petab_prob)`. Can be a `Vector` or `ComponentArray`.
- `n_samples`: Number of samples to draw, including burn-in.
- `alg`: AdaptiveMCMC.jl sampler, provided as `Alg(x0; kwargs...)`. The following adaptive
  samplers are supported:
  - `RobustAdaptiveMetropolis` (recommended): Robust Adaptive Metropolis (RAM) [1].
  - `AdaptiveMetropolis`: Adaptive Metropolis (AM) [2].
  - `AdaptiveScalingMetropolis`: Adaptive Scaling Metropolis (ASM) [3].
  - `AdaptiveScalingWithinAdaptiveMetropolis`: Adaptive scaling within adaptive
    Metropolis [3].

# Keyword arguments
Keyword arguments are passed to `adaptive_rwm`; see
[this page](https://mvihola.github.io/docs/AdaptiveMCMC.jl/#AdaptiveMCMC.adaptive_rwm).

# References
1. Vihola, Matti. "Robust adaptive Metropolis algorithm with coerced acceptance rate." *Statistics and Computing* 22.5 (2012): 997-1008.
2. Haario, Heikki, Eero Saksman, and Johanna Tamminen. "An adaptive Metropolis algorithm." *Bernoulli* 7.2 (2001): 223-242.
3. Andrieu, Christophe, and Johannes Thoms. "A tutorial on adaptive MCMC." *Statistics and Computing* 18.4 (2008): 343-373.
"""
function sample(
        log_target::PEtabBayesLogDensity, x0::PEtabBayes.InputVector, n_samples::Integer,
        alg; kwargs...
    )
    @argcheck n_samples > 0
    return _sample(log_target, x0, n_samples, alg; kwargs...)
end
"""
    sample(log_target)

Here options can be added for the AdvancedHMC end.
"""
function sample(log_target::PEtabBayesLogDensity; kwargs...)
end

function _sample end
