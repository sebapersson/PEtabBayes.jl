using PEtabBayes, Distributions, StableRNGs, Test
using HypothesisTests: ExactOneSampleKSTest, pvalue

include(joinpath(@__DIR__, "common.jl"))

b1_dist = Gamma(1.0, 1.0)
b2_dist = LogNormal(1.0, 1.0)
sigma_dist = Uniform(1.0e-3, 1.0e1)
p_est = [
    PEtabParameter(:b1, prior = b1_dist, scale = :lin)
    PEtabParameter(:b2, prior = b2_dist, scale = :log10)
    PEtabParameter(:sigma, lb = 1.0e-3, ub = 1.0e1)
]

_prob = get_prob_saturated(p_est)
log_target = PEtabBayesLogDensity(_prob)

# Test prior sampling returns values on the PEtab parameter scale.
rng = StableRNGs.StableRNG(42)
chain_prior = PEtabBayes.sample(rng, log_target, PEtabPrior(), 100000)
expected_draws = Matrix{Float64}(undef, 100000, 3)
expected_rng = StableRNGs.StableRNG(42)
for (j, prior) in pairs(log_target.inference_info.priors)
    expected_draws[:, j] .= rand(expected_rng, prior, 100000)
end
for i in axes(expected_draws, 1)
    expected_draws[i, :] .= PEtabBayes._to_petab_scale(
        log_target.inference_info.bijectors(@view(expected_draws[i, :])),
        log_target.inference_info,
    )
end
@test Array(chain_prior)[:, :, 1] == expected_draws
