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

# Test prior sampling. ExactOneSampleKSTest tests for a certain sample being drawn from a
# specific distribution.
rng = StableRNGs.StableRNG(42)
chain_prior = PEtabBayes.sample(rng, log_target, PEtabPrior(), 100000)
res_b1 = ExactOneSampleKSTest(chain_prior[:b1].data[:], b1_dist)
res_b2 = ExactOneSampleKSTest(chain_prior[:b2].data[:], b2_dist)
res_sigma = ExactOneSampleKSTest(chain_prior[:sigma].data[:], sigma_dist)
@test pvalue(res_b1) > 3.0e-1
@test pvalue(res_b2) > 3.0e-1
@test pvalue(res_sigma) > 3.0e-1
