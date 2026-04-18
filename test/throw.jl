using AdaptiveMCMC, PEtabBayes, Distributions, Test

include(joinpath(@__DIR__, "common.jl"))

# Setup
p_est = [
    PEtabParameter(:b1, value = 1.0, lb = 0.0, ub = 5.0, scale = :lin),
    PEtabParameter(:b2, value = 0.2, lb = 0.0, ub = 5.0, scale = :lin),
    PEtabParameter(:sigma, value = 0.03, lb = 1.0e-3, ub = 1.0e2, scale = :lin),
]
prob = get_prob_saturated(p_est)
log_target = PEtabBayesLogDensity(prob)
x0 = get_x(prob)

# Must take at least one sample
@test_throws ArgumentError begin
    _ = PEtabBayes.sample(log_target, RobustAdaptiveMetropolis(x0), 0, x0)
end
# Forbidden keyword
@test_throws ArgumentError begin
    _ = PEtabBayes.sample(
        log_target, RobustAdaptiveMetropolis(x0), 20, x0; algorithm = :rwm
    )
end
