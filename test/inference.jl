using PEtabBayes, Distributions, Random, Test, AdaptiveMCMC, AdvancedHMC, LogDensityProblems

include(joinpath(@__DIR__, "common.jl"))

@testset "Check inference linear priors + parameters" begin
    _b1 = PEtabParameter(:b1, value = 1.0, lb = 0.0, ub = 5.0, scale = :lin)
    _b2 = PEtabParameter(:b2, value = 0.2, lb = 0.0, ub = 5.0, scale = :lin)
    _sigma = PEtabParameter(:sigma, value = 0.03, lb = 1.0e-3, ub = 1.0e2, scale = :lin)
    pest = [_b1, _b2, _sigma]
    prob = get_prob_saturated(pest)
    log_target = PEtabBayesLogDensity(prob)
    x0 = get_x(prob)

    # Reference chain based on 10,000 iterations
    reference_stats = get_reference_stats(
        joinpath(@__DIR__, "inference_results", "Saturated_chain.csv")
    )

    # HMC inference
    chain_hmc = PEtabBayes.sample(
        log_target, NUTS(0.8), 3000, x0; n_adapts = 1000, drop_warmup = true,
        progress = false, verbose = true
    )
    hmc_stats = summarystats(chain_hmc)
    @testset "HMC" begin
        @test reference_stats[:b1, stat = :mean] ≈ hmc_stats[:b1, stat = :mean] atol = 2.0e-1
        @test reference_stats[:b2, stat = :mean] ≈ hmc_stats[:b2, stat = :mean] atol = 1.0e-2
        @test reference_stats[:sigma, stat = :mean] ≈ hmc_stats[:sigma, stat = :mean] atol = 1.0e-2
        @test reference_stats[:b1, stat = :std] ≈ hmc_stats[:b1, stat = :std] atol = 1.0e-1
        @test reference_stats[:b2, stat = :std] ≈ hmc_stats[:b2, stat = :std] atol = 1.0e-2
        @test reference_stats[:sigma, stat = :std] ≈ hmc_stats[:sigma, stat = :std] atol = 1.0e-2
    end

    # AdaptiveMCMC
    Random.seed!(1234)
    chain_adapt1 = PEtabBayes.sample(
        log_target, RobustAdaptiveMetropolis(x0), 200000, x0; progress = false
    )
    adaptive_stats1 = summarystats(chain_adapt1)
    @testset "Adaptive MCMC RAM" begin
        @test reference_stats[:b1, stat = :mean] ≈ adaptive_stats1[:b1, stat = :mean] atol = 2.0e-1
        @test reference_stats[:b2, stat = :mean] ≈ adaptive_stats1[:b2, stat = :mean] atol = 1.0e-2
        @test reference_stats[:sigma, stat = :mean] ≈ adaptive_stats1[:sigma, stat = :mean] atol = 1.0e-2
        @test reference_stats[:b1, stat = :std] ≈ adaptive_stats1[:b1, stat = :std] atol = 5.0e-1
        @test reference_stats[:b2, stat = :std] ≈ adaptive_stats1[:b2, stat = :std] atol = 1.0e-2
        @test reference_stats[:sigma, stat = :std] ≈ adaptive_stats1[:sigma, stat = :std] atol = 1.0e-2
    end
    # Test other adaptive MCMC sampler
    chain_adapt2 = PEtabBayes.sample(log_target, AdaptiveMetropolis(x0), 200000, x0)
    adaptive_stats2 = summarystats(chain_adapt2)
    @testset "Adaptive MCMC AM" begin
        @test reference_stats[:b1, stat = :mean] ≈ adaptive_stats2[:b1, stat = :mean] atol = 2.0e-1
        @test reference_stats[:b2, stat = :mean] ≈ adaptive_stats2[:b2, stat = :mean] atol = 1.0e-2
        @test reference_stats[:sigma, stat = :mean] ≈ adaptive_stats2[:sigma, stat = :mean] atol = 1.0e-2
        @test reference_stats[:b1, stat = :std] ≈ adaptive_stats2[:b1, stat = :std] atol = 5.0e-1
        @test reference_stats[:b2, stat = :std] ≈ adaptive_stats2[:b2, stat = :std] atol = 1.0e-2
        @test reference_stats[:sigma, stat = :std] ≈ adaptive_stats2[:sigma, stat = :std] atol = 1.0e-2
    end
end
