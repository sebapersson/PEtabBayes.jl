using PEtabBayes, Distributions, Random, Test, AdaptiveMCMC, AdvancedHMC, LogDensityProblems

include(joinpath(@__DIR__, "common.jl"))

@testset "Check inference linear priors + parameters" begin
    _b1 = PEtabParameter(:b1, value = 1.0, lb = 0.0, ub = 5.0, scale = :lin)
    _b2 = PEtabParameter(:b2, value = 0.2, lb = 0.0, ub = 5.0, scale = :lin)
    _sigma = PEtabParameter(:sigma, value = 0.03, lb = 1.0e-3, ub = 1.0e2, scale = :lin)
    pest = [_b1, _b2, _sigma]
    prob = get_prob_saturated(pest)
    # Reference chain based on 10,000 iterations
    reference_stats = get_reference_stats(
        joinpath(@__DIR__, "inference_results", "Saturated_chain.csv")
    )

    # HMC inference case
    Random.seed!(123)
    target = PEtabBayesLogDensity(prob)
    sampler = NUTS(0.8)
    xprior = PEtabBayes.to_prior_scale(prob.xnominal_transformed, target)
    xinference = target.inference_info.bijectors(xprior)
    res = AdvancedHMC.sample(
        target, sampler,
        3000;
        n_adapts = 1000,
        initial_params = xinference,
        drop_warmup = true,
        progress = false,
        verbose = true
    )
    chain_hmc = PEtabBayes.to_chains(res, target)
    hmc_stats = summarystats(chain_hmc)
    @testset "HMC" begin
        @test reference_stats.nt.mean[1] ≈ hmc_stats.nt.mean[1] atol = 2.0e-1
        @test reference_stats.nt.mean[2] ≈ hmc_stats.nt.mean[2] atol = 1.0e-2
        @test reference_stats.nt.mean[3] ≈ hmc_stats.nt.mean[3] atol = 1.0e-2
        @test reference_stats.nt.std[1] ≈ hmc_stats.nt.std[1] atol = 1.0e-1
        @test reference_stats.nt.std[2] ≈ hmc_stats.nt.std[2] atol = 1.0e-2
        @test reference_stats.nt.std[3] ≈ hmc_stats.nt.std[3] atol = 1.0e-2
    end

    # AdaptiveMCMC
    Random.seed!(1234)
    x0 = get_x(prob)
    chain_adapt1 = PEtabBayes.sample(
        target, x0, 200000, RobustAdaptiveMetropolis(x0), progress = true
    )
    adaptive_stats1 = summarystats(chain_adapt1)
    @testset "Adaptive MCMC RAM" begin
        @test reference_stats.nt.mean[1] ≈ adaptive_stats1.nt.mean[1] atol = 2.0e-1
        @test reference_stats.nt.mean[2] ≈ adaptive_stats1.nt.mean[2] atol = 1.0e-2
        @test reference_stats.nt.mean[3] ≈ adaptive_stats1.nt.mean[3] atol = 1.0e-2
        @test reference_stats.nt.std[1] ≈ adaptive_stats1.nt.std[1] atol = 5.0e-1
        @test reference_stats.nt.std[2] ≈ adaptive_stats1.nt.std[2] atol = 1.0e-2
        @test reference_stats.nt.std[3] ≈ adaptive_stats1.nt.std[3] atol = 1.0e-2
    end
    # Test other adaptive MCMC sampler
    chain_adapt2 = PEtabBayes.sample(
        target, x0, 200000, AdaptiveMetropolis(x0), progress = false
    )
    adaptive_stats2 = summarystats(chain_adapt2)
    @testset "Adaptive MCMC AM" begin
        @test reference_stats.nt.mean[1] ≈ adaptive_stats2.nt.mean[1] atol = 2.0e-1
        @test reference_stats.nt.mean[2] ≈ adaptive_stats2.nt.mean[2] atol = 1.0e-2
        @test reference_stats.nt.mean[3] ≈ adaptive_stats2.nt.mean[3] atol = 1.0e-2
        @test reference_stats.nt.std[1] ≈ adaptive_stats2.nt.std[1] atol = 5.0e-1
        @test reference_stats.nt.std[2] ≈ adaptive_stats2.nt.std[2] atol = 1.0e-2
        @test reference_stats.nt.std[3] ≈ adaptive_stats2.nt.std[3] atol = 1.0e-2
    end
end
