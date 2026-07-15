using PEtabBayes, Distributions, Random, Test, AdaptiveMCMC, AdvancedHMC

include(joinpath(@__DIR__, "common.jl"))

@testset "Sampling multiple chains" begin
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

    # Two initial points => two chains. The accuracy tolerances mirror those in
    # `inference.jl`; they check that each returned multi-chain object contains a
    # well-mixed posterior, independent of the sampler seed.
    x0s = [x0, x0]

    function check_stats(chain)
        @test size(chain, 3) == 2
        stats = summarystats(chain)
        @test reference_stats.nt.mean[1] ≈ stats.nt.mean[1] atol = 2.0e-1
        @test reference_stats.nt.mean[2] ≈ stats.nt.mean[2] atol = 1.0e-2
        @test reference_stats.nt.mean[3] ≈ stats.nt.mean[3] atol = 1.0e-2
        @test reference_stats.nt.std[1] ≈ stats.nt.std[1] atol = 5.0e-1
        @test reference_stats.nt.std[2] ≈ stats.nt.std[2] atol = 1.0e-2
        @test reference_stats.nt.std[3] ≈ stats.nt.std[3] atol = 1.0e-2
    end

    @testset "Serial (AdaptiveMCMC)" begin
        chain = PEtabBayes.sample(
            log_target, RobustAdaptiveMetropolis(x0), 200000, x0s; progress = false
        )
        check_stats(chain)
    end

    @testset "Threads (AdaptiveMCMC)" begin
        chain = PEtabBayes.sample(
            log_target, RobustAdaptiveMetropolis(x0), 200000, x0s;
            parallel = ThreadedSampling(), progress = false
        )
        check_stats(chain)
    end

    @testset "Distributed (AdaptiveMCMC)" begin
        chain = PEtabBayes.sample(
            log_target, RobustAdaptiveMetropolis(x0), 200000, x0s;
            parallel = DistributedSampling(2), progress = false
        )
        check_stats(chain)
    end

    @testset "HMC (multiple chains)" begin
        chain = PEtabBayes.sample(
            log_target, NUTS(0.8), 3000, x0s; n_adapts = 1000, drop_warmup = true,
            progress = false, verbose = false, parallel = DistributedSampling(2)
        )
        check_stats(chain)
    end
end
