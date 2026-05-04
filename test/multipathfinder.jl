using Test
using Random
using Optim
using LineSearches
using Pathfinder
using PEtabBayes
using LogDensityProblems

include(joinpath(@__DIR__, "common.jl"))

@testset "PEtabBayes.multipathfinder wrapper" begin
    rng = Random.default_rng()

    _b1 = PEtabParameter(:b1, value = 1.0, lb = 0.0, ub = 5.0, scale = :lin)
    _b2 = PEtabParameter(:b2, value = 0.2, lb = 0.0, ub = 5.0, scale = :lin)
    _sigma = PEtabParameter(:sigma, value = 0.03, lb = 1.0e-3, ub = 1.0e2, scale = :lin)

    pest = [_b1, _b2, _sigma]
    petab_prob = get_prob_saturated(pest)
    log_target = PEtabBayesLogDensity(petab_prob)

    ndraws = 10
    dim = LogDensityProblems.dimension(log_target)

    function test_multipathfinder_result(result, ndraws, dim)
        @test result isa Pathfinder.MultiPathfinderResult

        @test size(result.draws, 1) == dim
        @test size(result.draws, 2) == ndraws
        @test all(isfinite, result.draws)

        return nothing
    end

    @testset "Explicit init and default optimizer" begin
        init = PEtab.get_startguesses(petab_prob, 10) .|> collect

        result = PEtabBayes.multipathfinder(
            log_target,
            ndraws,
            init,
        )

        test_multipathfinder_result(result, ndraws, dim)
    end

    @testset "Explicit init and user provided optimizer" begin
        init = PEtab.get_startguesses(petab_prob, 10) .|> collect

        user_optimizer = Optim.LBFGS(
            m = Pathfinder.DEFAULT_HISTORY_LENGTH,
            linesearch = LineSearches.HagerZhang(),
            alphaguess = LineSearches.InitialHagerZhang(),
        )

        result = PEtabBayes.multipathfinder(
            log_target,
            ndraws,
            init,
            user_optimizer,
        )

        test_multipathfinder_result(result, ndraws, dim)
    end

    @testset "init_sampler works when explicit init is not provided" begin
        result = PEtabBayes.multipathfinder(
            log_target,
            ndraws;
            nruns = 10,
        )

        test_multipathfinder_result(result, ndraws, dim)
    end

    @testset "petab_prior_sampler mutates and returns valid x" begin
        x = zeros(dim)

        returned_x = PEtabBayes._petab_prior_sampler(rng, x, log_target)

        @test returned_x === x
        @test length(x) == dim
        @test all(isfinite, x)

        @test isfinite(LogDensityProblems.logdensity(log_target, x))
    end

    @testset "invalid optimizer is rejected" begin
        init = PEtab.get_startguesses(petab_prob, 10) .|> collect

        @test_throws Exception PEtabBayes.multipathfinder(
            log_target,
            ndraws,
            init,
            "not an optimizer",
        )
    end
end
