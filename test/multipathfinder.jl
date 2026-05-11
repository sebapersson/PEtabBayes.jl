using Test
using Random
using Optim
using LineSearches
using Pathfinder
using PEtabBayes
using LogDensityProblems

include(joinpath(@__DIR__, "common.jl"))

@testset "PEtabBayes multipathfinder wrapper" begin
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
            linesearch = LineSearches.BackTracking(),
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
    @testset "multipathfinder sampling" begin
        ndraws_new = 20
        ndraws_per_run = 15
        rng = Random.default_rng()

        # First create an existing MultiPathfinderResult
        init = PEtab.get_startguesses(petab_prob, 10) .|> collect

        result = PEtabBayes.multipathfinder(
            log_target,
            10,
            init;
            nruns = 10,
        )

        @test result isa Pathfinder.MultiPathfinderResult

        @testset "samples new draws with importance resampling" begin
            sample = PEtabBayes.sample_new_multipathfinder_draws(
                result,
                ndraws_new;
                rng = rng,
                ndraws_per_run = ndraws_per_run,
                importance = true,
            )

            @test sample isa NamedTuple
            @test hasproperty(sample, :draws)
            @test hasproperty(sample, :draw_component_ids)
            @test hasproperty(sample, :psis_result)
            @test hasproperty(sample, :proposal_draws)
            @test hasproperty(sample, :proposal_component_ids)

            dim = LogDensityProblems.dimension(log_target)
            nruns = length(result.pathfinder_results)

            @test size(sample.draws) == (dim, ndraws_new)
            @test length(sample.draw_component_ids) == ndraws_new
            @test all(1 .<= sample.draw_component_ids .<= nruns)
            @test all(isfinite, sample.draws)

            @test size(sample.proposal_draws) == (dim, ndraws_per_run * nruns)
            @test length(sample.proposal_component_ids) == ndraws_per_run * nruns
            @test all(1 .<= sample.proposal_component_ids .<= nruns)
            @test all(isfinite, sample.proposal_draws)

            @test sample.psis_result !== nothing
        end

        @testset "samples new draws without importance resampling" begin
            sample = PEtabBayes.sample_new_multipathfinder_draws(
                result,
                ndraws_new;
                rng = Random.default_rng(),
                ndraws_per_run = ndraws_per_run,
                importance = false,
            )

            dim = LogDensityProblems.dimension(log_target)
            nruns = length(result.pathfinder_results)

            @test size(sample.draws) == (dim, ndraws_new)
            @test length(sample.draw_component_ids) == ndraws_new
            @test all(1 .<= sample.draw_component_ids .<= nruns)
            @test all(isfinite, sample.draws)

            @test size(sample.proposal_draws) == (dim, ndraws_per_run * nruns)
            @test length(sample.proposal_component_ids) == ndraws_per_run * nruns
            @test all(isfinite, sample.proposal_draws)

            @test sample.psis_result === nothing
        end

        @testset "rejects invalid draw counts" begin
            @test_throws ArgumentError PEtabBayes.sample_new_multipathfinder_draws(
                result,
                0,
            )

            @test_throws ArgumentError PEtabBayes.sample_new_multipathfinder_draws(
                result,
                10;
                ndraws_per_run = 0,
            )
        end

        @testset "new proposal draws are reproducible with fixed RNG" begin
            sample1 = PEtabBayes.sample_new_multipathfinder_draws(
                result,
                ndraws_new;
                rng = Random.MersenneTwister(1),
                ndraws_per_run = ndraws_per_run,
                importance = false,
            )

            sample2 = PEtabBayes.sample_new_multipathfinder_draws(
                result,
                ndraws_new;
                rng = Random.MersenneTwister(1),
                ndraws_per_run = ndraws_per_run,
                importance = false,
            )

            @test sample1.draws == sample2.draws
            @test sample1.draw_component_ids == sample2.draw_component_ids
            @test sample1.proposal_draws == sample2.proposal_draws
            @test sample1.proposal_component_ids == sample2.proposal_component_ids
        end
    end

end
