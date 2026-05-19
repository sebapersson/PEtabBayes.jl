using ADTypes, AdvancedVI, LinearAlgebra, Optimisers, PEtabBayes, Random, ReverseDiff,
    Statistics, Test

include(joinpath(@__DIR__, "common.jl"))

@testset "AdvancedVI optimize wrapper" begin
    pest = [
        PEtabParameter(:b1, value = 1.0, lb = 0.0, ub = 5.0, scale = :lin),
        PEtabParameter(:b2, value = 0.2, lb = 0.0, ub = 5.0, scale = :lin),
        PEtabParameter(:sigma, value = 0.03, lb = 1.0e-3, ub = 1.0e2, scale = :lin),
    ]
    prob = get_prob_saturated(pest)
    log_target = PEtabBayesLogDensity(prob)
    reference_stats = get_reference_stats(
        joinpath(@__DIR__, "inference_results", "Saturated_chain.csv")
    )

    d = PEtabBayes.LogDensityProblems.dimension(log_target)
    x0_parameter_scale = collect(get_x(prob))
    x0_inference_scale = log_target.inference_info.bijectors(
        PEtabBayes.to_prior_scale(x0_parameter_scale, log_target)
    )
    q_init = AdvancedVI.MeanFieldGaussian(x0_parameter_scale, Diagonal(fill(0.1, d)))
    q_init_inference_scale = PEtabBayes._vi_initialization_to_inference_scale(
        q_init, log_target
    )
    @test q_init.location == x0_parameter_scale
    @test q_init_inference_scale.location ≈ x0_inference_scale
    @test q_init_inference_scale.scale == q_init.scale

    alg = AdvancedVI.KLMinRepGradDescent(
        ADTypes.AutoReverseDiff();
        optimizer = Optimisers.Adam(2.0e-2),
        n_samples = 10,
        operator = AdvancedVI.ClipScale(),
    )

    rng = Random.default_rng()
    Random.seed!(rng, 1234)
    q_out, info, state = PEtabBayes.optimize(
        rng, alg, 200, log_target, q_init; show_progress = false
    )

    @test q_out isa PEtabBayes.ParameterScaleVariationalDistribution
    @test q_out.inference_scale_distribution isa typeof(q_init)
    @test length(info) == 200
    @test state !== nothing

    rng_direct = Random.MersenneTwister(4321)
    draws_inference_scale = rand(rng_direct, q_out.inference_scale_distribution, 20)
    direct_draws_parameter_scale = PEtabBayes._to_petab_scale(
        draws_inference_scale, log_target.inference_info
    )
    rng_wrapped = Random.MersenneTwister(4321)
    wrapped_draws_parameter_scale = rand(rng_wrapped, q_out, 20)
    @test wrapped_draws_parameter_scale ≈ direct_draws_parameter_scale
    @test size(rand(q_out, 3)) == (d, 3)

    draws_parameter_scale = rand(rng, q_out, 5_000)

    vi_mean = vec(mean(draws_parameter_scale; dims = 2))
    vi_std = vec(std(draws_parameter_scale; dims = 2))

    lower_bounds = collect(prob.lower_bounds)
    upper_bounds = collect(prob.upper_bounds)
    @test all(lower_bounds .<= draws_parameter_scale)
    @test all(draws_parameter_scale .<= upper_bounds)

    @test reference_stats.nt.mean[1] ≈ vi_mean[1] atol = 6.0e-1
    @test reference_stats.nt.mean[2] ≈ vi_mean[2] atol = 5.0e-2
    @test reference_stats.nt.mean[3] ≈ vi_mean[3] atol = 2.0e-2
    @test all(isfinite, vi_std)
    @test all(vi_std .> 0.0)
    @test all(vi_std .< 3 .* reference_stats.nt.std)
end
