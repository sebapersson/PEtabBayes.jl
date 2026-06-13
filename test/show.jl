using PEtab, PEtabBayes, OrdinaryDiffEqRosenbrock, Distributions, Random, DataFrames, Test,
    ModelingToolkitBase, IOCapture
using ModelingToolkitBase: t_nounits as t, D_nounits as D

include(joinpath(@__DIR__, "common.jl"))

@testset "Show and Describe" begin
    _b1 = PEtabParameter(:b1, value = 1.0, lb = 0.0, ub = 5.0, scale = :lin)
    _b2 = PEtabParameter(:b2, value = 0.2, lb = 0.0, ub = 5.0, scale = :lin)
    _sigma = PEtabParameter(:sigma, value = 0.03, lb = 1.0e-3, ub = 1.0e2, scale = :lin)
    pest = [_b1, _b2, _sigma]
    prob = get_prob_saturated(pest)
    target = PEtabBayesLogDensity(prob)

    @testset "show() function" begin
        @test "$target" ==
            "PEtabBayesLogDensity ODESystemModel: 3 parameters to infer\n" *
            "(for more statistics, call `describe(logdensity)`)\n"
    end

    @testset "describe() function" begin
        c = IOCapture.capture() do
            describe(target)
        end

        expected =
            "PEtabBayesLogDensity ODESystemModel\n" *
            "Problem statistics\n" *
            "  Parameters to estimate: 3\n" *
            "  ODE: 1 states, 2 parameters\n" *
            "  Observables: 1\n" *
            "  Simulation conditions: 1\n" *
            "\n" *
            "Inference setup\n" *
            "  Parameters scale : [:lin, :lin, :lin]\n" *
            "  Inference dimension: 3\n"

        @test c.output == expected
    end

    @testset "parameters() function" begin
        expected =
            "  Priors :\n" *
            "    b1: Uniform(a=0.0, b=5.0) :: lin\n" *
            "    b2: Uniform(a=0.0, b=5.0) :: lin\n" *
            "    sigma: Uniform(a=0.001, b=100.0) :: lin\n"

        @test "$(PEtabBayes.parameters(target))" == expected
    end
end
