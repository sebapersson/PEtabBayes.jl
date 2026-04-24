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
        # Test that show produces non-empty output
        io = IOBuffer()
        show(io, target)
        output = String(take!(io))
        print(output)
        @test !isempty(output)
        @test contains(output, "PEtabBayesLogDensity")
        @test contains(output, "3 parameters")
    end

    @testset "describe() function" begin
        # Capture output from describe(target)
        PEtabBayes.describe(target)
        c = IOCapture.capture() do
            PEtabBayes.describe(target)
        end
        @test !isempty(c.output)
        @test contains(c.output, "PEtabBayesLogDensity")
        @test contains(c.output, "Problem statistics")
        @test contains(c.output, "Inference setup")
    end

    @testset "_describe internal function" begin
        # Test styled version
        output_styled = PEtabBayes._describe(target; styled = true)
        @test !isempty(output_styled)

        # Test non-styled version
        output_plain = PEtabBayes._describe(target; styled = false)
        @test !isempty(output_plain)
    end

    @testset "priors() function" begin
        output = PEtabBayes.priors(target)
        print(output)
        @test !isempty(output)
        @test contains(output, "Priors")
        @test contains(output, "b1")
        @test contains(output, "b2")
        @test contains(output, "sigma")
    end
end
