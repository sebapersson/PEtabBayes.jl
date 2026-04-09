using PEtab, PEtabBayes, OrdinaryDiffEqRosenbrock, Distributions, Random, DataFrames, Test,
    ModelingToolkitBase, IOCapture
using ModelingToolkitBase: t_nounits as t, D_nounits as D

function get_prob_saturated(pest)::PEtabODEProblem
    sps = @variables x(t) = 0.0
    ps = @parameters b1 b2
    eqs = [D(x) ~ b2 * (b1 - x)]
    @named sys_model = System(eqs, t, sps, ps)
    sys = mtkcompile(sys_model)

    Random.seed!(1234)
    # Simulate the model
    parameter_map = [:b1 => 1.0, :b2 => 0.2]
    u0_map = [:x => 0.0]
    oprob = ODEProblem(sys, merge(Dict(u0_map), Dict(parameter_map)), (0.0, 2.5))
    tsave = collect(range(0.0, 2.5, 101))
    _sol = solve(
        oprob, Rodas5P(), abstol=1.0e-12, reltol=1.0e-12, saveat=tsave, tstops=tsave
    )
    obs = _sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))

    ## Setup the parameter estimation problem
    @parameters sigma
    observables = PEtabObservable(:obs_X, :x, sigma)

    measurements = DataFrame(
        obs_id="obs_X", time=_sol.t, measurement=obs
    )

    model = PEtabModel(sys, observables, measurements, pest; verbose=false)
    return PEtabODEProblem(
        model; odesolver=ODESolver(Rodas5P(), abstol=1.0e-6, reltol=1.0e-6)
    )
end

@testset "Show and Describe" begin
    _b1 = PEtabParameter(:b1, value=1.0, lb=0.0, ub=5.0, scale=:lin)
    _b2 = PEtabParameter(:b2, value=0.2, lb=0.0, ub=5.0, scale=:lin)
    _sigma = PEtabParameter(:sigma, value=0.03, lb=1.0e-3, ub=1.0e2, scale=:lin)
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
        output_styled = PEtabBayes._describe(target; styled=true)
        @test !isempty(output_styled)

        # Test non-styled version
        output_plain = PEtabBayes._describe(target; styled=false)
        @test !isempty(output_plain)
    end
end
