using DataFrames, CSV, MCMCChains, ModelingToolkitBase, OrdinaryDiffEqRosenbrock, PEtab
using ModelingToolkitBase: t_nounits as t, D_nounits as D

function get_reference_stats(path_data)
    # Reference chain 10000 samples via Turing of HMC
    chain_reference_df = CSV.read(path_data, DataFrame)
    _chain_reference = Array{Float64, 3}(undef, 10000, 3, 1)
    _chain_reference[:, :, 1] .= Matrix(chain_reference_df)
    chain_reference = MCMCChains.Chains(_chain_reference)
    reference_stats = summarystats(chain_reference)
    return reference_stats
end

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
        oprob, Rodas5P(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave, tstops = tsave
    )
    obs = _sol[:x] .+ rand(Normal(0.0, 0.03), length(tsave))

    ## Setup the parameter estimation problem
    @parameters sigma
    observables = PEtabObservable(:obs_X, :x, sigma)

    measurements = DataFrame(
        obs_id = "obs_X", time = _sol.t, measurement = obs
    )

    model = PEtabModel(sys, observables, measurements, pest; verbose = false)
    return PEtabODEProblem(
        model; odesolver = ODESolver(Rodas5P(), abstol = 1.0e-6, reltol = 1.0e-6)
    )
end
