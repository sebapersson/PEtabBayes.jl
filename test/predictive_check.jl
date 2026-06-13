using PEtabBayes, CSV, DataFrames, Distributions, MCMCChains, PEtab, Plots, StableRNGs,
    Statistics, Test

include(joinpath(@__DIR__, "common.jl"))

b1_dist = Gamma(1.0, 1.0)
b2_dist = Truncated(LogNormal(1.0, 1.0), 0.0, 10.0)
sigma_dist = Uniform(1e-3, 1e1)
p_est = [
    PEtabParameter(:b1, prior = b1_dist, scale = :lin)
    PEtabParameter(:b2, prior = b2_dist, scale = :log10)
    PEtabParameter(:sigma, lb = 1e-3, ub = 1e1)
]
_prob = get_prob_saturated(p_est)
log_target = PEtabBayesLogDensity(_prob)

# ------------------------------------------------------------------------------------------
# Prior predictive check
# ------------------------------------------------------------------------------------------
rng = StableRNGs.StableRNG(42)
chain_prior = PEtabBayes.sample(rng, log_target, PEtabPrior(), 100000)
prior_predictive = predictive_check(
    chain_prior, log_target; n_tsave = 75, n_draws = 2500
)

# Test predictive check agree with manual computation
@test prior_predictive.simulation_id == :__c0__
@test prior_predictive.source == :prior
sample_values = PEtabBayes._get_samples(chain_prior, 2500)
h_test = similar(prior_predictive[:obs_X].h)
for i in axes(sample_values, 1)
    x = [sample_values[i, 1], log10(sample_values[i, 2]), log10(sample_values[i, 3])]
    t_save = range(0.0, 2.5, 75)
    ode_problem, _ = get_odeproblem(x, log_target.prob)
    sol_test = solve(ode_problem, Rodas5P(), abstol = 1e-8, reltol = 1e-8, saveat = t_save)
    h_test[:, i] .= sol_test[:x]
end
@test all(.≈(h_test, prior_predictive[:obs_X].h, atol=1e-6))

# Test plotting is correct
href = prior_predictive[:obs_X].h
p1 = plot(prior_predictive; quantiles = (0.20, 0.80), summary = :mean)
@testset "Prior predictive recipe" begin
    obs = prior_predictive.observables[1]

    # Expected summaries from the known model-fit matrix (n_timepoints × n_draws).
    expected_mean  = vec(Statistics.mean(href; dims = 2))
    expected_lower = [Statistics.quantile(view(href, k, :), 0.20) for k in axes(href, 1)]
    expected_upper = [Statistics.quantile(view(href, k, :), 0.80) for k in axes(href, 1)]

    # One observable -> three series: band, central line, data.
    @test length(p1.series_list) == 3
    band, central, data = p1.series_list[1], p1.series_list[2], p1.series_list[3]

    # Band: y is the upper quantile, fillrange the lower; both on the model grid.
    @test band.plotattributes[:seriestype] == :path
    @test band.plotattributes[:x] == obs.t_model
    @test band.plotattributes[:y] ≈ expected_upper
    @test band.plotattributes[:fillrange] ≈ expected_lower

    # Central line: per-time-point mean on the model grid.
    @test central.plotattributes[:seriestype] == :path
    @test central.plotattributes[:x] == obs.t_model
    @test central.plotattributes[:y] ≈ expected_mean

    # Data scatter at the measured points.
    @test data.plotattributes[:seriestype] == :scatter
    @test data.plotattributes[:x] == obs.t_obs
    @test data.plotattributes[:y] == obs.y_obs
end

# ------------------------------------------------------------------------------------------
# Posterior predictive check
# ------------------------------------------------------------------------------------------
# Read the reference chain
chain_reference_df = CSV.read(
    joinpath(@__DIR__, "inference_results", "Saturated_chain.csv"), DataFrame
)
chain_reference = Array{Float64, 3}(undef, 10000, 3, 1)
chain_reference[:, :, 1] .= Matrix(chain_reference_df)
chain_reference = MCMCChains.Chains(chain_reference)
chain_reference = setinfo(
    chain_reference, merge(chain_reference.info, (source = :posterior,))
)

posterior_predictive = predictive_check(
    chain_reference, log_target; n_draws = 7500
)

# Test predictive check agree with manual computation
@test posterior_predictive.simulation_id == :__c0__
@test posterior_predictive.source == :posterior
sample_values = PEtabBayes._get_samples(chain_reference, 7500)
h_test = similar(posterior_predictive[:obs_X].h)
for i in axes(sample_values, 1)
    x = [sample_values[i, 1], log10(sample_values[i, 2]), log10(sample_values[i, 3])]
    t_save = range(0.0, 2.5, 50)
    ode_problem, _ = get_odeproblem(x, log_target.prob)
    sol_test = solve(ode_problem, Rodas5P(), abstol = 1e-8, reltol = 1e-8, saveat = t_save)
    h_test[:, i] .= sol_test[:x]
end
@test all(.≈(h_test, posterior_predictive[:obs_X].h, atol=1e-6))

href = posterior_predictive[:obs_X].h
p1 = plot(posterior_predictive; quantiles = (0.05, 0.95), summary = :median)
@testset "Posterior predictive recipe" begin
    obs = posterior_predictive.observables[1]

    # Expected summaries from the known model-fit matrix (n_timepoints × n_draws).
    expected_median  = vec(Statistics.median(href; dims = 2))
    expected_lower = [Statistics.quantile(view(href, k, :), 0.05) for k in axes(href, 1)]
    expected_upper = [Statistics.quantile(view(href, k, :), 0.95) for k in axes(href, 1)]

    # One observable -> three series: band, central line, data.
    @test length(p1.series_list) == 3
    band, central, data = p1.series_list[1], p1.series_list[2], p1.series_list[3]

    # Band: y is the upper quantile, fillrange the lower; both on the model grid.
    @test band.plotattributes[:seriestype] == :path
    @test band.plotattributes[:x] == obs.t_model
    @test band.plotattributes[:y] ≈ expected_upper
    @test band.plotattributes[:fillrange] ≈ expected_lower

    # Central line: per-time-point mean on the model grid.
    @test central.plotattributes[:seriestype] == :path
    @test central.plotattributes[:x] == obs.t_model
    @test central.plotattributes[:y] ≈ expected_median

    # Data scatter at the measured points.
    @test data.plotattributes[:seriestype] == :scatter
    @test data.plotattributes[:x] == obs.t_obs
    @test data.plotattributes[:y] == obs.y_obs
end
