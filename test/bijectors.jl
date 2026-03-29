using LogDensityProblems, Bijectors, DataFrames, PEtab, PEtabBayes, FiniteDifferences,
    Catalyst, Test, OrdinaryDiffEqRosenbrock

function _get_petab_problem(p_est)
    rs = @reaction_network begin
        (k1, k2), X1 <--> X2
    end
    u0 = [:X1 => 1.0, :X2 => 0.0]
    @unpack X1 = rs
    observables = PEtabObservable(:obs_X1, X1, 0.5)
    measurements = DataFrame(
        obs_id = "obs_X1", time = [1.0, 2.0, 3.0], measurement = [1.1, 1.2, 1.3]
    )
    model = PEtabModel(rs, observables, measurements, p_est; speciemap = u0)
    return PEtabODEProblem(model)
end

function _get_x_prior_scale(x::Real, scale::Symbol)
    if scale === :lin
        return x
    elseif scale === :log10
        return exp10(x)
    elseif scale === :log
        return exp(x)
    end
end

# Test priors, log-likelihood and its gradient are computed correctly
function _prior_ref(x_prior_scale, p_est)
    log_prior = 0.0
    for i in eachindex(p_est)
        log_prior += logpdf(p_est[i].prior, x_prior_scale[i])
    end
    return log_prior
end

function _llh_ref(x_prior_scale)
    rs = @reaction_network begin
        (k1, k2), X1 <--> X2
    end
    u0 = [:X1 => 1.0, :X2 => 0.0]
    ps = [:k1 => x_prior_scale[1], :k2 => x_prior_scale[2]]
    ode_problem = ODEProblem(rs, u0, (0.0, 3.0), ps)
    sol = solve(
        ode_problem, Rodas5P(); saveat = [1.0, 2.0, 3.0], abstol = 1e-8, reltol = 1e-8
    )
    diff = ((sol[:X1] - [1.1, 1.2, 1.3]) ./ 0.5).^2
    return -log(0.5) * 3 - 3 / 2.0 *log(2π) - 0.5*sum(diff)
end

function test_bijectors(p_est::Vector{PEtabParameter})::Nothing
    prob_test = _get_petab_problem(p_est)
    log_density = PEtabBayesLogDensity(prob_test)

    x_petab_scale = get_x(prob_test)
    x_prior_scale = PEtabBayes.to_prior_scale(x_petab_scale, log_density)
    x_inference_scale = log_density.inference_info.bijectors(x_prior_scale)

    # Test prior
    log_prior_ref = _prior_ref(x_prior_scale, p_est)
    log_prior_test = PEtabBayes._log_prior(x_inference_scale, log_density.inference_info)
    @test log_prior_ref ≈ log_prior_test atol = 1.0e-10

    # Test log-target
    log_correction = Bijectors.logabsdetjac(
        log_density.inference_info.inv_bijectors, x_inference_scale
    )
    log_target_ref = log_prior_ref + log_correction + _llh_ref(x_prior_scale)
    log_target_test = LogDensityProblems.logdensity(log_density, x_inference_scale)
    @test log_target_ref ≈ log_target_test atol = 1.0e-10

    # Test log-target gradient
    f_ref = x -> begin
        x_prior = log_density.inference_info.inv_bijectors(x)
        log_correction = Bijectors.logabsdetjac(
            log_density.inference_info.inv_bijectors, x
        )
        log_prior_ref = _prior_ref(x_prior, p_est)
        return log_prior_ref + log_correction + _llh_ref(x_prior)
    end
    grad_ref = FiniteDifferences.grad(central_fdm(5, 1), f_ref, x_inference_scale)[1]
    log_target_test, grad_test = LogDensityProblems.logdensity_and_gradient(
        log_density, x_inference_scale
    )
    @test log_target_ref ≈ log_target_test atol = 1.0e-6
    @test all(.≈(grad_ref, grad_test; atol = 1.0e-6))
    return nothing
end

# Unbounded priors on linear scale
p_est = [
    PEtabParameter(:k1; scale = :lin, prior = Normal(1.0, 1.0), value = 1.1),
    PEtabParameter(:k2; scale = :lin, prior = Normal(0.5, 3.0), value = 0.9)
]
test_bijectors(p_est)

# Bounded priors and parameters on linear scale
p_est = [
    PEtabParameter(:k1; scale = :lin, prior = Uniform(0.0, 2.0), value = 1.1),
    PEtabParameter(:k2; scale = :lin, prior = Gamma(1.0, 1.0), value = 0.9)
]
test_bijectors(p_est)

# Truncated priors
p_est = [
    PEtabParameter(:k1; scale = :lin, prior = truncated(Normal(1.0, 1.0), 0.0, 2.0), value = 1.1),
    PEtabParameter(:k2; scale = :lin, prior = truncated(Gamma(1.0, 1.0), 0.0, 10.0), value = 0.9)
]
test_bijectors(p_est)

# log and log10 scale
p_est = [
    PEtabParameter(:k1; scale = :log, prior = LogNormal(0.0, 1.0), value = 1.1),
    PEtabParameter(:k2; scale = :log10, prior = LogNormal(0.0, 1.0), value = 0.9)
]
test_bijectors(p_est)
