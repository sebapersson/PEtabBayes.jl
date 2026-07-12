#=
    Orchestration for sampling multiple MCMC chains, optionally in parallel.

    Per-chain sampling is delegated to the `_sample` methods defined in the sampler
    extensions (AdvancedHMC, AdaptiveMCMC, ...).
=#

function _sample_chains(
        ::SerialSampling, log_target::PEtabBayesLogDensity, alg, n_samples::Integer,
        x0::AbstractVector; kwargs...
    )::Chains
    chains = map(x0) do x0_chain
        _sample(log_target, alg, n_samples, x0_chain; kwargs...)
    end
    return _combine_chains(chains)
end

function _sample_chains(
        ::ThreadedSampling, log_target::PEtabBayesLogDensity, alg, n_samples::Integer,
        x0::AbstractVector; kwargs...
    )::Chains
    if Threads.nthreads() == 1
        @warn "Sampling with `parallel = ThreadedSampling()`, but Julia is running with a \
            single thread (`Threads.nthreads() == 1`); the chains will be sampled \
            sequentially. Start Julia with multiple threads (e.g. `julia --threads=auto`) \
            to sample the chains in parallel." maxlog = 1
    end

    n_chains = length(x0)
    chains = Vector{Chains}(undef, n_chains)
    Threads.@threads for i in 1:n_chains
        log_target_i = deepcopy(log_target)
        chains[i] = _sample(log_target_i, alg, n_samples, x0[i]; kwargs...)
    end
    return _combine_chains(chains)
end

# Distributed: run the chains on `nprocs` worker processes via `pmap`, mirroring how
# `PEtab.calibrate_multistart` parallelizes. Following PEtab.jl, `nprocs == 1` adds no
# workers and runs the chains on the calling process.
function _sample_chains(
        backend::DistributedSampling, log_target::PEtabBayesLogDensity, alg,
        n_samples::Integer, x0::AbstractVector; kwargs...
    )::Chains
    _nprocs = backend.nprocs == 1 ? 0 : backend.nprocs
    pids = Distributed.addprocs(_nprocs)
    chains = try
        _load_packages_workers(pids, alg)
        sample_chain = let log_target = log_target, alg = alg, n_samples = n_samples,
                kwargs = kwargs
            x0_chain -> PEtabBayes._sample(log_target, alg, n_samples, x0_chain; kwargs...)
        end
        Distributed.pmap(sample_chain, x0)
    finally
        Distributed.rmprocs(pids)
    end
    return _combine_chains(chains)
end

# Load PEtabBayes and the sampler package that defines `alg` on the worker processes, so
# that the correct `_sample` extension method is available there.
function _load_packages_workers(pids::Vector{Int}, alg)::Nothing
    isempty(pids) && return nothing
    @eval Distributed.@everywhere $pids Base.eval(Main, :(using PEtabBayes))
    sampler_pkg = nameof(Base.moduleroot(parentmodule(typeof(alg))))
    loaded_packages = Set(Symbol(m.name) for m in keys(Base.loaded_modules))
    if sampler_pkg in loaded_packages
        load_expr = :(using $(sampler_pkg))
        @eval Distributed.@everywhere $pids Base.eval(Main, $load_expr)
    end
    return nothing
end

function _combine_chains(chains::AbstractVector)::Chains
    length(chains) == 1 && return only(chains)
    combined = reduce(chainscat, chains)

    # `chainscat` keeps the first chain's `info`; rebuild the relevant entries so the
    # reported wall-clock time spans the whole run and the posterior source is preserved.
    infos = [chain.info for chain in chains]
    info = NamedTuple()
    if all(haskey(i, :start_time) for i in infos)
        info = merge(info, (start_time = minimum(i.start_time for i in infos),))
    end
    if all(haskey(i, :stop_time) for i in infos)
        info = merge(info, (stop_time = maximum(i.stop_time for i in infos),))
    end
    if all(haskey(i, :source) for i in infos)
        info = merge(info, (source = first(infos).source,))
    end
    return isempty(info) ? combined : setinfo(combined, info)
end
