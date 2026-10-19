using PEtabBayes, FlexiChains, Dates, Test

@testset "FlexiChains conversion and combination" begin
    draws = reshape(collect(1.0:24.0), 4, 3, 2)
    ids = [:a, :b, :c]
    chain = PEtabBayes._to_chain(
        draws, ids; source = :prior, start_time = DateTime(2026),
        end_time = DateTime(2026) + Second(2)
    )
    @test chain isa SymChain
    @test FlexiChains.parameters(chain) == ids
    @test Array(chain) == permutedims(draws, (1, 3, 2))
    # Predictive checks pool iterations across chains and exclude provenance extras.
    expected = reshape(permutedims(draws, (1, 3, 2)), 8, 3)
    @test PEtabBayes._get_samples(chain, nothing) == expected
    @test PEtabBayes._get_samples(chain, 3) == expected[[1, 4, 8], :]
    @test PEtabBayes._infer_source(chain) == :prior
    @test FlexiChains.sampling_time(chain) == [2.0, 2.0]

    combined = PEtabBayes._combine_chains([chain, chain])
    @test size(combined) == (4, 4)
    @test Array(combined) == cat(Array(chain), Array(chain); dims = 2)
    @test PEtabBayes._infer_source(combined) == :prior
    @test FlexiChains.sampling_time(combined) == fill(2.0, 4)
    @test PEtabBayes._combine_chains([chain]) === chain

    posterior = PEtabBayes._to_chain(draws, ids)
    @test PEtabBayes._infer_source(posterior) == :posterior
    @test all(ismissing, FlexiChains.sampling_time(posterior))
    external = SymChain(permutedims(draws, (1, 3, 2)), Tuple(FlexiChains.Parameter.(ids)))
    @test PEtabBayes._infer_source(external) == :posterior
end
