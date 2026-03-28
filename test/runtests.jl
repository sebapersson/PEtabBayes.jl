using SafeTestsets, PEtabBayes

@safetestset "Bijectors" begin
    include("bijectors.jl")
end

@safetestset "Bayesian Inference" begin
    include("inference.jl")
end
