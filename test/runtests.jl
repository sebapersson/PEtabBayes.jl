using SafeTestsets

@safetestset "Bijectors" begin
    include("bijectors.jl")
end

@safetestset "Bayesian inference" begin
    include("inference.jl")
end

@safetestset "Error throwing" begin
    include("throw.jl")
end

@safetestset "Show and Describe" begin
    include("show.jl")
end
