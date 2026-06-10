using SafeTestsets

@safetestset "Aqua Quality Check" begin
    include("aqua.jl")
end

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

@safetestset "Multipathfinder wrapper" begin
    include("multipathfinder.jl")
end

@safetestset "AdvancedVI optimize wrapper" begin
    include("advanced_vi.jl")
end
