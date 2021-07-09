using SafeTestsets, Test

@safetestset "Model" begin include("model.jl") end
@safetestset "DictModel" begin include("dict_model.jl") end
@safetestset "JuMP" begin include("jump.jl") end
@testset "MMA" begin
    @safetestset "Approximation" begin include("approximation.jl") end
    @safetestset "Algorithm" begin include("mma.jl") end
end
@safetestset "AugLag/Percival" begin include("percival.jl") end
@safetestset "AugLag2" begin include("auglag.jl") end
@safetestset "Ipopt" begin include("ipopt.jl") end
@safetestset "NLopt" begin include("nlopt.jl") end
@safetestset "Juniper" begin include("juniper.jl") end
@safetestset "Pavito" begin include("pavito.jl") end
@safetestset "Hyperopt" begin include("hyperopt.jl") end
@safetestset "Bayesian optimization" begin include("bayesian.jl") end
@safetestset "Multiple trajectory search" begin include("mts.jl") end
@safetestset "Deflation" begin include("deflation.jl") end
