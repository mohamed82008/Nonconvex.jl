using Nonconvex, Test

include("functions/non_convex.jl")

@testset "test function sanity checks" begin
    for F in TEST_FUNCTIONS
        @test F(minimum_location(F, 10)) ≈ 0
    end
end

d_tol = Dict(SHIFTED_QUADRATIC=>1e-6, GRIEWANK=>1e-6, LEVY_MONTALVO2=>1e-6, RASTRIGIN=>1e-6, ROSENBROCK=>1e-1)
tol(F) = (d_tol[F])

test_dim = 2
UNEXPORTED_MTS_OPTIMIZATION_METHODS = setdiff(Nonconvex._MTS_OPTIMIZATION_METHODS, Nonconvex.MTS_OPTIMIZATION_METHODS)

@testset "MTS" begin
    # Temporary disable ROSENBROCK
    for F in setdiff(TEST_FUNCTIONS, (ROSENBROCK, ))
        println("Testing exported methods... ")
        for M in Nonconvex.MTS_OPTIMIZATION_METHODS
            println("Testing nonconvex function: ", F, " with method: ", M)
            LS1_options = MTSOptions(
                method = M
            )
            m = Model(x -> F(x))
            lb = [lu_bounds(F)[1] for _ in 1:test_dim]
            ub = [lu_bounds(F)[2] for _ in 1:test_dim]
            addvar!(m, lb, ub)
            alg = MTSAlg()
            r = Nonconvex.optimize(m, alg, options = LS1_options)
            println(r.minimizer)
            println(r.minimum)
            @test abs(r.minimum) < tol(F)
        end
    end
end

# Robust test, test unexported methods (localsearch2, localsearch3)
@testset "MTS-unexported" begin
    println("Testing unexported methods...")
    for F in TEST_FUNCTIONS
        for M in UNEXPORTED_MTS_OPTIMIZATION_METHODS
            println("Testing nonconvex function: ", F, " with method: ", M)
            LS1_options = MTSOptions(
                method = M
            )
            m = Model(x -> F(x))
            lb = [lu_bounds(F)[1] for _ in 1:test_dim]
            ub = [lu_bounds(F)[2] for _ in 1:test_dim]
            addvar!(m, lb, ub)
            alg = MTSAlg()
            r = Nonconvex.optimize(m, alg, options = LS1_options)
            println(r.minimizer)
            println(r.minimum)
        end
    end
end