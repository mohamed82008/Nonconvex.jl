######################################################
# Common objects and methods used for all algorithms
######################################################

"""
    Workspace

Abstract Workspace, which is a struct that stores states and information needed for optimization process.
Implementation depends on specific algorithm.
"""
abstract type Workspace end
function reset!(w::Workspace, x0 = nothing)
    if x0 !== nothing
        w.x0 .= x0
    end
    return w
end

"""
    ConvergenceState

A struct that summarizes the convergence state of a solution. The fields in this struct are:
 - `Δx`: the infinity norm of the change in the solution `x`.
 - `Δf`: the change in the objective value `f`.
 - `relΔf`: the ratio of change in the objective value `f`.
 - `kkt_residual`: the Karush-Kuhn-Tucker (KKT) residual of the solution. If [`ScaledKKTCriteria`](@ref) is used instead of [`KKTCriteria`](@ref), the `kkt_residual` will be divided by a factor.
 - `ipopt_residual`: the modified KKT residual used in the IPOPT solver.
 - `infeas`: maximum infeasibility amount. This is 0 if the solution is feasible.
 - `x_converged`: true if `Δx` is less than the `x` tolerance in [`MMAOptions`](@ref).
 - `fabs_converged`: true if `Δf` is less than the `f` tolerance in [`MMAOptions`](@ref).
 - `frel_converged`: true if `relΔf` is less than the `f` tolerance in [`MMAOptions`](@ref).
 - `kkt_converged`: true if the `kkt_residual` is less than the KKT tolerance in [`MMAOptions`](@ref).
 - `ipopt_converged`: true if the `ipopt_residual` is less than the KKT tolerance in [`MMAOptions`](@ref).
 - `infeas_converged`: true if `infeas` is less than the infeasibility tolerance in [`MMAOptions`](@ref).
 - `f_increased`: true if the objective value of the current solution is higher than that of the previous solution.
 - `converged`: true if the solution satisfies the convergence criteria of choice. See [`ConvergenceCriteria`](@ref) for more on the different convergence criteria available.
"""
@with_kw mutable struct ConvergenceState{T}
    Δx::T = Inf
    Δf::T = Inf
    relΔf::T = Inf
    kkt_residual::T = Inf
    ipopt_residual::T = Inf
    infeas::T = Inf
    x_converged::Bool = false
    fabs_converged::Bool = false
    frel_converged::Bool = false
    kkt_converged::Bool = false
    ipopt_converged::Bool = false
    infeas_converged::Bool = false
    f_increased::Bool = true
    converged::Bool = false
end
ConvergenceState(::Type{T}) where {T} = ConvergenceState{T}()

"""
    Tolerance

A struct specifying the different tolerances used to assess the convergence of the algorithms. The following are the fields of `Tolerance`:
 - `x`: the tolerance for `Δx` in [`ConvergenceState`](@ref). `x_converged` will be true if `Δx` is less than the `x` tolerance in `Tolerance`. This is used to assess convergence when the [`GenericCriteria`](@ref) is used as the convergence criteria. 
 - `fabs`: the tolerance for `Δf` in [`ConvergenceState`](@ref). `f_converged` will be true if `Δf` is less than the `fabs` tolerance in `Tolerance`. This is used to assess convergence when the [`GenericCriteria`](@ref) is used as the convergence criteria.
 - `frel`: the tolerance for `relΔf` in [`ConvergenceState`](@ref). `f_converged` will be true if `relΔf` is less than the `frel` tolerance in `Tolerance`. This is used to assess convergence when the [`GenericCriteria`](@ref) is used as the convergence criteria.
 - `kkt`: the KKT tolerance. `kkt_converged` in [`ConvergenceState`](@ref) will be true if the `kkt_residual` is less than the KKT tolerance. And `ipopt_converged` in [`ConvergenceState`](@ref) will be true if `ipopt_residual` is less than the KKT tolerance. This is used to assess convergence when the [`KKTCriteria`](@ref), the [`ScaledKKTCriteria`](@ref) or [`IpoptCriteria`](@ref) criteria is used as the convergence criteria.
 - `infeas`: the maximum infeasibility tolerance. `infeas_converged` in [`ConvergenceState`](@ref) will be true if the maximum infeasibility is less than the infeasibility tolerance. This is used to assess convergence regardless of the convergence criteria used.

For more on convergence criteria, see [`GenericCriteria`](@ref), [`KKTCriteria`](@ref), [`ScaledKKTCriteria`](@ref) and [`IpoptCriteria`](@ref).
"""
struct Tolerance{Tx, Tf, Tkkt, Tinfeas}
    x::Tx
    fabs::Tf
    frel::Tf
    kkt::Tkkt
    infeas::Tinfeas
end
function Tolerance(;
    x = 0.0, f = 1e-5, fabs = f, frel = f,
    kkt = 1e-5, infeas = 1e-5,
)
    Tolerance(x, fabs, frel, kkt, infeas)
end
function (tol::Tolerance{<:Function, <:Function, <:Function})(i)
    return Tolerance(tol.x(i), tol.fabs(i), tol.frel(i), tol.kkt(i), tol.infeas)
end

function Base.:*(t::Tolerance, m::Real)
    return Tolerance(t.x * m, t.fabs * m, t.frel * m, t.kkt * m, t.infeas * m)
end

"""
    ConvergenceCriteria

This an abstract type with 4 subtypes:
1. [`GenericCriteria`](@ref)
2. [`KKTCriteria`](@ref)
3. [`ScaledKKTCriteria`](@ref)
4. [`IpoptCriteria`](@ref)
"""
abstract type ConvergenceCriteria end

"""
    GenericCriteria

This is a generic convergence criteria that uses:
1. The maximum change in the solution, `Δx`,
2. The change in the objective value, `Δf`, and
3. The change percentage in the objective value, `Δf`, and
4. The maximum infeasibility `infeas`.
to assess convergence. More details are given in [`assess_convergence!`](@ref).
"""
struct GenericCriteria <: ConvergenceCriteria end

"""
    KKTCriteria

This convergence criteria uses the Karush-Kuhn-Tucker residual and maximum infeasibility to assess convergence. More details are given in [`assess_convergence!`](@ref).
"""
struct KKTCriteria <: ConvergenceCriteria end

"""
    IpoptCriteria

This convergence criteria uses a scaled version of the Karush-Kuhn-Tucker (KKT) residual and maximum infeasibility to assess convergence. This scaled KKT residual is used in the IPOPT nonlinear programming solver as explained in [this paper](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf). More details are given in [`assess_convergence!`](@ref). 
"""
struct IpoptCriteria <: ConvergenceCriteria end

"""
    ScaledKKTCriteria

This convergence criteria uses another scaled version of the Karush-Kuhn-Tucker (KKT) residual and maximum infeasibility to assess convergence. In particular if the objective was scaled by a factor `m`, the KKT residual will be scaled down by a factor `max(m, 1/m)`. This scaling was found to make the convergence criteria less sensitive to scale compared to using the traditional KKT residual. More details are given in [`assess_convergence!`](@ref). 
"""
struct ScaledKKTCriteria <: ConvergenceCriteria end


"""
    Solution

A struct that stores all the information about a solution. The following are the fields of `Solution`:
 - `x`: the current primal solution
 - `prevx`: the previous primal solution
 - `λ`: the current dual solution
 - `f`: the objective value of `x`
 - `prevf`: the obejctive value of `prevx`
 - `∇f`: the gradient of the objective at `x`
 - `g`: the constraint function value at `x`
 - `∇g`: the Jacobian of the constraint functions at `x`
 - `convstate`: the convergence state of the solution
"""
@params mutable struct Solution
    prevx
    x
    λ
    prevf
    f
    ∇f
    g
    ∇g
    convstate::ConvergenceState
end

"""
    Solution(dualmodel, λ)

Construct an empty solution for the dual model `dualmodel` given a sample dual solution `λ`.
"""
function Solution(dualmodel, λ)
    prevx = copy(getxk(dualmodel))
    x = copy(prevx)
    λ = copy(λ)
    prevf = Inf
    fg = getfk(dualmodel)
    ∇fg = get∇fk(dualmodel)
    f = fg[1]
    g = fg[2:end]
    ∇f = ∇fg[1, :]
    ∇g = ∇fg[2:end, :]
    convstate = ConvergenceState()
    return Solution(prevx, x, λ, prevf, f, ∇f, g, ∇g, convstate)
end

"""  
    AbstractResult

An abstract type that stores optimization result.
"""
abstract type AbstractResult end

"""
    GenericResult

A summary result struct returned by [`optimize`](@ref), including following fields:
 - `optimizer`: the optimization algorithm used
 - `initial_x`: the initial primal solution
 - `minimizer`: the best solution found
 - `minimum`: the optimal value
 - `iter`: number of inner iterations run
 - `maxiter_reached`: true if the algorithm stopped due to reaching the maximum number of iterations
 - `tol`: an instance of [`Tolerance`](@ref) that specifies the convergence tolerances
 - `convstate`: an instance of [`ConvergenceCriteria`](@ref) that summarizes the convergence state of the best solution found
 - `fcalls`: the number of times the objective and constraint functions were called during the optimization
"""
@params mutable struct GenericResult <: AbstractResult
    optimizer
    initial_x
    minimizer
    minimum::Real
    iter::Int
    maxiter_reached::Bool
    tol::Tolerance
    convstate::Union{ConvergenceState, Nothing}
    fcalls::Int
end

abstract type AbstractModel end

"""
```
optimize(
    model::AbstractModel,
    optimizer::AbstractOptimizer = MMA02(),
    x0::AbstractVector;
    options,
    convcriteria::ConvergenceCriteria = KKTCriteria(),
    plot_trace::Bool = false,
    callback::Function = plot_trace ? LazyPlottingCallback() : NoCallback(),
    kwargs...,
)
```

Optimizes `model` using the algorithm `optimizer`, e.g. an instance of [`MMA87`](@ref) or [`MMA02`](@ref). `x0` is the initial solution. The keyword arguments are:
 - `options`: used to set the optimization options. It is an instance of [`MMAOptions`](@ref) for [`MMA87`](@ref) and [`MMA02`](@ref).
 - `convcriteria`: an instance of [`ConvergenceCriteria`](@ref) that specifies the convergence criteria of the MMA algorithm.
 - `plot_trace`: a Boolean that if true specifies the callback to be an instance of [`PlottingCallback`](@ref) and plots a live trace of the last 50 solutions.
 - `callback`: a function that is called on `solution` in every iteration of the algorithm. This can be used to store information about the optimization process.

 The details of the MMA optimization algorithms can be found in the original [1987 MMA paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620240207) and the [2002 paper](https://epubs.siam.org/doi/abs/10.1137/S1052623499362822).
"""
function optimize(model::AbstractModel, optimizer::AbstractOptimizer, x0, args...; kwargs...)
    _model, _x0, unflatten = tovecmodel(model, x0)
    r = optimize(_model, optimizer, _x0, args...; kwargs...)
    return @set r.minimizer = unflatten(r.minimizer)
end

"""
 optimize without x0
"""
 function optimize(model::AbstractModel, optimizer::AbstractOptimizer, args...; kwargs...)
    _model, _, unflatten = tovecmodel(model)
    r = optimize(_model, optimizer, args...; kwargs...)
    return @set r.minimizer = unflatten(r.minimizer)
end

"""
Clamp the box constraint and evaluate objective function at point x
"""
function evaluate!(model::AbstractModel, x::AbstractVector)
    @unpack box_min, box_max = model
    x .= clamp.(x, box_min, box_max)
    model.objective(x)
end

"""
Evaluate without truncate
"""
function evaluate(model::AbstractModel, x::AbstractVector)
    model.objective(x)
end