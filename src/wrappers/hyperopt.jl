import .Hyperopt

@params struct HyperoptAlg <: AbstractOptimizer
    sub_alg::AbstractOptimizer
end

"""
    HyperoptOptions: options performing starting point optimization using Hyperopt.jl

- `sub_options`: options for the sub-optimizer.
- `lb`: Lower bound of starting point, if don't specify it, the default value will be `nothing`, 
            then will end up be replaced by the lower bound of optimization problem.
- `ub`: Upper bound of starting point, same as above. 
- `searchspace_size::Integer`: How many potential starting points we generate.
- `iters::Integer`: Among all generated potential starting points, how many of them will be evaluated. 
- `sampler::Hyperopt.Sampler`: An instance of 'Hyperopt.Sampler', which decides search algorithm. 
- `ctol`: infeasibility tolerance for accepting a solution as feasible
- `keep_all`: if true, all the solutions of the sub-problems will be saved
"""
@params struct HyperoptOptions
    sub_options
    lb
    ub
    searchspace_size::Integer
    iters::Integer
    sampler::Hyperopt.Sampler
    ctol
    keep_all::Bool
end
function HyperoptOptions(;
    sub_options, lb = nothing, ub = nothing,
    iters = 40, searchspace_size = iters,
    sampler = RandomSampler(), ctol = 1e-5,
    keep_all = true,
)
    return HyperoptOptions(
        sub_options, lb, ub, searchspace_size,
        iters, sampler, ctol, keep_all
    )
end

@params struct HyperoptWorkspace <: Workspace
    sub_workspace::Workspace
    x0::AbstractVector
    options::HyperoptOptions
end

function Workspace(model::VecModel, alg::HyperoptAlg, x0::AbstractVector; options, kwargs...)
    sub_options = options.sub_options isa Function ? options.sub_options(1) : options.sub_options
    return HyperoptWorkspace(
        Workspace(
            model, alg.sub_alg, copy(x0);
            options = sub_options, kwargs...,
        ),
        x0,
        options,
    )
end

"""
    HyperoptResult

When using multiple x0 in [`optimize`](@ref), return this result, including following fields: 
- `minimum`: the optimal value among all searches.
- `minmizer`: the optimal solutions found.
- `results`: all the search results.
- `optimal_ind`: the index of the optimal solution in `results`.
"""
@params struct HyperoptResult <: AbstractResult
    minimum
    minimizer
    results
    optimal_ind
end

function optimize!(workspace::HyperoptWorkspace)
    @unpack options, x0, sub_workspace = workspace
    @unpack model = sub_workspace
    @unpack lb, ub, searchspace_size, ctol, keep_all = options
    @unpack sampler, iters = options

    lb = lb === nothing ? getmin(model) : lb
    ub = ub === nothing ? getmax(model) : ub
    @assert all(isfinite, lb) && all(isfinite, ub) throw("Please use finite bounds for the starting point search.")

    @info "Multistart algorithm started. "
    @assert searchspace_size >= 1 "searchspace_size must be a positive integer. "

    # Generate a search space
    _sampler = sampler isa Hyperopt.Hyperband ? sampler.inner : sampler
    if _sampler isa Hyperopt.GPSampler
        params, candidates = get_linrange_candidates(lb, ub, searchspace_size)
        objective = (i, x0...) -> begin
            if sampler isa Hyperopt.Hyperband
                @assert options.sub_options isa Function
                sub_options = options.sub_options(i)
                _sub_workspace = @set sub_workspace.options = sub_options
            else
                _sub_workspace = sub_workspace
            end
            _x0 = [x0...]
            reset!(_sub_workspace, _x0)
            return optimize!(_sub_workspace), _x0
        end
    else
        params, candidates = get_sobol_candidates(lb, ub, x0, searchspace_size)
        objective = (i, x0) -> begin
            if sampler isa Hyperopt.Hyperband
                @assert options.sub_options isa Function
                sub_options = options.sub_options(i)
                _sub_workspace = @set sub_workspace.options = sub_options
            else
                _sub_workspace = sub_workspace
            end
            reset!(_sub_workspace, x0)
            return optimize!(_sub_workspace), x0
        end
    end
    _ho = Hyperopt.Hyperoptimizer(
        iterations = iters, sampler = sampler,
        objective = objective, params = params,
        candidates = candidates,
    )
    ho = @set _ho.objective = (args...) -> begin
        if length(args) == 2 && sampler isa Hyperopt.Hyperband
            if _sampler isa Hyperopt.GPSampler
                res, _ = _ho.objective(args[1], args[2]...)
            else
                res, _ = _ho.objective(args[1], args[2])
            end
        else
            res, _ = _ho.objective(args...)
        end
        push!(_ho.results, res.minimum)
        if sampler isa Hyperopt.Hyperband
            return isfeasible(model, res.minimizer, ctol = ctol) ? res.minimum : Inf, res.minimizer
        else
            return res
        end
    end
    if _sampler isa Union{Hyperopt.LHSampler, Hyperopt.CLHSampler, Hyperopt.GPSampler}
        Hyperopt.init!(_sampler, ho)
    end
    if sampler isa Hyperopt.Hyperband
        optimal = Hyperopt.hyperband(ho)
        return HyperoptResult(optimal[1], optimal[3], nothing, nothing)
    else
        hor = map(ho) do (args)
            ho.objective(values(args)...)
        end
        ind = argmin(map(res -> (isfeasible(model, res.minimizer, ctol = ctol) ? res.minimum : Inf), hor))
        optimal = hor[ind]
        if keep_all
            return HyperoptResult(optimal.minimum, optimal.minimizer, identity.(hor), ind)
        else
            return HyperoptResult(optimal.minimum, optimal.minimizer, nothing, nothing)
        end
    end
end

function get_linrange_candidates(lb, ub, n)
    return Tuple(Symbol.(:x0_, 1:length(lb))), Tuple(LinRange.(lb, ub, n))
end

function get_sobol_candidates(lb, ub, x0, n)
    sobol_seq_generator = SobolSeq(lb, ub)
    return (:x0,), ([[x0]; collect(next!(sobol_seq_generator) for _ = 1:n-1)],)
end

function RandomSampler(args...; kwargs...)
    return Hyperopt.RandomSampler(args...; kwargs...)
end
function Hyperband(args...; kwargs...)
    return Hyperopt.Hyperband(args...; kwargs...)
end
function LHSampler(args...; kwargs...)
    return Hyperopt.LHSampler(args...; kwargs...)
end
function CLHSampler(dims = Hyperopt.Continuous())
    return Hyperopt.CLHSampler(dims = [dims])
end
function GPSampler(args...; kwargs...)
    return Hyperopt.GPSampler(Hyperopt.Min, args...; kwargs...)
end
