
# MTS Algorithms
# MTS: Multiple Trajectory Search for Large Scale Global Optimization, 
# By [Lin-Yu Tseng and Chun Chen, 2008](https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsCEC08/tseng08mts.pdf)

using Random: randperm

@with_kw struct MTSOptions
    method::Function = MTS
    M = 100
    maxiter=200
    search_range_tol=1e-15
    n_foreground = 80
    n_local_search = 200
    n_local_search_test = 3
    n_local_search_best = 300
    BONUS1 = 10
    BONUS2 = 1 
    a_min = 0.4
    a_max = 0.5
    b_min = 0.1
    b_max = 0.3
    c_min = 0
    c_max = 1
    function MTSOptions(args...; kwargs...)
        @assert (length(args) > 0 && args[1] in _MTS_OPTIMIZATION_METHODS) || kwargs[:method] in _MTS_OPTIMIZATION_METHODS
        new(args...; kwargs...)
    end
end

@params mutable struct MTSWorkspace <: Workspace
    model::VecModel
    x0::AbstractVector
    x::AbstractVector
    options::MTSOptions
    enable::AbstractVector
    improve::AbstractVector
    # Volitale variables
    search_range::AbstractVector
    optimal_x::AbstractVector
    optimal_ind::Int
    optimal_val::Real
end
function MTSWorkspace(model::VecModel, x0::AbstractVector, options::MTSOptions; kwargs...)
    @unpack box_min, box_max = model
    M = options.M
    # Initialize improve and serch range
    enable = [true for _ in 1:M]
    improve = [true for _ in 1:M]
    search_range = [(box_max-box_min) ./ 2 for _ in 1:M]
    MTSWorkspace(model, x0, copy(x0), options, enable, improve, search_range, x0[1], -1, Inf)
end

if debugging[]
    function Base.setproperty!(workspace::MTSWorkspace, name::Symbol, x)
        if name in [:optimal_x, :optimal_ind, :optimal_val]
            println("New ",  name, ": ", x)
        end
        setfield!(workspace, name, x)
    end
end

@params struct MTSResult <: AbstractResult
    minimum
    minimizer
end

struct MTSAlg <: AbstractOptimizer end

function reduce_search_range(search_range, k, n_dim, box_min, box_max, search_range_tol)
    search_range[k] ./= 2
    for i in 1:n_dim
        if search_range[k][i] < search_range_tol
            search_range[k][i] = (box_max[i] - box_min[i]) * 0.4
        end
    end
end

function _localsearch1(workspace, k)
    @unpack model, options = workspace
    @unpack x, improve, search_range = workspace
    @unpack BONUS1, BONUS2, search_range_tol = options
    @unpack box_min, box_max, n_dim = model
    grade = 0
    # search_range in paper is one-dimensional. Expand it to multidimensional. 
    if improve[k] == false
        reduce_search_range(search_range, k, n_dim, box_min, box_max, search_range_tol)
    end
    improve[k] = false
    for i in 1:n_dim
        # Original value
        _xk = copy(x[k])
        xk_val = evaluate!(model, _xk)
        update_xki = true
        _xk[i] -= search_range[k][i]
        # Value after update
        _xk_val = evaluate!(model, _xk)
        # Better than current best solution
        if _xk_val < workspace.optimal_val
            grade += BONUS1
            workspace.optimal_x, workspace.optimal_ind, workspace.optimal_val = copy(_xk), k, _xk_val
        end
        # Value stays the same
        if _xk_val == xk_val
            # Restore 
            update_xki = false
        else
            # Value degenerates
            if _xk_val > xk_val
                # Restore x_k
                _xk = copy(x[k])
                _xk[i] += 0.5 * search_range[k][i]
                # Current value
                _xk_val = evaluate!(model, _xk)
                if _xk_val < workspace.optimal_val
                    grade += BONUS1
                    workspace.optimal_x, workspace.optimal_ind, workspace.optimal_val = copy(_xk), k, _xk_val
                end
                if _xk_val >= xk_val
                    # Restore
                    update_xki = false
                else
                    grade += BONUS2
                    improve[k] = true
                end
            else
                grade += BONUS2
                improve[k] = true
            end
        end
        if update_xki
            # if debugging[]
            #     println("Updating xki, original xk: ", x[k], " new xk: ", _xk)
            #     @assert evaluate(model, x[k]) > evaluate(model, _xk)
            # end
            x[k][i] = _xk[i]
        end
    end
    return grade
end

function _localsearch2(workspace, k)
    @unpack model, options = workspace
    @unpack x, improve, search_range = workspace
    @unpack BONUS1, BONUS2 = options
    @unpack n_dim = model
    grade = 0
    improve[k] = false
    for i in 1:n_dim
        # Original value
        xk_val = evaluate!(model, x[k])
        update_xk = true
        D = [rand([-1, 1]) for _ in 1:n_dim]
        r = [rand([0, 1, 2, 3]) for _ in 1:n_dim]
        # Value after update
        _xk = copy(x[k])
        _xk .= [r[_i] == 0 ? _xk[_i]-search_range[k][i]*D[_i] : _xk[_i] for _i in 1:length(r)]
        _xk_val = evaluate!(model, _xk)
        if _xk_val < workspace.optimal_val
            grade += BONUS1
            workspace.optimal_x, workspace.optimal_ind, workspace.optimal_val = copy(_xk), k, _xk_val
        end
        # Value stays the same
        if _xk_val == xk_val
            # Restore 
            update_xk = false
        else
            # Value degenerates
            if _xk_val > xk_val
                # Restore x_k
                _xk = copy(x[k])
                _xk .= [r[_i] == 0 ? _xk[_i]+0.5*search_range[k][i]*D[_i] : _xk[_i] for _i in 1:length(r)]
                _xk_val = evaluate!(model, _xk)
                if _xk_val < workspace.optimal_val
                    grade += BONUS1
                    workspace.optimal_x, workspace.optimal_ind, workspace.optimal_val = copy(_xk), k, _xk_val
                end
                if _xk_val >= xk_val
                    update_xk = false
                else
                    grade += BONUS2
                    improve[k] = true
                end
            else
                grade += BONUS2
                improve[k] = true
            end
        end
        if update_xk
            x[k] = _xk
        end
    end
    return grade
end

function _localsearch3(workspace, k)
    @unpack model, options = workspace
    @unpack x = workspace
    @unpack BONUS1, BONUS2 = options 
    @unpack a_min, a_max, b_min, b_max, c_min, c_max = options
    grade = 0
    _xk = copy(x[k])
    update_xk = false
    for i in 1:model.n_dim
        # Original value
        _xk_val = evaluate!(model, _xk)
        update_xk = true
        # Heuristic search
        _xk_x1, _xk_y1, _xk_x2  = copy(_xk), copy(_xk), copy(_xk)
        _xk_x1[i] += 0.1
        _xk_y1[i] -= 0.1
        _xk_x2[i] += 0.2
        _xk_x1_val, _xk_y1_val, _xk_x2_val = evaluate!(model, _xk_x1), evaluate!(model, _xk_y1), evaluate!(model, _xk_x2)
        if _xk_x1_val < workspace.optimal_val
            grade += BONUS1
            workspace.optimal_x, workspace.optimal_ind, workspace.optimal_val = copy(_xk), k, _xk_val
        end
        if _xk_y1_val < workspace.optimal_val
            grade += BONUS1
            workspace.optimal_x, workspace.optimal_ind, workspace.optimal_val = copy(_xk), k, _xk_val
        end
        if _xk_x2_val < workspace.optimal_val
            grade += BONUS1
            workspace.optimal_x, workspace.optimal_ind, workspace.optimal_val = copy(_xk), k, _xk_val
        end
        D1, D2, D3 = _xk_val - _xk_x1_val, _xk_val - _xk_y1_val, _xk_val - _xk_x2_val
        grade += ((D1>0) + (D2>0) + (D3>0)) * BONUS2
        if update_xk == false
            update_xk = (D1>0) || (D2>0) || (D3>0)
        end
        a = (rand() * (a_max - a_min)) + a_min
        b = (rand() * (b_max - b_min)) + b_min
        c = (rand() * (c_max - c_max)) + c_min
        _xk[i] += a*(D1 - D2) + b*(D3 - 2*D1) + c
    end
    if update_xk
        # Update x[k]
        x[k] = _xk
    end
    return grade
end

# Optimization using localsearch1
function localsearch1(workspace::MTSWorkspace)
    M = workspace.options.M
    for i in 1:M
        _localsearch1(workspace, i)
    end
end

# Optimization using localsearch2
function localsearch2(workspace::MTSWorkspace)
    M = workspace.options.M
    for i in 1:M
        _localsearch2(workspace, i)
    end
end

# Optimization using localsearch3
function localsearch3(workspace::MTSWorkspace)
    M = workspace.options.M
    for i in 1:M
        _localsearch3(workspace, i)
    end
end

const LOCAL_SEARCH_METHODS = [_localsearch1, _localsearch2, _localsearch3]
const N_METHODS = length(LOCAL_SEARCH_METHODS)

function MTS(workspace::MTSWorkspace)
    # Multiple Trajectory Search
    @unpack options, enable = workspace
    @unpack M, n_foreground, n_local_search, n_local_search_test, n_local_search_best = options
    grade_x = [-Inf for _ in 1:M]
    for i in 1:M
        if !enable[i]
            continue
        end
        LS_testgrades = [0 for _ in 1:N_METHODS]
        for _ in 1:n_local_search_test
            LS_testgrades = [LS_testgrades[m]+_local_search(workspace, i) for (m, _local_search) in enumerate(LOCAL_SEARCH_METHODS)]
        end
        _best_local_search = LOCAL_SEARCH_METHODS[argmax(LS_testgrades)]
        grade_x[i] = 0
        for _ in 1:n_local_search
            grade_x[i] += _best_local_search(workspace, i)
        end
    end
    for _ in 1:n_local_search_best
        _localsearch1(workspace, workspace.optimal_ind)
    end
    enable_ind = reverse(sortperm(grade_x))[begin:n_foreground]
    enable[:] .= false
    enable[enable_ind] .= true
end

function optimize!(workspace::MTSWorkspace)
    options = workspace.options
    println("Optimize using method: ", options.method)
    for iter in 1:options.maxiter
        if debugging[] && iter % 50 == 0
                println("Iter ", iter, " max iter: ", options.maxiter)
        end
        options.method(workspace)
    end
    MTSResult(workspace.optimal_val, workspace.optimal_x) 
end

const _MTS_OPTIMIZATION_METHODS = [localsearch1, localsearch2, localsearch3, MTS]
const MTS_OPTIMIZATION_METHODS = [localsearch1, MTS]

# Build a SOA (Simultaed Orthogonal Array). Refers to section 2 of the paper posted above. 
function build_SOA(m, k, q)
    SOA = Array{Int,2}(undef, m, k)
    for c in 1:k
        q_perm = randperm(q) .- 1
        p = 0
        for r in randperm(m)
            p = (p%q)+1
            SOA[r, c] = q_perm[p]
        end
    end
    SOA
end

function initialize_x(model::VecModel, options::MTSOptions)
    @unpack n_dim, box_min, box_max = model
    @unpack M = options
    SOA = build_SOA(M, n_dim, M)
    # Initialize x0
    x0 = Array{Real,2}(undef, M, n_dim)
    for i in 1:M
        for j in 1:n_dim
            # To be confirmed: At the 5th line of "Multi Trajectory Search" algorithm in paper, I do think (u_i, l_i) shoule be (u_j, l_j)
            x0[i, j] = box_min[j] + (box_max[j]-box_min[j])*(SOA[i, j]/(M-1))
        end
    end
    [x0[i, :] for i in 1:size(x0, 1)]
end

# Workspace constructor with x0
function Workspace(model::VecModel, optimizer::MTSAlg, x0::AbstractVector; options::MTSOptions=MTSOptions(), kwargs...,)
    @assert length(x0) > 0 && x0[1] isa AbstractVector
    if length(model.ineq_constraints) > 0 || length(model.eq_constraints) > 0
        @warn "MTS does not support (in)equality constraints. Your input would be ignored. "
    end
    return MTSWorkspace(model, x0, options)
end

# Workspace constructor without x0 (use method in paper to initialize)
function Workspace(model::VecModel, optimizer::MTSAlg; options::MTSOptions=MTSOptions(), kwargs...)
    x0 = initialize_x(model, options)
    return Workspace(model, optimizer, x0; options=options)
end
