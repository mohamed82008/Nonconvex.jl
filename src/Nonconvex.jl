module Nonconvex

const debugging = Ref(false)
const show_residuals = Ref(false)

export  Model,
        DictModel,
        addvar!,
        set_objective!,
        add_ineq_constraint!,
        add_eq_constraint!,
        getmin,
        getmax,
        isinteger,
        setmin!,
        setmax!,
        setinteger!,
        optimize,
        localsearch1,
        MTS,
        Workspace,
        MMA87,
        MMA,
        MMA02,
        GCMMA,
        MMALag,
        IpoptAlg,
        DeflatedAlg,
        NLoptAlg,
        AugLag,
        PercivalAlg,
        JuniperIpoptAlg,
        PavitoIpoptCbcAlg,
        HyperoptAlg,
        BayesOptAlg,
        MTSAlg,
        KKTCriteria,
        IpoptCriteria,
        FunctionWrapper,
        MMAOptions,
        IpoptOptions,
        DeflatedOptions,
        NLoptOptions,
        AugLagOptions,
        PercivalOptions,
        JuniperIpoptOptions,
        PavitoIpoptCbcOptions,
        HyperoptOptions,
        BayesOptOptions,
        MTSOptions,
        Tolerance,
        @constructor,
        RandomSampler,
        Hyperband,
        LHSampler,
        CLHSampler,
        GPSampler

using Parameters, Zygote, ChainRulesCore, ForwardDiff
using Ipopt, ADNLPModels, NLPModelsModifiers
using Requires, SparseArrays, Reexport
using Cbc, Sobol, NamedTupleTools
@reexport using LinearAlgebra, OrderedCollections
using Optim: Optim, AbstractOptimizer
using Requires, Reexport, Setfield
import JuMP, MathOptInterface
const MOI = MathOptInterface
using JuMP: VariableRef, is_binary, is_integer, has_lower_bound,
            has_upper_bound, lower_bound, upper_bound,
            start_value, ConstraintRef, constraint_object,
            AffExpr, objective_function, objective_sense

# General

include("utilities/params.jl")
include("functions/functions.jl")
include("functions/value_jacobian.jl")
include("functions/function_docs.jl")
include("functions/counting_function.jl")
include("functions/aggregations.jl")
include("algorithms/common.jl")

# Models

include("models/flatten.jl")
include("models/model.jl")
include("models/vec_model.jl")
include("models/dict_model.jl")
include("models/model_docs.jl")
include("mma_approximation/mma_approx.jl")
include("mma_approximation/xmma_approx.jl")
include("mma_approximation/mma_approx_docs.jl")
include("models/mma_model.jl")
include("models/dual_model.jl")
include("models/mmalag_model.jl")
include("models/jump.jl")

# MMA

include("utilities/convergence.jl")
include("utilities/callbacks.jl")
include("utilities/options.jl")
include("algorithms/mma_algorithm.jl")

# MMA-AugLag2

include("algorithms/stoch_optimizers.jl")
include("algorithms/nonstoch_optimizers.jl")
include("algorithms/ammal.jl")

# Augmented Lagrangian

include("models/auglag_model.jl")
include("algorithms/auglag_algorithm.jl")

# Bayesian optimization

include("algorithms/bayesian.jl")

# Bayesian optimization

include("algorithms/mts.jl")

# Deflated algorithms

include("algorithms/deflation.jl")

# Wrappers

include("wrappers/moi.jl")
include("wrappers/ipopt.jl")
@init begin
    @require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" begin
        include("wrappers/nlopt.jl")
    end
    @require Percival="01435c0c-c90d-11e9-3788-63660f8fbccc" begin
        include("wrappers/percival.jl")
    end
    @require Juniper="2ddba703-00a4-53a7-87a5-e8b9971dde84" begin
        include("wrappers/juniper.jl")
    end
    @require Pavito="cd433a01-47d1-575d-afb7-6db927ee8d8f" begin
        include("wrappers/pavito.jl")
    end
    @require Hyperopt="93e5fe13-2215-51db-baaf-2e9a34fb2712" begin
        include("wrappers/hyperopt.jl")
    end
end

end
