# Multiple Trajectory Search (MTS)

## Description

MTS: Multiple Trajectory Search for Large Scale Global Optimization, is a derivative-free heuristic optimization method presented in paper (Lin-Yu Tseng and Chun Chen, 2008)[https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsCEC08/tseng08mts.pdf]. 
The main algorihtm `MTS` contains three subalgorithms `localsearch1`, `localsearch2` and `localsearch3`. This module implelents all the optimization methods in paper. People often use entire `MTS` or only `localsearch1` subalgorithm to optimize functions, and generally nobody use `localsearch2` or `localsearch3` independently. Therefore, the module only exports `MTS` and `localsearch1`.

## Quick start

Using default `MTSOptions()`. `MTS` is used for optimization. 

```julia
LS1_options = MTSOptions()
m = Model(f)
lb = [0, 0]
ub = [5, 5]
# Must have a box constraint. And (in)equality constraints are not supported for MTS methods.
addvar!(m, lb, ub)
result = optimize(model, alg, x0, options = options
```

## Options

You can choose which algorithm to use by spycifying `option.method`. Avaliable list is `[MTS (default), localsearch1, Nonconvex.localsearch2 (not recommend), Nonconvex.localsearch3 (not recommend)]`.

```julia
LS1_options = MTSOptions(method=localsearch1)
m = Model(f))
lb = [0, 0]
ub = [5, 5]
# Must have a box constraint. And (in)equality constraints are not supported in MTS methods.
addvar!(m, lb, ub)
result = optimize(model, alg, x0, options = options
```
