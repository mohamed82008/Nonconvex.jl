import Hyperopt
using Documenter, Nonconvex

makedocs(
    sitename="Nonconvex.jl",
    pages = [
        "Getting started" => "index.md",
        "Problem definition" => "problem.md",
        "Algorithms" => [
            "Overview" => "algorithms/algorithms.md",
            "algorithms/mma.md",
            "algorithms/ipopt.md",
            "algorithms/nlopt.md",
            "algorithms/auglag.md",
            "algorithms/minlp.md",
            "algorithms/hyperopt.md",
            "algorithms/surrogate.md",
            "algorithms/mts.md",
        ],
        "Gradients, Jacobians and Hessians" => "gradients.md",
    ],
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/mohamed82008/Nonconvex.jl.git",
        push_preview=true,
    )
end
