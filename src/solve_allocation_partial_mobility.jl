
using Ipopt
using LinearAlgebra

function solve_allocation_partial_mobility(x0, auxdata, verbose=true)

    # Recover parameters
    graph = auxdata.graph
    param = auxdata.param
    A = auxdata.A

    # Check compatibility
    if any(sum(param.Zjn .> 0, dims=2) .> 1)
        error("This code only supports one good at most per location.")
    end

    if isempty(x0)
        x0 = [zeros(param.nregions); 1e-6*ones(graph.J*param.N); zeros(graph.ndeg*param.N); ones(graph.J)/graph.J]
    end

    # Parametrize Ipopt
    nlp = create_nlp(x0, auxdata, verbose)

    # Run Ipopt
    solver = IpoptSolver(print_level=verbose ? 5 : 0)
    result = solve(nlp, solver)

    # Return results
    flag = result.status
    x = result.sol
    results = recover_allocation(x, auxdata)
    results.Pjn = reshape(result.lambda[graph.J+1:graph.J+graph.J*param.N], graph.J, param.N)
    results.PCj = (sum(results.Pjn .^ (1-param.sigma), dims=2)) .^ (1/(1-param.sigma))

    return results, flag, x
end

function create_nlp(x0, auxdata, verbose)
    # Init functions
    funcs = Dict(
        :objective => (x) -> objective(x, auxdata),
        :gradient => (x) -> gradient(x, auxdata),
        :constraints => (x) -> constraints(x, auxdata),
        :jacobian => (x) -> jacobian(x, auxdata),
        :hessian => (x, sigma, lambda) -> hessian(x, auxdata, sigma, lambda)
    )

    # Options
    options = Dict(
        :lb => [-Inf*ones(param.nregions); -100*ones(graph.J*param.N); -Inf*ones(graph.ndeg*param.N); 1e-8*ones(graph.J)],
        :ub => [Inf*ones(param.nregions); Inf*ones(graph.J*param.N); Inf*ones(graph.ndeg*param.N); ones(graph.J)],
        :cl => [-Inf*ones(graph.J); -Inf*ones(graph.J*param.N); zeros(param.nregions)],
        :cu => [zeros(graph.J); 1e-3*ones(graph.J*param.N); zeros(param.nregions)]
    )

    return MathProgBase.NonlinearModel(funcs, x0, options)
end

# Define the remaining functions (objective, gradient, constraints, jacobian, hessian, recover_allocation) in a similar way.


# Please note that this is a rough translation and might need adjustments. The `MathProgBase` package used here is deprecated and has been replaced by `MathOptInterface`. However, `Ipopt.jl` has not yet been updated to use `MathOptInterface` and still uses `MathProgBase`. Also, the `NonlinearModel` function used here to create the nonlinear programming problem might need to be replaced by a different function depending on the exact requirements of the problem.