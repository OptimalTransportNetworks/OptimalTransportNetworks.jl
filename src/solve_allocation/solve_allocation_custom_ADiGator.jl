
using Ipopt
using ForwardDiff

function solve_allocation_custom_ADiGator(x0, auxdata, funcs, verbose=true)

    # Recover parameters
    graph = auxdata.graph
    param = auxdata.param

    if isempty(x0)
        x0 = [1e-6*ones(graph.J*param.N); zeros(graph.ndeg*param.N); sum(param.Lj)/(graph.J*param.N)*ones(graph.J*param.N)]
    end

    # Parametrize IPOPT
    funcs = Dict(
        :objective => (x) -> objective_custom(x, auxdata),
        :gradient => (x) -> ForwardDiff.gradient(x -> objective_custom(x, auxdata), x),
        :constraints => (x) -> constraints_custom(x, auxdata),
        :jacobian => (x) -> ForwardDiff.jacobian(x -> constraints_custom(x, auxdata), x),
        :hessian => (x, sigma, lambda) -> ForwardDiff.hessian(x -> objective_custom(x, auxdata), x)
    )

    # Options
    options = Dict(
        :lb => [1e-8*ones(graph.J*param.N); -Inf*ones(graph.ndeg*param.N); 1e-8*ones(graph.J*param.N)],
        :ub => [Inf*ones(graph.J*param.N); Inf*ones(graph.ndeg*param.N); Inf*ones(graph.J*param.N)],
        :cl => [-Inf*ones(graph.J*param.N); -Inf*ones(graph.J)],
        :cu => [zeros(graph.J*param.N); zeros(graph.J)]
    )

    if verbose
        options[:print_level] = 5
    else
        options[:print_level] = 0
    end

    # Run IPOPT
    x, info = ipopt(x0, funcs, options)

    # Return results
    flag = info
    results = recover_allocation(x, auxdata)
    results.Pjn = reshape(info.lambda[1:graph.J*param.N], graph.J, param.N)
    results.PCj = sum(results.Pjn.^(1-param.sigma), dims=2).^(1/(1-param.sigma))
    results.welfare = -funcs[:objective](x)

    return results, flag, x
end

function recover_allocation(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph

    # Recover parameters
    results = Dict()
    results[:Lj] = param.Lj

    # Recover populations
    results[:Ljn] = reshape(x[graph.J*param.N+graph.ndeg*param.N+1:end], graph.J, param.N)

    # Production
    results[:Yjn] = param.Zjn .* (results[:Ljn] .^ param.a)

    # Domestic absorption
    results[:Cjn] = reshape(x[1:graph.J*param.N], graph.J, param.N)

    # Total availability of tradeable good
    results[:Cj] = sum(results[:Cjn] .^ ((param.sigma-1)/param.sigma), dims=2) .^ (param.sigma/(param.sigma-1))

    # Consumption per capita
    results[:cj] = results[:Cj] ./ results[:Lj]
    results[:cj][results[:Lj] .== 0] = 0

    # Non-tradeable per capita
    results[:hj] = param.hj

    # Vector of welfare per location
    results[:uj] = param.u(results[:cj], results[:hj])

    # Recover flows Qin in dimension [Ndeg,N]
    results[:Qin] = reshape(x[graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N], graph.ndeg, param.N)

    # recover the flows Qjkn in dimension [J,J,N]
    results[:Qjkn] = zeros(graph.J, graph.J, param.N)
    id = 1
    for i in 1:graph.J
        for j in 1:length(graph.nodes[i].neighbors)
            if graph.nodes[i].neighbors[j] > i
                results[:Qjkn][i, graph.nodes[i].neighbors[j], :] = max.(results[:Qin][id, :], 0)
                results[:Qjkn][graph.nodes[i].neighbors[j], i, :] = max.(-results[:Qin][id, :], 0)
                id += 1
            end
        end
    end

    return results
end


# Please note that the `ipopt` function used in the MATLAB code is from the MATLAB interface to the IPOPT solver. In Julia, you can use the `Ipopt.jl` package which provides a similar interface to the IPOPT solver. The `ForwardDiff.jl` package is used for automatic differentiation in Julia. The `objective_custom` and `constraints_custom` functions are not defined in the provided MATLAB code, so you would need to define them in Julia as well.