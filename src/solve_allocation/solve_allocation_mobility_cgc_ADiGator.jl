
using Ipopt
using ForwardDiff

function solve_allocation_mobility_cgc_ADiGator(x0, auxdata, funcs, verbose=true)
    # extract parameters
    graph = auxdata.graph
    param = auxdata.param

    if isempty(x0)
        x0 = vcat(0, 1e-6*ones(graph.J*param.N), zeros(2*graph.ndeg*param.N), sum(param.Lj)/graph.J*ones(graph.J), 1e-6*ones(graph.J), sum(param.Lj)/(graph.J*param.N)*ones(graph.J*param.N))
    end

    # Update functions
    funcs = Dict(
        :objective => (x) -> objective_mobility_cgc(x, auxdata),
        :gradient => (x) -> ForwardDiff.gradient(funcs[:objective], x),
        :constraints => (x) -> constraints_mobility_cgc(x, auxdata),
        :jacobian => (x) -> ForwardDiff.jacobian(funcs[:constraints], x),
        :hessian => (x, sigma, lambda) -> ForwardDiff.hessian(funcs[:objective], x)
    )

    # Options
    options = Dict(
        :lb => vcat(-Inf, 1e-8*ones(graph.J*param.N), 1e-8*ones(2*graph.ndeg*param.N), 1e-8*ones(graph.J), 1e-8*ones(graph.J), 1e-8*ones(graph.J*param.N)),
        :ub => vcat(Inf, Inf*ones(graph.J*param.N), Inf*ones(2*graph.ndeg*param.N), sum(param.Lj)*ones(graph.J), Inf*ones(graph.J), Inf*ones(graph.J*param.N)),
        :cl => vcat(-Inf*ones(graph.J), -Inf*ones(graph.J*param.N), -1e-8, -Inf*ones(graph.J), -1e-8*ones(graph.J)),
        :cu => vcat(-1e-8*ones(graph.J), -1e-8*ones(graph.J*param.N), 1e-8, -1e-8*ones(graph.J), 1e-8*ones(graph.J))
    )

    if verbose
        options[:print_level] = 5
    else
        options[:print_level] = 0
    end

    # Run Ipopt
    x, info = Ipopt.solve(x0, funcs, options)

    # Return results
    flag = info

    results = recover_allocation(param, graph, x)

    results.Pjn = reshape(info.lambda[graph.J+1:graph.J+graph.J*param.N], graph.J, param.N) # Price vector
    results.PCj = info.lambda[graph.J+graph.J*param.N+2:graph.J+graph.J*param.N+1+graph.J] # Price of tradeable bundle

    return results, flag, x
end

function recover_allocation(param, graph, x)
    # Welfare
    results = Dict(:welfare => x[1])

    # Consumption per capita
    results[:cj] = x[1+graph.J*param.N+2*graph.ndeg*param.N+graph.J+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J+graph.J]

    # Population
    results[:Lj] = x[1+graph.J*param.N+2*graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J]

    # Total consumption of tradeable good
    results[:Cj] = results[:cj] .* results[:Lj]

    # Working population
    results[:Ljn] = reshape(x[1+graph.J*param.N+2*graph.ndeg*param.N+2*graph.J+1:end], graph.J, param.N)

    # Production
    results[:Yjn] = param.Zjn .* (results[:Ljn] .^ param.a)

    # Vector of welfare per location
    results[:uj] = (results[:cj] / param.alpha) .^ param.alpha .* ((param.Hj ./ results[:Lj]) / (1 - param.alpha)) .^ (1 - param.alpha)

    # Domestic absorption per good per location
    results[:Djn] = reshape(x[2:1+graph.J*param.N], graph.J, param.N) # Consumption per good pre-transport cost

    # Total availability of tradeable good
    results[:Dj] = sum(results[:Djn] .^ ((param.sigma - 1) / param.sigma), dims=2) .^ (param.sigma / (param.sigma - 1)) # Aggregate consumption pre-transport cost

    # Non-tradeable good per capita
    results[:hj] = param.Hj ./ results[:Lj]
    results[:hj][results[:Lj] .== 0] = 0

    # Trade flows
    Qin_direct = reshape(x[1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N], graph.ndeg, param.N)
    Qin_indirect = reshape(x[1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N], graph.ndeg, param.N)
    results[:Qin] = zeros(graph.ndeg*param.N)

    for i in 1:param.N*graph.ndeg
        if Qin_direct[i] > Qin_indirect[i]
            results[:Qin][i] = Qin_direct[i]
        else
            results[:Qin][i] = -Qin_indirect[i]
        end
    end
    results[:Qin] = reshape(results[:Qin], graph.ndeg, param.N)

    # recover the Qjkn's
    results[:Qjkn] = zeros(graph.J, graph.J, param.N)
    id = 1
    for i in 1:graph.J
        for j in 1:length(graph.nodes[i])
            if graph.nodes[i][j] > i
                results[:Qjkn][i, graph.nodes[i][j], :] = max.(results[:Qin][id, :], 0)
                results[:Qjkn][graph.nodes[i][j], i, :] = max.(-results[:Qin][id, :], 0)
                id += 1
            end
        end
    end

    return results
end


# Please note that the functions `objective_mobility_cgc`, `constraints_mobility_cgc` and their derivatives are not provided in the original MATLAB code, so they are not translated here. You would need to provide these functions in Julia as well.