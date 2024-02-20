
using Ipopt

function solve_allocation_ADiGator(x0, auxdata, funcs, verbose=true)

    graph = auxdata.graph
    param = auxdata.param

    if isempty(x0)
        x0 = [1e-6*ones(graph.J*param.N); zeros(graph.ndeg*param.N); sum(param.Lj)/(graph.J*param.N)*ones(graph.J*param.N)]
    end

    options = Dict(
        :lb => [1e-8*ones(graph.J*param.N); -Inf*ones(graph.ndeg*param.N); 1e-8*ones(graph.J*param.N)],
        :ub => [Inf*ones(graph.J*param.N); Inf*ones(graph.ndeg*param.N); Inf*ones(graph.J*param.N)],
        :cl => [-Inf*ones(graph.J*param.N); -Inf*ones(graph.J)],
        :cu => [zeros(graph.J*param.N); zeros(graph.J)],
        :print_level => verbose ? 5 : 0
    )

    x, info = ipopt(x0, funcs, options)

    flag = info
    results = recover_allocation(x, auxdata)

    results.Pjn = reshape(info.lambda[1:graph.J*param.N], graph.J, param.N)
    results.PCj = sum(results.Pjn.^(1-param.sigma), dims=2).^(1/(1-param.sigma))
    results.welfare = -funcs.objective(x)

    return results, flag, x
end

function recover_allocation(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph

    results = Dict()
    results[:Lj] = param.Lj
    results[:Ljn] = reshape(x[graph.J*param.N+graph.ndeg*param.N+1:end], graph.J, param.N)
    results[:Yjn] = param.Zjn .* (results[:Ljn] .^ param.a)
    results[:Cjn] = reshape(x[1:graph.J*param.N], graph.J, param.N)
    results[:Cj] = sum(results[:Cjn] .^ ((param.sigma-1)/param.sigma), dims=2) .^ (param.sigma/(param.sigma-1))
    results[:cj] = results[:Cj] ./ results[:Lj]
    results[:cj][results[:Lj] .== 0] = 0
    results[:hj] = param.hj
    results[:uj] = param.u(results[:cj], results[:hj])
    results[:Qin] = reshape(x[graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N], graph.ndeg, param.N)
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


# Please note that the `ipopt` function used in the Matlab code is from the IPOPT library, which is a library for large scale nonlinear optimization. The Julia equivalent also uses the IPOPT library, but through the Ipopt.jl package. You may need to adjust the code to fit the exact syntax and usage of the Ipopt.jl package.