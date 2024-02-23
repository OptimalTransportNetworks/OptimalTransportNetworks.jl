
using Ipopt

function solve_allocation_mobility_ADiGator(x0, auxdata, funcs, verbose=true)

    graph = auxdata.graph
    param = auxdata.param

    if any(sum(param.Zjn .> 0, dims=2) .> 1) && param.a == 1
        error("Version of the code with more than 1 good per location and a=1 not supported yet.")
    end

    if isempty(x0)
        x0 = vcat(0, 1e-6*ones(graph.J*param.N), zeros(graph.ndeg*param.N), 1/graph.J*ones(graph.J), 1/(graph.J*param.N)*ones(graph.J*param.N))
    end

    funcs.objective = (x) -> objective_mobility(x, auxdata)
    funcs.gradient = (x) -> objective_mobility_Grd(x, auxdata)
    funcs.constraints = (x) -> constraints_mobility(x, auxdata)
    funcs.jacobian = (x) -> constraints_mobility_Jac(x, auxdata)
    funcs.hessian = (x, sigma, lambda) -> objective_mobility_Hes(x, auxdata, sigma, lambda)

    options = Dict(
        :lb => vcat(-Inf, 1e-8*ones(graph.J*param.N), -Inf*ones(graph.ndeg*param.N), 1e-8*ones(graph.J), 1e-8*ones(graph.J*param.N)),
        :ub => vcat(Inf, Inf*ones(graph.J*param.N), Inf*ones(graph.ndeg*param.N), ones(graph.J), Inf*ones(graph.J*param.N)),
        :cl => vcat(-Inf*ones(graph.J), -Inf*ones(graph.J*param.N), -1e-8, -1e-8*ones(graph.J)),
        :cu => vcat(-1e-8*ones(graph.J), -1e-8*ones(graph.J*param.N), 1e-8, 1e-8*ones(graph.J))
    )

    if verbose
        options[:print_level] = 5
    else
        options[:print_level] = 0
    end

    x, info = ipopt(x0, funcs, options)

    flag = info
    results = recover_allocation(x, auxdata)
    results.Pjn = reshape(info.lambda[graph.J+1:graph.J+graph.J*param.N], graph.J, param.N)
    results.PCj = sum(results.Pjn.^(1-param.sigma), dims=2).^(1/(1-param.sigma))

    return results, flag, x
end

function recover_allocation(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph

    results = Dict()
    results["welfare"] = x[1]
    results["Cjn"] = reshape(x[2:graph.J*param.N+1], graph.J, param.N)
    results["Cj"] = sum(results["Cjn"].^((param.sigma-1)/param.sigma), dims=2).^(param.sigma/(param.sigma-1))
    results["Lj"] = x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+graph.ndeg*param.N+graph.J+1]
    results["Ljn"] = reshape(x[graph.J*param.N+graph.ndeg*param.N+graph.J+2:end], graph.J, param.N)
    results["Yjn"] = param.Zjn .* results["Ljn"].^param.a
    results["cj"] = results["Cj"] ./ results["Lj"]
    results["cj"][results["Lj"].==0] = 0
    results["hj"] = param.Hj ./ results["Lj"]
    results["hj"][results["Lj"].==0] = 0
    results["uj"] = param.u(results["cj"], results["hj"])
    results["Qin"] = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)

    results["Qjkn"] = zeros(graph.J, graph.J, param.N)
    id = 1
    for i in 1:graph.J
        for j in 1:length(graph.nodes[i])
            if graph.nodes[i][j] > i
                results["Qjkn"][i, graph.nodes[i][j], :] = max.(results["Qin"][id, :], 0)
                results["Qjkn"][graph.nodes[i][j], i, :] = max.(-results["Qin"][id, :], 0)
                id += 1
            end
        end
    end

    return results
end


# Please note that the functions `objective_mobility`, `objective_mobility_Grd`, `constraints_mobility`, `constraints_mobility_Jac`, and `objective_mobility_Hes` are not provided in the original MATLAB code, so they are not translated here. You will need to provide these functions in Julia. Also, the `ipopt` function in Julia may have a different syntax and usage than the `ipopt` function in MATLAB. You may need to adjust the code accordingly.