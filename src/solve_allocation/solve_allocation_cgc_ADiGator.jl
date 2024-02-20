
using Ipopt
using LinearAlgebra

function solve_allocation_cgc_ADiGator(x0, auxdata, funcs, verbose=true)

    # Recover parameters
    graph = auxdata.graph
    param = auxdata.param

    if isempty(x0)
        x0 = [1e-6*ones(graph.J*param.N); zeros(2*graph.ndeg*param.N); 1e-6*ones(graph.J); 1e-6*ones(graph.J*param.N)] # starting point
    end

    # Precalculations
    graph = auxdata.graph
    param = auxdata.param

    # Parametrize Ipopt
    funcs = Dict(
        "objective" => (x) -> objective_cgc(x, auxdata),
        "gradient" => (x) -> objective_cgc_Grd(x, auxdata),
        "constraints" => (x) -> constraints_cgc(x, auxdata),
        "jacobian" => (x) -> constraints_cgc_Jac(x, auxdata),
        "hessian" => (x, sigma, lambda) -> objective_cgc_Hes(x, auxdata, sigma, lambda)
    )

    # Options
    options = Dict(
        "lb" => [1e-8*ones(graph.J*param.N); 1e-8*ones(2*graph.ndeg*param.N); 1e-8*ones(graph.J); 1e-8*ones(graph.J*param.N)],
        "ub" => [Inf*ones(graph.J*param.N); Inf*ones(2*graph.ndeg*param.N+graph.J); Inf*ones(graph.J*param.N)],
        "cl" => -Inf*ones(graph.J*param.N+2*graph.J),
        "cu" => -1e-8*ones(graph.J*param.N+2*graph.J)
    )

    if verbose
        options["ipopt.print_level"] = 5
    else
        options["ipopt.print_level"] = 0
    end

    # Run Ipopt
    x, info = ipopt(x0, funcs, options)

    # Return results
    flag = info
    results = recover_allocation(param, graph, x)
    results.welfare = -funcs["objective"](x)    # Total economy welfare
    results.Pjn = reshape(info.lambda[1:graph.J*param.N], graph.J, param.N) # Price vector
    results.PCj = (sum(results.Pjn.^(1-param.sigma), dims=2)).^(1/(1-param.sigma)) # Price of tradeable

    return results, flag, x
end

function recover_allocation(param, graph, x)

    # Domestic absorption per good per location
    results = Dict()
    results["Djn"] = reshape(x[1:graph.J*param.N], graph.J, param.N) # Consumption per good pre-transport cost

    # Total availability of tradeable good
    results["Dj"] = (sum(results["Djn"].^((param.sigma-1)/param.sigma), dims=2)).^(param.sigma/(param.sigma-1)) # Aggregate consumption pre-transport cost

    # Population
    results["Lj"] = param.Lj

    # Consumption per capita
    results["cj"] = x[graph.J*param.N+2*graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N+graph.J]

    # Vector of welfare per location
    results["uj"] = ((results["cj"]/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(1-param.rho)/(1-param.rho)  

    # Total consumption of tradeable good
    results["Cj"] = results["cj"].*param.Lj

    # Working population
    results["Ljn"] = reshape(x[graph.J*param.N+2*graph.ndeg*param.N+graph.J+1:end], graph.J, param.N)

    # Non-tradeable per capita
    results["hj"] = param.hj

    # Production
    results["Yjn"] = param.Zjn.*(results["Ljn"].^param.a)

    # Trade flows
    Qin_direct = x[graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N]
    Qin_indirect = x[graph.J*param.N+graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N]
    results["Qin"] = zeros(graph.ndeg*param.N)

    for i=1:param.N*graph.ndeg
        if Qin_direct[i]>Qin_indirect[i]
            results["Qin"][i] = Qin_direct[i]-Qin_indirect[i]
        else
            results["Qin"][i] = Qin_direct[i]-Qin_indirect[i]
        end
    end
    results["Qin"] = reshape(results["Qin"], graph.ndeg, param.N)

    # recover the Qjkn's
    results["Qjkn"] = zeros(graph.J, graph.J, param.N)
    id = 1
    for i=1:graph.J
        for j=1:length(graph.nodes[i].neighbors)
            if graph.nodes[i].neighbors[j]>i
               results["Qjkn"][i, graph.nodes[i].neighbors[j], :] = max(results["Qin"][id, :], 0)
               results["Qjkn"][graph.nodes[i].neighbors[j], i, :] = max(-results["Qin"][id, :], 0)
               id += 1
            end
        end
    end

    return results
end


# Please note that the functions `objective_cgc`, `objective_cgc_Grd`, `constraints_cgc`, `constraints_cgc_Jac`, and `objective_cgc_Hes` are not provided in the MATLAB code, so they are not translated here. You will need to translate these functions separately. Also, the `ipopt` function in Julia may not have the same syntax and functionality as the `ipopt` function in MATLAB. You may need to adjust the code accordingly.