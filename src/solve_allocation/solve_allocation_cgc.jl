
using Ipopt
using JuMP
using LinearAlgebra
using SparseArrays

function solve_allocation_cgc(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param

    if any(sum(param.Zjn .> 0, dims=2) .> 1)
        error("This code only supports one good at most per location. Use the ADiGator version instead.")
    end

    if isempty(x0)
        C = 1e-6
        x0 = [C/param.Lj[1]*ones(graph.J); C*ones(graph.J*param.N); 1e-8*ones(2*graph.ndeg*param.N)] 
    end

    model = createProblem(x0, auxdata, verbose)
    status = solveProblem(model, x0)

    results = recover_allocation(model, auxdata)
    results.welfare = -objective(model, x0)
    results.Pjn = reshape(getduals(model), (graph.J, param.N))
    results.PCj = sum(results.Pjn .^ (1-param.sigma), dims=2) .^ (1/(1-param.sigma))

    return results, status, getvalue(model)
end

function createProblem(x0, auxdata, verbose)
    graph = auxdata.graph
    param = auxdata.param
    utility = param.u

    lb = [1e-8*ones(graph.J); 1e-6*ones(graph.J*param.N); 1e-8*ones(2*graph.ndeg*param.N)]
    ub = [Inf*ones(graph.J); Inf*ones(graph.J*param.N); Inf*ones(2*graph.ndeg*param.N)]
    cl = -Inf*ones(graph.J*(1+param.N))
    cu = zeros(graph.J*(1+param.N))

    model = Model(with_optimizer(Ipopt.Optimizer, print_level=verbose ? 5 : 0, max_iter=2000))
    @variable(model, lb[i] <= x[i=1:length(x0)] <= ub[i])
    @NLobjective(model, Min, -sum(param.omegaj .* param.Lj .* utility(x[1:graph.J], param.hj)))
    @NLconstraint(model, [i=1:graph.J], x[i]*param.Lj[i] + cost_direct[i] + cost_indirect[i] - Dj[i] == 0)
    @NLconstraint(model, [i=1:graph.J, n=1:param.N], Djn[i, n] + A*Qin_direct[:, n] - A*Qin_indirect[:, n] - Yjn[i, n] == 0)

    return model
end

function solveProblem(model, x0)
    set_start_value.(all_variables(model), x0)
    optimize!(model)
    return termination_status(model)
end

function recover_allocation(model, auxdata)
    graph = auxdata.graph
    param = auxdata.param
    x = getvalue(model)

    cj = x[1:graph.J]
    Cj = cj .* param.Lj
    Djn = reshape(x[graph.J+1:graph.J+graph.J*param.N], (graph.J, param.N))
    Dj = sum(Djn .^ ((param.sigma-1)/param.sigma), dims=2) .^ (param.sigma/(param.sigma-1))
    uj = ((cj/param.alpha) .^ param.alpha .* (param.hj/(1-param.alpha)) .^ (1-param.alpha)) .^ (1-param.rho) / (1-param.rho)
    Lj = param.Lj
    hj = param.hj
    Ljn = (param.Zjn .> 0) .* Lj
    Yjn = param.Zjn .* Lj .^ param.a
    Qin = zeros(graph.ndeg, param.N)

    for i=1:param.N*graph.ndeg
        if Qin_direct[i] > Qin_indirect[i]
            Qin[i] = Qin_direct[i] - Qin_indirect[i]
        else
            Qin[i] = Qin_direct[i] - Qin_indirect[i]
        end
    end

    Qjkn = zeros(graph.J, graph.J, param.N)
    id = 1
    for i=1:graph.J
        for j=1:length(graph.nodes[i])
            if graph.nodes[i][j] > i
                Qjkn[i, graph.nodes[i][j], :] = max(Qin[id, :], 0)
                Qjkn[graph.nodes[i][j], i, :] = max(-Qin[id, :], 0)
                id += 1
            end
        end
    end

    return Dict("cj" => cj, "Cj" => Cj, "Djn" => Djn, "Dj" => Dj, "uj" => uj, "Lj" => Lj, "hj" => hj, "Ljn" => Ljn, "Yjn" => Yjn, "Qin" => Qin, "Qjkn" => Qjkn)
end


# Please note that this is a direct translation and might not work as expected due to differences in how Matlab and Julia handle certain operations. You might need to adjust the code to fit your specific needs.