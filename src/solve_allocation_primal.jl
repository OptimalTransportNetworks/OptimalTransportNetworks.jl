
using Ipopt
using LinearAlgebra

function solve_allocation_primal(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param

    if any(sum(param.Zjn .> 0, dims=2) .> 1)
        error("This code only supports one good at most per location. Use the ADiGator version instead.")
    end

    if isempty(x0)
        x0 = [1e-6*ones(graph.J*param.N); zeros(graph.ndeg*param.N)]
    end

    model = createModel(auxdata)
    addVar!(model, x0, [1e-8*ones(graph.J*param.N); -Inf*ones(graph.ndeg*param.N)], [Inf*ones(graph.J*param.N); Inf*ones(graph.ndeg*param.N)])
    addConstr!(model, -Inf*ones(graph.J*param.N), zeros(graph.J*param.N))

    if verbose
        setPrintLevel!(model, 5)
    else
        setPrintLevel!(model, 0)
    end

    x, info = solveModel!(model)

    results = recover_allocation(x, auxdata)
    results.Pjn = reshape(info.lambda[1:graph.J*param.N], graph.J, param.N)
    results.PCj = sum(results.Pjn .^ (1-param.sigma), dims=2) .^ (1/(1-param.sigma))
    results.welfare = -objective(x, auxdata)

    return results, info, x
end

function recover_allocation(x, auxdata)
    graph = auxdata.graph
    param = auxdata.param

    results = Dict()

    results["Cjn"] = reshape(x[1:graph.J*param.N], graph.J, param.N)
    results["Cj"] = sum(results["Cjn"] .^ ((param.sigma-1)/param.sigma), dims=2) .^ (param.sigma/(param.sigma-1))
    results["Lj"] = param.Lj
    results["Ljn"] = (param.Zjn .> 0) .* results["Lj"]
    results["Yjn"] = param.Zjn .* results["Lj"] .^ param.a
    results["cj"] = results["Cj"] ./ results["Lj"]
    results["cj"][results["Lj"] .== 0] = 0
    results["hj"] = param.Hj ./ results["Lj"]
    results["hj"][results["Lj"] .== 0] = 0
    results["uj"] = param.u(results["cj"], results["hj"])
    results["Qin"] = reshape(x[graph.J*param.N+1:end], graph.ndeg, param.N)

    results["Qjkn"] = zeros(graph.J, graph.J, param.N)
    id = 1
    for i in 1:graph.J
        for j in 1:length(graph.nodes[i].neighbors)
            if graph.nodes[i].neighbors[j] > i
                results["Qjkn"][i, graph.nodes[i].neighbors[j], :] = max.(results["Qin"][id, :], 0)
                results["Qjkn"][graph.nodes[i].neighbors[j], i, :] = max.(-results["Qin"][id, :], 0)
                id += 1
            end
        end
    end

    return results
end

function objective(x, auxdata)
    param = auxdata.param
    results = recover_allocation(x, auxdata)

    return -sum(param.omegaj .* results["Lj"] .* param.u(results["cj"], results["hj"]))
end

function gradient(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    results = recover_allocation(x, auxdata)

    return [-repeat(param.omegaj .* param.uprime(results["cj"], results["hj"]), param.N) .* results["Cjn"] .^ (-1/param.sigma) .* repeat(results["Cj"] .^ (1/param.sigma), param.N); zeros(graph.ndeg*param.N)]
end

function constraints(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    A = auxdata.A
    kappa_ex = auxdata.kappa_ex
    results = recover_allocation(x, auxdata)
    Qin = reshape(x[graph.J*param.N+1:end], graph.ndeg, param.N)

    cons_Q = zeros(graph.J, param.N)
    for n in 1:param.N
        M = max.(A .* (ones(graph.J) * sign.(Qin[:, n]))', 0)
        cons_Q[:, n] = results["Cjn"][:, n] + A * Qin[:, n] - results["Yjn"][:, n] + M * (abs.(Qin[:, n]) .^ (1+param.beta) ./ kappa_ex)
    end

    return cons_Q[:]
end

function jacobian(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    A = auxdata.A
    kappa_ex = auxdata.kappa_ex
    results = recover_allocation(x, auxdata)
    Qin = reshape(x[graph.J*param.N+1:end], graph.ndeg, param.N)

    cons_Q = zeros(graph.J*param.N, graph.ndeg*param.N)
    for n in 1:param.N
        M = max.(A .* (ones(graph.J) * sign.(Qin[:, n]))', 0)
        cons_Q[(n-1)*graph.J+1:n*graph.J, (n-1)*graph.ndeg+1:n*graph.ndeg] = A + (1+param.beta) * M .* repeat((sign.(Qin[:, n]) .* abs.(Qin[:, n]) .^ param.beta ./ kappa_ex)', graph.J)
    end

    cons_Q = [I(graph.J*param.N), cons_Q]

    return sparse(cons_Q)
end

function hessian(x, auxdata, sigma_IPOPT, lambda_IPOPT)
    param = auxdata.param
    graph = auxdata.graph
    A = auxdata.A
    kappa_ex = auxdata.kappa_ex
    results = recover_allocation(x, auxdata)
    Qin = reshape(x[graph.J*param.N+1:end], graph.ndeg, param.N)
    Pjn = reshape(lambda_IPOPT[1:graph.J*param.N], graph.J, param.N)

    Hcdiag = -sigma_IPOPT * (-1/param.sigma) .* results["Cjn"] .^ (-1/param.sigma-1) .* repeat(param.omegaj .* results["Cj"] .^ (1/param.sigma) .* param.uprime(results["cj"], results["hj"]), param.N)

    CC = repeat(results["Cjn"] .^ (-1/param.sigma), 1, graph.J*param.N)

    Hcnondiag = -sigma_IPOPT * repeat(param.omegaj .* (1/param.sigma * param.uprime(results["cj"], results["hj"]) .* results["Cj"] .^ (2/param.sigma-1) + 1 ./ param.Lj .* param.usecond(results["cj"], results["hj"]) .* results["Cj"] .^ (2/param.sigma)), param.N, graph.J*param.N) .* CC .* CC'

    mask = repeat(.~I(graph.J), param.N, param.N)
    Hcnondiag[mask] = 0

    Hq = zeros(graph.ndeg*param.N)
    if param.beta > 0
        for n in 1:param.N
            Hq[(n-1)*graph.ndeg+1:n*graph.ndeg] = (1+param.beta) * param.beta * abs.(Qin[:, n]) .^ (param.beta-1) ./ kappa_ex .* sum(max.((A .* repeat(Pjn[:, n], 1, graph.ndeg)) .* repeat(sign.(Qin[:, n])', graph.J), 0), dims=1)
        end
    end

    return sparse(tril([Diagonal(Hcdiag) + Hcnondiag zeros(graph.J*param.N, graph.ndeg*param.N); zeros(graph.ndeg*param.N, graph.J*param.N) Diagonal(Hq)]))
end


# Please note that this is a direct translation and might not work as expected due to differences in indexing (1-based in Julia vs 0-based in Matlab), differences in function names and behavior, and other language-specific features. You might need to adjust the code to fit your specific needs and to ensure it works as expected.