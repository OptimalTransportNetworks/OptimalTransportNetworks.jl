
using Ipopt
using LinearAlgebra
using SparseArrays

function solve_allocation_mobility_cgc(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param

    if any(sum(param.Zjn .> 0, dims=2) .> 1)
        error("This code only supports one good at most per location. Use the ADiGator version instead.")
    end

    if isempty(x0)
        C = 1e-6
        L = 1 / graph.J
        x0 = vcat(0, fill(C / L, graph.J), fill(C, graph.J * param.N), fill(1e-6, 2 * graph.ndeg * param.N), fill(L, graph.J))
    end

    nlp = create_nlp(x0, auxdata, verbose)
    solver = IpoptSolver(print_level=verbose ? 5 : 0, max_iter=2000)
    res = solve(nlp, solver)

    x = getvalue(nlp.x)
    flag = getterminationstatus(nlp)
    results = recover_allocation(x, auxdata)
    results.omegaj = getdual(nlp.c)[1:graph.J]
    results.Pjn = reshape(getdual(nlp.c)[2 * graph.J + 1:2 * graph.J + graph.J * param.N], graph.J, param.N)
    results.PCj = (sum(results.Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma)))
    results.PHj = param.alpha / (1 - param.alpha) .* results.PCj .* results.cj ./ results.hj
    results.W = getdual(nlp.c)[2 * graph.J + graph.J * param.N + 1]

    return results, flag, x
end

function create_nlp(x0, auxdata, verbose)
    graph = auxdata.graph
    param = auxdata.param
    utility = param.u

    nlp = Model(solver=IpoptSolver(print_level=verbose ? 5 : 0))
    @variable(nlp, -Inf <= x[1:length(x0)] <= Inf)
    for i in 2:length(x0)
        setlowerbound(x[i], 1e-6)
    end
    setlowerbound(x[end - graph.J + 1:end], 1e-8)

    @NLobjective(nlp, Min, -x[1])

    @NLconstraint(nlp, cons_u[j=1:graph.J], x[1] * x[end - graph.J + j] - utility(x[1 + j], param.Hj[j]) == 0)
    @NLconstraint(nlp, cons_C[j=1:graph.J], x[1 + j] + cost_direct[j] + cost_indirect[j] - Dj[j] == 0)
    @NLconstraint(nlp, cons_Q[j=1:graph.J, n=1:param.N], Djn[j, n] + A * Qin_direct[:, n] - A * Qin_indirect[:, n] - Yjn[j, n] == 0)
    @NLconstraint(nlp, cons_L, sum(x[end - graph.J + 1:end]) - 1 == 0)

    return nlp
end

function recover_allocation(x, auxdata)
    graph = auxdata.graph
    param = auxdata.param

    results = Dict()
    results["welfare"] = x[1]
    results["Cj"] = x[2:graph.J + 1]
    results["Lj"] = x[end - graph.J + 1:end]
    results["cj"] = results["Cj"] ./ results["Lj"]
    results["cj"][results["Lj"] .== 0] = 0
    results["hj"] = param.Hj ./ results["Lj"]
    results["hj"][results["Lj"] .== 0] = 0
    results["uj"] = ((results["cj"] / param.alpha) .^ param.alpha .* (results["hj"] / (1 - param.alpha)) .^ (1 - param.alpha)) .^ (1 - param.rho) / (1 - param.rho)
    results["Ljn"] = (param.Zjn .> 0) .* results["Lj"]
    results["Yjn"] = param.Zjn .* results["Lj"] .^ param.a
    results["Djn"] = max.(0, reshape(x[graph.J + 2:graph.J + 1 + graph.J * param.N], graph.J, param.N))
    results["Dj"] = sum(results["Djn"] .^ ((param.sigma - 1) / param.sigma), dims=2) .^ (param.sigma / (param.sigma - 1))
    results["Qin_direct"] = reshape(x[graph.J + 1 + graph.J * param.N + 1:graph.J + 1 + graph.J * param.N + graph.ndeg * param.N], graph.ndeg, param.N)
    results["Qin_indirect"] = reshape(x[graph.J + 1 + graph.J * param.N + graph.ndeg * param.N + 1:graph.J + 1 + graph.J * param.N + 2 * graph.ndeg * param.N], graph.ndeg, param.N)
    results["Qin"] = max.(results["Qin_direct"] - results["Qin_indirect"], 0) - max.(results["Qin_indirect"] - results["Qin_direct"], 0)
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


# Please note that this is a direct translation and might not work as expected due to differences in how Matlab and Julia handle arrays and indexing. You might need to adjust the code to fit your specific needs.