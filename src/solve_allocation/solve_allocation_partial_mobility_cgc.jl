
using Ipopt
using LinearAlgebra
using SparseArrays

function solve_allocation_partial_mobility_cgc(x0, auxdata, verbose=true)

    graph = auxdata.graph
    param = auxdata.param

    if any(sum(param.Zjn .> 0, dims=2) .> 1)
        error("This code only supports one good at most per location. Use the ADiGator version instead.")
    end

    if isempty(x0)
        C = 1e-6
        L = 1 / graph.J
        x0 = vcat(zeros(param.nregions), fill(C/L, graph.J), fill(C, graph.J*param.N), fill(1e-8, 2*graph.ndeg*param.N), fill(L, graph.J))
    end

    location = zeros(param.nregions, graph.J)
    for i in 1:param.nregions
        location[i, :] = (graph.region .== i)
    end

    nlp = create_nlp(x0, auxdata, location)

    options = Ipopt.Options()
    options["max_iter"] = 2000
    options["print_level"] = verbose ? 5 : 0

    solver = Ipopt.IpoptSolver(options)
    results = JuMP.solve(nlp, solver)

    flag = getterminationstatus(nlp)
    x = getvalue(nlp)

    return recover_allocation(x, auxdata), flag, x
end

function create_nlp(x0, auxdata, location)
    param = auxdata.param
    graph = auxdata.graph

    nlp = Model(solver=IpoptSolver(print_level=0))
    @variable(nlp, x[1:length(x0)])
    setvalue(x, x0)

    @NLobjective(nlp, Min, -sum(param.omegar .* param.Lr .* x[1:param.nregions]))

    @NLconstraint(nlp, x[1:param.nregions] .== ((x[param.nregions+1:param.nregions+graph.J] / param.alpha) .^ param.alpha .* (param.Hj / (1-param.alpha)) .^ (1-param.alpha)) ./ x[param.nregions+graph.J+graph.J*param.N+2*graph.ndeg*param.N+1:end])

    @NLconstraint(nlp, x[param.nregions+1:param.nregions+graph.J] .+ Apos * (sum(repmat(param.m', graph.ndeg, 1) .* x[param.nregions+graph.J+graph.J*param.N+1:param.nregions+graph.J+graph.J*param.N+graph.ndeg*param.N] .^ param.nu, dims=2) .^ ((param.beta+1) / param.nu) ./ kappa_ex) + Aneg * (sum(repmat(param.m', graph.ndeg, 1) .* x[param.nregions+graph.J+graph.J*param.N+graph.ndeg*param.N+1:param.nregions+graph.J+graph.J*param.N+2*graph.ndeg*param.N] .^ param.nu, dims=2) .^ ((param.beta+1) / param.nu) ./ kappa_ex) .== sum(reshape(x[param.nregions+graph.J+1:param.nregions+graph.J+graph.J*param.N], graph.J, param.N) .^ ((param.sigma-1) / param.sigma), dims=2) .^ (param.sigma / (param.sigma-1)))

    @NLconstraint(nlp, [j=1:graph.J, n=1:param.N], reshape(x[param.nregions+graph.J+1:param.nregions+graph.J+graph.J*param.N], graph.J, param.N)[j, n] + A * x[param.nregions+graph.J+graph.J*param.N+(n-1)*graph.ndeg+1:param.nregions+graph.J+graph.J*param.N+n*graph.ndeg] - A * x[param.nregions+graph.J+graph.J*param.N+graph.ndeg*param.N+(n-1)*graph.ndeg+1:param.nregions+graph.J+graph.J*param.N+graph.ndeg*param.N+n*graph.ndeg] .== param.Zjn[j, n] * x[param.nregions+graph.J+graph.J*param.N+2*graph.ndeg*param.N+j] .^ param.a)

    @NLconstraint(nlp, sum(location .* x[param.nregions+graph.J+graph.J*param.N+2*graph.ndeg*param.N+1:end]', dims=2) .== param.Lr)

    setlowerbound(x, fill(-Inf, param.nregions), fill(1e-6, graph.J), fill(1e-6, graph.J*param.N), fill(1e-8, 2*graph.ndeg*param.N), fill(1e-8, graph.J))
    setupperbound(x, fill(Inf, length(x)))

    return nlp
end

function recover_allocation(x, auxdata)
    graph = auxdata.graph
    param = auxdata.param

    results = Dict()

    results["Cj"] = x[param.nregions+1:param.nregions+graph.J]
    results["Lj"] = x[param.nregions+graph.J+graph.J*param.N+2*graph.ndeg*param.N+1:end]
    results["cj"] = results["Cj"] ./ results["Lj"]
    results["cj"][results["Lj"] .== 0] = 0
    results["hj"] = param.Hj ./ results["Lj"]
    results["hj"][results["Lj"] .== 0] = 0
    results["uj"] = ((results["cj"] / param.alpha) .^ param.alpha .* (results["hj"] / (1-param.alpha)) .^ (1-param.alpha))
    results["welfare"] = sum(param.omegar .* param.Lr .* x[1:param.nregions])
    results["ur"] = x[1:param.nregions]

    results["Ljn"] = (param.Zjn .> 0) .* results["Lj"]
    results["Yjn"] = param.Zjn .* results["Lj"] .^ param.a
    results["Djn"] = max.(0, reshape(x[param.nregions+graph.J+1:param.nregions+graph.J+graph.J*param.N], graph.J, param.N))
    results["Dj"] = sum(results["Djn"] .^ ((param.sigma-1) / param.sigma), dims=2) .^ (param.sigma / (param.sigma-1))

    results["Qin"] = zeros(graph.ndeg, param.N)
    for i in 1:param.N*graph.ndeg
        if Qin_direct[i] > Qin_indirect[i]
            results["Qin"][i] = Qin_direct[i] - Qin_indirect[i]
        else
            results["Qin"][i] = Qin_direct[i] - Qin_indirect[i]
        end
    end

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


# Please note that this is a direct translation and might not work as expected due to differences in how Matlab and Julia handle arrays and indexing. You might need to adjust the code to fit Julia's 1-based indexing and column-major order. Also, the Ipopt package in Julia might have a different API than the one in Matlab, so you might need to adjust the code accordingly.