#=
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

solve_allocation_primal(...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case without labor mobility with
a primal approach (quasiconcave) (used when dual is not
twice-differentiable).
It DOES NOT use the autodifferentiation package Adigator to generate the 
functional inputs for IPOPT.

Arguments:
- x0: initial seed for the solver
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A)
- verbose: {true | false} tells IPOPT to display results or not

Results:
- results: structure of results (Cj,Qjkn,etc.)
- flag: flag returned by IPOPT
- x: returns the 'x' variable returned by IPOPT (useful for warm start)

REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
=#

# using Ipopt
# using LinearAlgebra

function solve_allocation_primal(x0, auxdata, verbose=true)

    # ==================
    # RECOVER PARAMETERS
    # ==================
    
    graph = auxdata[:graph]
    param = auxdata[:param]

    # check compatibility
    if any(sum(param[:Zjn] .> 0, dims=2) .> 1)
        error("this code only supports one good at most per location. Use the ADiGator version instead.")
    end

    if isempty(x0)
        x0 = [1e-6 * ones(graph.J * param[:N]); zeros(graph.ndeg * param[:N])]
        # only the case with at most one good produced per location is
        # coded when ADiGator is not used: no optimization on Ljn
    end

    # =================
    # PARAMETRIZE IPOPT
    # =================

    # Init functions
    objective_function = x -> objective_primal(x, auxdata)
    gradient_function = x -> gradient_primal(x, auxdata)
    constraint_function = x -> constraints_primal(x, auxdata)
    jacobian_function = x -> jacobian_primal(x, auxdata)
    jacobianstructure_function = () -> sparse(I, 1:graph.J * param[:N], 1:graph.J * param[:N], graph.J * param[:N], graph.J * param[:N] + graph.ndeg * param[:N])

    hessian_function = (x, sigma, lambda) -> hessian_primal(x, auxdata, sigma, lambda)
    hessianstructure_function = () -> sparse(tril(ones(graph.J * param[:N] + graph.ndeg * param[:N], graph.J * param[:N] + graph.ndeg * param[:N])))

    # Options
    lb = [1e-8 * ones(graph.J * param[:N]); -Inf * ones(graph.ndeg * param[:N])]
    ub = [Inf * ones(graph.J * param[:N]); Inf * ones(graph.ndeg * param[:N])]
    cl = -Inf * ones(graph.J * param[:N]) # lower bound on constraint function
    cu = 0 * ones(graph.J * param[:N])    # upper bound on constraint function

    # Ipopt options setup
    ipopt_options = Dict(
        "hessian_approximation" => "limited-memory",
        "max_iter" => 2000,
        "ma57_pre_alloc" => 3.0,
        "print_level" => verbose ? 5 : 0
    )

    # =========
    # RUN IPOPT
    # =========
    x, info = ipopt(x0, objective_function, gradient=gradient_function, hessian=hessian_function, 
                    constraints=(constraint_function, cl, cu), jacobian=jacobian_function,
                    hess_structure=hessianstructure_function, jac_structure=jacobianstructure_function, 
                    lb=lb, ub=ub, options=ipopt_options)

    # ==============
    # RETURN RESULTS
    # ==============
    
    # return allocation
    flag = info.status

    results = recover_allocation_primal(x, auxdata)

    results.Pjn = reshape(info.lambda[1:graph.J * param[:N]], (graph.J, param[:N])) # vector of prices
    results.PCj = sum(results.Pjn .^ (1 - param[:sigma]), dims=2) .^ (1 / (1 - param[:sigma])) # price of tradeable
    results.welfare = -objective_function(x)

    return results, flag, x
end


function recover_allocation_primal(x, auxdata)
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

function objective_primal(x, auxdata)
    param = auxdata.param
    results = recover_allocation_primal(x, auxdata)

    return -sum(param.omegaj .* results["Lj"] .* param.u(results["cj"], results["hj"]))
end

function gradient_primal(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    results = recover_allocation_primal(x, auxdata)

    return [-repeat(param.omegaj .* param.uprime(results["cj"], results["hj"]), param.N) .* results["Cjn"] .^ (-1/param.sigma) .* repeat(results["Cj"] .^ (1/param.sigma), param.N); zeros(graph.ndeg*param.N)]
end

function constraints_primal(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    A = auxdata.A
    kappa_ex = auxdata.kappa_ex
    results = recover_allocation_primal(x, auxdata)
    Qin = reshape(x[graph.J*param.N+1:end], graph.ndeg, param.N)

    cons_Q = zeros(graph.J, param.N)
    for n in 1:param.N
        M = max.(A .* (ones(graph.J) * sign.(Qin[:, n]))', 0)
        cons_Q[:, n] = results["Cjn"][:, n] + A * Qin[:, n] - results["Yjn"][:, n] + M * (abs.(Qin[:, n]) .^ (1+param.beta) ./ kappa_ex)
    end

    return cons_Q[:]
end

function jacobian_primal(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    A = auxdata.A
    kappa_ex = auxdata.kappa_ex
    results = recover_allocation_primal(x, auxdata)
    Qin = reshape(x[graph.J*param.N+1:end], graph.ndeg, param.N)

    cons_Q = zeros(graph.J*param.N, graph.ndeg*param.N)
    for n in 1:param.N
        M = max.(A .* (ones(graph.J) * sign.(Qin[:, n]))', 0)
        cons_Q[(n-1)*graph.J+1:n*graph.J, (n-1)*graph.ndeg+1:n*graph.ndeg] = A + (1+param.beta) * M .* repeat((sign.(Qin[:, n]) .* abs.(Qin[:, n]) .^ param.beta ./ kappa_ex)', graph.J)
    end

    cons_Q = [I(graph.J*param.N), cons_Q]

    return sparse(cons_Q)
end

function hessian_primal(x, auxdata, sigma_IPOPT, lambda_IPOPT)
    param = auxdata.param
    graph = auxdata.graph
    A = auxdata.A
    kappa_ex = auxdata.kappa_ex
    results = recover_allocation_primal(x, auxdata)
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