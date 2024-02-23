
# using Ipopt
# using LinearAlgebra
# using SparseArrays

function solve_allocation_mobility(x0, auxdata, verbose=true)

    graph = auxdata.graph
    param = auxdata.param
    A = auxdata.A

    if any(sum(param.Zjn .> 0, dims=2) .> 1)
        error("This code only supports one good at most per location.")
    end

    if isempty(x0)
        x0 = [0; 1e-6*ones(graph.J*param.N); 1e-4*ones(2*graph.ndeg*param.N); 1/graph.J*ones(graph.J)]
    end

    function objective(x)
        return -x[1]
    end

    function gradient(x)
        g = zeros(1+graph.J*param.N+2*graph.ndeg*param.N+graph.J)
        g[1] = -1
        return g
    end

    function constraints(x)
        u = x[1]
        Cjn = reshape(x[2:graph.J*param.N+1], graph.J, param.N)
        Qin_direct = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)
        Qin_indirect = reshape(x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+2*graph.ndeg*param.N+1], graph.ndeg, param.N)
        Lj = x[graph.J*param.N+2*graph.ndeg*param.N+2:end]
        Cj = sum(Cjn.^((param.sigma-1)/param.sigma), dims=2).^(param.sigma/(param.sigma-1))
        Yjn = param.Zjn .* Lj.^param.a

        cons_u = Lj .* u - (Cj / param.alpha).^param.alpha .* (param.Hj / (1-param.alpha)).^(1-param.alpha)

        cons_Q = zeros(graph.J, param.N)
        for n in 1:param.N
            cons_Q[:, n] = Cjn[:, n] + Apos * (Qin_direct[:, n].^(1+param.beta) ./ kappa_ex) + Aneg * (Qin_indirect[:, n].^(1+param.beta) ./ kappa_ex) + A * Qin_direct[:, n] - A * Qin_indirect[:, n] - Yjn[:, n]
        end

        cons_L = sum(Lj) - 1

        return [cons_u; cons_Q[:]; cons_L]
    end

    function jacobian(x)
        u = x[1]
        Cjn = reshape(x[2:graph.J*param.N+1], graph.J, param.N)
        Qin_direct = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)
        Qin_indirect = reshape(x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+2*graph.ndeg*param.N+1], graph.ndeg, param.N)
        Lj = x[graph.J*param.N+2*graph.ndeg*param.N+2:end]
        Cj = sum(Cjn.^((param.sigma-1)/param.sigma), dims=2).^(param.sigma/(param.sigma-1))

        cons_u = zeros(graph.J, graph.J*param.N)
        for n in 1:param.N
            cons_u[:, (n-1)*graph.J+1:n*graph.J] = Diagonal(-Cjn[:, n].^-(1/param.sigma) .* Cj.^(1/param.sigma) .* (Cj / param.alpha).^(param.alpha-1) .* (param.Hj / (1-param.alpha)).^(1-param.alpha))
        end

        cons_u = [Lj cons_u zeros(graph.J, 2*graph.ndeg*param.N) u*Matrix(I, graph.J, graph.J)]

        cons_Q = zeros(graph.J*param.N, 2*graph.ndeg*param.N)
        for n in 1:param.N
            cons_Q[(n-1)*graph.J+1:n*graph.J, (n-1)*graph.ndeg+1:n*graph.ndeg] = A + (1+param.beta) * Apos .* (Qin_direct[:, n].^(param.beta) ./ kappa_ex)'
            cons_Q[(n-1)*graph.J+1:n*graph.J, graph.ndeg*param.N+(n-1)*graph.ndeg+1:graph.ndeg*param.N+n*graph.ndeg] = -A + (1+param.beta) * Aneg .* (Qin_indirect[:, n].^(param.beta) ./ kappa_ex)'
        end

        cons_Z = zeros(graph.J*param.N, graph.J)
        for n in 1:param.N
            cons_Z[(n-1)*graph.J+1:n*graph.J, :] = -param.a * param.Zjn[:, n] .* Lj.^(param.a-1)
        end

        cons_Q = [zeros(graph.J*param.N, 1) Matrix(I, graph.J*param.N, graph.J*param.N) cons_Q cons_Z]

        cons_L = [zeros(1, 1+graph.J*param.N+2*graph.ndeg*param.N) ones(1, graph.J)]

        return sparse([cons_u; cons_Q; cons_L])
    end

    function hessian(x, sigma, lambda)
        Cjn = reshape(x[2:graph.J*param.N+1], graph.J, param.N)
        Qin_direct = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)
        Qin_indirect = reshape(x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+2*graph.ndeg*param.N+1], graph.ndeg, param.N)
        Lj = x[graph.J*param.N+2*graph.ndeg*param.N+2:end]
        Cj = sum(Cjn.^((param.sigma-1)/param.sigma), dims=2).^(param.sigma/(param.sigma-1))

        omegaj = lambda[1:graph.J]
        Pjn = reshape(lambda[graph.J+1:graph.J+graph.J*param.N], graph.J, param.N)

        Hcdiag = 1/param.sigma * omegaj .* param.uprime(Cj, param.Hj) .* Cj.^(1/param.sigma) .* Cjn.^(-1/param.sigma-1)

        CC = repmat(Cjn.^(-1/param.sigma), 1, graph.J*param.N)

        Hcnondiag = -repmat(omegaj .* param.usecond(Cj, param.Hj) .* Cj.^(2/param.sigma) + 1/param.sigma * omegaj .* param.uprime(Cj, param.Hj) .* Cj.^(2/param.sigma-1), param.N, graph.J*param.N) .* CC .* CC'

        mask = repmat(!I, param.N, param.N)
        Hcnondiag[mask] = 0

        Hqpos = zeros(graph.ndeg*param.N)
        Hqneg = zeros(graph.ndeg*param.N)
        for n in 1:param.N
            Hqpos[(n-1)*graph.ndeg+1:n*graph.ndeg] = (1+param.beta) * param.beta * Qin_direct[:, n].^(param.beta-1) ./ kappa_ex .* (Apos' * Pjn[:, n])
            Hqneg[(n-1)*graph.ndeg+1:n*graph.ndeg] = (1+param.beta) * param.beta * Qin_indirect[:, n].^(param.beta-1) ./ kappa_ex .* (Aneg' * Pjn[:, n])
        end

        Hl = Diagonal(sum(-param.a * (param.a-1) * Pjn .* Lj.^(param.a-2), dims=2))

        h = sparse(tril([zeros(1, 1+graph.J*param.N+2*graph.ndeg*param.N+graph.J); zeros(graph.J*param.N, 1) Diagonal(Hcdiag) + Hcnondiag zeros(graph.J*param.N, 2*graph.ndeg*param.N+graph.J); zeros(graph.ndeg*param.N, 1+graph.J*param.N) Diagonal(Hqpos) zeros(graph.ndeg*param.N, graph.ndeg*param.N+graph.J); zeros(graph.ndeg*param.N, 1+graph.J*param.N+graph.ndeg*param.N) Diagonal(Hqneg) zeros(graph.ndeg*param.N, graph.J); omegaj zeros(graph.J, graph.J*param.N+2*graph.ndeg*param.N) Hl]))
        return h
    end

    lb = [-Inf; 1e-6*ones(graph.J*param.N); 1e-6*ones(2*graph.ndeg*param.N); 1e-8*ones(graph.J)]
    ub = [Inf; Inf*ones(graph.J*param.N); Inf*ones(2*graph.ndeg*param.N); ones(graph.J)]
    cl = [-Inf*ones(graph.J); -Inf*ones(graph.J*param.N); 0]
    cu = [zeros(graph.J); 1e-3*ones(graph.J*param.N); 0]

    nlp = createProblem(objective, gradient, x0, lb, ub, constraints, jacobian, cl, cu, hessian)

    options = Dict(:max_iter => 2000, :print_level => verbose ? 5 : 0)

    x, info = solveProblem(nlp, options)

    flag = info

    results = recover_allocation_mobility(x, auxdata)

    results.Pjn = reshape(info.lambda[graph.J+1:graph.J+graph.J*param.N], graph.J, param.N)
    results.PCj = sum(results.Pjn.^(1-param.sigma), dims=2).^(1/(1-param.sigma))

    return results, flag, x
end

function recover_allocation_mobility(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph

    results = Dict()

    results["welfare"] = x[1]

    results["Cjn"] = reshape(x[2:graph.J*param.N+1], graph.J, param.N)

    results["Cj"] = sum(results["Cjn"].^((param.sigma-1)/param.sigma), dims=2).^(param.sigma/(param.sigma-1))

    results["Lj"] = x[graph.J*param.N+2*graph.ndeg*param.N+2:end]

    results["Ljn"] = (param.Zjn .> 0) .* results["Lj"]

    results["Yjn"] = param.Zjn .* results["Lj"].^param.a

    results["cj"] = results["Cj"] ./ results["Lj"]
    results["cj"][results["Lj"] .== 0] = 0

    results["hj"] = param.Hj ./ results["Lj"]
    results["hj"][results["Lj"] .== 0] = 0

    results["uj"] = param.u(results["cj"], results["hj"])

    results["Qin_direct"] = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)
    results["Qin_indirect"] = reshape(x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+2*graph.ndeg*param.N+1], graph.ndeg, param.N)

    results["Qin"] = zeros(graph.ndeg, param.N)

    for i in 1:param.N*graph.ndeg
        if results["Qin_direct"][i] > results["Qin_indirect"][i]
            results["Qin"][i] = results["Qin_direct"][i] - results["Qin_indirect"][i]
        else
            results["Qin"][i] = results["Qin_direct"][i] - results["Qin_indirect"][i]
        end
    end

    results["Qjkn"] = zeros(graph.J, graph.J, param.N)
    id = 1
    for i in 1:graph.J
        for j in 1:length(graph.nodes[i])
            if graph.nodes[i][j] > i
                results["Qjkn"][i, graph.nodes[i][j], :] = max(results["Qin"][id, :], 0)
                results["Qjkn"][graph.nodes[i][j], i, :] = max(-results["Qin"][id, :], 0)
                id += 1
            end
        end
    end

    return results
end


# Please note that this translation assumes that the `Ipopt.jl` package in Julia has similar functionality to the IPOPT library in Matlab. Also, the `createProblem` and `solveProblem` functions are not standard Julia functions and might need to be replaced with appropriate Julia equivalents.