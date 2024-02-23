
# using Ipopt
# using LinearAlgebra
# using SparseArrays

function solve_allocation_mobility_cgc_inefficiency(x0, auxdata, verbose=true)
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

    function objective(x)
        u = x[1]
        return -u
    end

    function gradient(x)
        return vcat(-1, zeros(graph.J + graph.J * param.N + 2 * graph.ndeg * param.N + graph.J))
    end

    function constraints(x)
        u = x[1]
        Cj = x[2:graph.J+1]
        Djn = reshape(x[graph.J+2:graph.J+graph.J*param.N+1], graph.J, param.N)
        Dj = sum(Djn .^ ((param.sigma - 1) / param.sigma), dims=2) .^ (param.sigma / (param.sigma - 1))
        Qin_direct = reshape(x[graph.J+graph.J*param.N+2:graph.J+graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)
        Qin_indirect = reshape(x[graph.J+graph.J*param.N+graph.ndeg*param.N+2:graph.J+graph.J*param.N+2*graph.ndeg*param.N+1], graph.ndeg, param.N)
        Lj = x[graph.J+graph.J*param.N+2*graph.ndeg*param.N+2:end]
        Yjn = param.Zjn .* Lj .^ param.a

        cons_u = u * Lj - param.u(Cj, param.Hj)
        cost_direct = Apos * (sum(repmat(param.m', graph.ndeg, 1) .* Qin_direct .^ param.nu, dims=2) .^ ((param.beta + 1) / param.nu) .* graph.delta_tau_spillover_ex_direct ./ kappa_ex)
        cost_indirect = Aneg * (sum(repmat(param.m', graph.ndeg, 1) .* Qin_indirect .^ param.nu, dims=2) .^ ((param.beta + 1) / param.nu) .* graph.delta_tau_spillover_ex_indirect ./ kappa_ex)
        cons_C = Cj + cost_direct + cost_indirect - Dj

        cons_Q = zeros(graph.J, param.N)
        for n in 1:param.N
            cons_Q[:, n] = Djn[:, n] + A * Qin_direct[:, n] - A * Qin_indirect[:, n] - Yjn[:, n]
        end

        cons_L = sum(Lj) - 1
        return vcat(cons_u, cons_C, vec(cons_Q), cons_L)
    end

    function jacobian(x, auxdata)
        # Unpack auxdata
        param = auxdata[:param]
        graph = auxdata[:graph]
        A = auxdata[:A]
        Apos = auxdata[:Apos]
        Aneg = auxdata[:Aneg]
        kappa_ex = auxdata[:kappa_ex]
    
        # Recover variables
        u = x[1]
        Cj = x[2:1+graph.J]
        Djn = reshape(x[1+graph.J+1:1+graph.J+graph.J*param[:N]], (graph.J, param[:N]))
        Dj = sum(Djn.^((param[:sigma]-1)/param[:sigma]), dims=2).^(param[:sigma]/(param[:sigma]-1)) # total availability of final good, not consumption!
        Qin_direct = reshape(x[1+graph.J+graph.J*param[:N]+1:1+graph.J+graph.J*param[:N]+graph.ndeg*param[:N]], (graph.ndeg, param[:N]))
        Qin_indirect = reshape(x[1+graph.J+graph.J*param[:N]+graph.ndeg*param[:N]+1:1+graph.J+graph.J*param[:N]+2*graph.ndeg*param[:N]], (graph.ndeg, param[:N]))
        Lj = x[end-length(graph.J)+1:end]
    
        # Compute jacobian of welfare equalization
        J1 = [Lj'; -Diagonal(param[:uprime].(Cj, param[:Hj])); zeros(graph.J, graph.J*param[:N]+2*graph.ndeg*param[:N]); u*Matrix(I, graph.J, graph.J)]
    
        # Compute jacobian of final good availability
    
        # part corresponding to Djn
        JD = zeros(graph.J, graph.J*param[:N])
        for n in 1:param[:N]
            JD[:, (graph.J*(n-1)+1):graph.J*n] = -Diagonal(Dj.^(-1/param[:sigma]) .* Djn[:, n])
        end
    
        # part corresponding to Q
        matm = repeat(param[:m]', graph.ndeg)
        costpos = (sum(matm .* Qin_direct.^param[:nu], dims=1).^(1/param[:nu]) .* graph.delta_tau_spillover_ex_direct ./ kappa_ex).^(param[:beta])
        costneg = (sum(matm .* Qin_indirect.^param[:nu], dims=1).^(1/param[:nu]) .* graph.delta_tau_spillover_ex_indirect ./ kappa_ex).^(param[:beta])
    
        JQpos = zeros(graph.J, graph.ndeg*param[:N])
        JQneg = zeros(graph.J, graph.ndeg*param[:N])
        for n in 1:param[:N]
            vecpos = (1+param[:beta]) .* costpos .* param[:m][n] .* Qin_direct[:, n].^(param[:nu]-1)
            vecneg = (1+param[:beta]) .* costneg .* param[:m][n] .* Qin_indirect[:, n].^(param[:nu]-1)
    
            JQpos[:, (graph.ndeg*(n-1)+1):graph.ndeg*n] = Apos .* repeat(vecpos', graph.J)
            JQneg[:, (graph.ndeg*(n-1)+1):graph.ndeg*n] = Aneg .* repeat(vecneg', graph.J)
        end
    
        J2 = [zeros(graph.J, 1), Matrix(I, graph.J, graph.J), JD, JQpos, JQneg, zeros(graph.J, graph.J)]
    
        # Compute jacobian of flow conservation constraint
    
        # part related to L
        JL = zeros(graph.J*param[:N], graph.J)
        for n in 1:param[:N]
            id = (n-1)*graph.J+1:n*graph.J
            JL[id, :] = -Diagonal(param[:a] .* param[:Zjn][:, n] .* Lj.^(param[:a]-1))
        end
    
        J3 = [zeros(graph.J*param[:N], 1+graph.J), Matrix(I, graph.J*param[:N], graph.J*param[:N]), kron(Matrix(I, param[:N], param[:N]), A), kron(Matrix(I, param[:N], param[:N]), -A), JL]
    
        # Compute jacobian of total labor availability
        J4 = [zeros(1, 1+graph.J+graph.J*param[:N]+2*graph.ndeg*param[:N]), ones(1, graph.J)]
    
        # return full jacobian
        return sparse(vcat(J1, J2, J3, J4))
    end

    
    function sub2ind(sz::Int, rows::AbstractArray, cols::AbstractArray)
        return (rows .- 1) .* sz .+ cols
    end

    # This code has been optimized to exploit the sparse structure of the hessian.
    function hessian(x, auxdata, sigma_IPOPT, lambda_IPOPT)
        param = auxdata.param
        graph = auxdata.graph
        Apos = auxdata.Apos
        Aneg = auxdata.Aneg
        kappa_ex = auxdata.kappa_ex
    
        # -----------------
        # Recover variables
    
        u = x[1]
        Cj = x[2:1+graph.J]
        Djn = reshape(x[2+graph.J:1+graph.J+graph.J*param[:N]], (graph.J, param[:N]))
        Dj = sum(Djn.^((param[:sigma]-1)/param[:sigma]), dims=2).^((param[:sigma])/(param[:sigma]-1))  # total availability of final good, not consumption!
        Qin_direct = reshape(x[2+graph.J+graph.J*param[:N]:1+graph.J+graph.J*param[:N]+graph.ndeg*param[:N]], (graph.ndeg, param[:N]))
        Qin_indirect = reshape(x[2+graph.J+graph.J*param[:N]+graph.ndeg*param[:N]:1+graph.J+graph.J*param[:N]+2*graph.ndeg*param[:N]], (graph.ndeg, param[:N]))
        Lj = x[2+graph.J+graph.J*param[:N]+2*graph.ndeg*param[:N]:end]
    
        omega = lambda_IPOPT[1:graph.J]
        lambda = lambda_IPOPT[(graph.J+1):(2*graph.J)]
        Pjn = reshape(lambda_IPOPT[(2*graph.J+1):(2*graph.J+graph.J*param[:N])], (graph.J, param[:N]))
    
        # preallocation of sparse matrix for maximum speed
        sz = 1 + graph.J + graph.J*param[:N] + 2*graph.ndeg*param[:N] + graph.J
        H = spzeros(sz, sz)
    
        # -----------------------------
        # Part of Hessian related to Lu
    
        id = 1:graph.J
        x = 2+graph.J+graph.J*param[:N]+2*graph.ndeg*param[:N]
        H[sub2ind(sz, id, ones(length(id)))] = omega
    
        # -----------------------------------------
        # Diagonal part of Hessian respective to Cj
    
        HC = -omega .* param[:usecond].(Cj, param[:Hj])
        H[sub2ind(sz, (x+1):(x+graph.J), (x+1):(x+graph.J))] = HC
    
        # -------------------------------------
        # Diagonal of Hessian respective to Djn
    
        HDdiag = repeat(lambda ./ param[:sigma] .* Dj.^(1/param[:sigma]), outer=(1, param[:N])) .* Djn.^(-1/param[:sigma] - 1)
        H[sub2ind(sz, (x+graph.J+1):(x+graph.J+graph.J*param[:N]), (x+graph.J+1):(x+graph.J+graph.J*param[:N]))] = HDdiag[:]
    
        # -------------------------------------
        # Diagonal of Hessian respective to Qin
    
        matm = repeat(param[:m]', outer=(graph.ndeg, 1))
        costpos = sum(matm .* Qin_direct.^param[:nu], dims=2)  # ndeg x 1 vector of congestion cost
        costneg = sum(matm .* Qin_indirect.^param[:nu], dims=2)
    
        if param[:nu] > 1  # if nu=1, diagonal term disappears
            matpos = repeat((1+param[:beta]) * (param[:nu]-1) * (Apos' * lambda) .* costpos.^((param[:beta]+1)/param[:nu]-1) .* graph.delta_tau_spillover_ex_direct ./ kappa_ex, outer=(1, param[:N])) .* repeat(param[:m]', outer=(graph.ndeg, 1)) .* Qin_direct.^(param[:nu]-2)
            matneg = repeat((1+param[:beta]) * (param[:nu]-1) * (Aneg' * lambda) .* costneg.^((param[:beta]+1)/param[:nu]-1) .* graph.delta_tau_spillover_ex_indirect ./ kappa_ex, outer=(1, param[:N])) .* repeat(param[:m]', outer=(graph.ndeg,1)]);
            xpos = x+graph.J*param[:N]+1
            ypos = xpos
            H[sub2ind(sz, (xpos):(xpos+graph.ndeg*param[:N]-1), (ypos):(ypos+graph.ndeg*param[:N]-1))] = matpos[:]
    
            xneg = xpos+graph.ndeg*param[:N]
            yneg = xneg
            H[sub2ind(sz, (xneg):(xneg+graph.ndeg*param[:N]-1), (yneg):(yneg+graph.ndeg*param[:N]-1))] = matneg[:]
        end
    
        # ------------------------------------
        # Diagonal of Hessian respective to Lj
    
        HLL = -param[:a]*(param[:a]-1) .* sum(Pjn .* param[:Zjn], dims=2) .* Lj.^(param[:a]-2)
        H[sub2ind(sz, end-graph.J+1:end, end-graph.J+1:end)] = HLL
    
        # -----------------
        # Nondiagonal parts
    
        for n in 1:param[:N]  # row
            for m in 1:param[:N]  # col
                # -----------------
                # Respective to Djn
    
                HDnondiag = -lambda ./ param[:sigma] .* Dj.^(-(param[:sigma]-2)/param[:sigma]) .* Djn[:, n].^(-1/param[:sigma]) .* Djn[:, m].^(-1/param[:sigma])
    
                x = 2+graph.J + graph.J*(n-1)
                y = 2+graph.J + graph.J*(m-1)
                id = (x+1):(x+graph.J)
                H[sub2ind(sz, id, id)] .+= HDnondiag
    
                # -----------------
                # Respective to Qin
    
                vecpos = (1+param[:beta])*((1+param[:beta])/param[:nu]-1)*param[:nu]*(Apos' * lambda) .* costpos.^((param[:beta]+1)/param[:nu]-2) .* graph.delta_tau_spillover_ex_direct ./ kappa_ex .* param[:m][n] .* Qin_direct[:, n].^(param[:nu]-1) .* param[:m][m] .* Qin_direct[:, m].^(param[:nu]-1)
    
                vecneg = (1+param[:beta])*((1+param[:beta])/param[:nu]-1)*param[:nu]*(Aneg' * lambda) .* costneg.^((param[:beta]+1)/param[:nu]-2) .* graph.delta_tau_spillover_ex_indirect ./ kappa_ex .* param[:m][n] .* Qin_indirect[:, n].^(param[:nu]-1) .* param[:m][m] .* Qin_indirect[:, m].^(param[:nu]-1)
    
                xpos = x+graph.J*param[:N]*param[:N]+graph.ndeg*(n-1)
                ypos = xpos
                xneg = xpos+graph.ndeg*param[:N]*param[:N]
                yneg = xneg
                id = xpos+1:xpos+graph.ndeg
                H[sub2ind(sz, id, id)] .+= vecpos
                H[sub2ind(sz, id+graph.ndeg*param[:N], id+graph.ndeg*param[:N])] .+= vecneg
            end
        end
    
        # -------------------
        # Return full hessian
    
        return H
    end

    options = Dict(
        :lb => vcat(-Inf, fill(1e-6, graph.J), fill(1e-6, graph.J * param.N), fill(1e-6, 2 * graph.ndeg * param.N), fill(1e-8, graph.J)),
        :ub => fill(Inf, length(x0)),
        :cl => vcat(fill(-Inf, graph.J * (2 + param.N)), 0),
        :cu => zeros(graph.J * (2 + param.N) + 1),
        :max_iter => 2000,
        :print_level => verbose ? 5 : 0
    )

    nlp = createProblem(x0, objective, gradient, constraints, jacobian, hessian, options)
    x, info = solveProblem(nlp)

    results = recover_allocation_cgc_inefficiency(x, auxdata)
    results.omegaj = lambda_IPOPT[1:graph.J]
    results.Pjn = reshape(lambda_IPOPT[2 * graph.J + 1:2 * graph.J + graph.J * param.N], graph.J, param.N)
    results.PCj = sum(results.Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma))
    results.PHj = param.alpha / (1 - param.alpha) .* results.PCj .* results.cj ./ results.hj
    results.W = lambda_IPOPT[2 * graph.J + graph.J * param.N + 1]

    return results, info, x
end

function recover_allocation_cgc_inefficiency(x, auxdata)
    graph = auxdata.graph
    param = auxdata.param
    # kappa=auxdata.kappa

    # Total economy welfare
    welfare = x[1]

    # Aggregate consumption 
    Cj = x[2:1+graph.J]

    # Population
    Lj = x[end-graph.J+1:end]

    # Consumption per capita 
    cj = Cj ./ Lj
    cj[Lj .== 0] .= 0 # catch errors for non-populated places

    # Non tradable good per capita
    hj = param[:Hj] ./ Lj
    hj[Lj .== 0] .= 0

    # Vector of welfare per location
    uj = ((cj / param[:alpha]) .^ param[:alpha] .* (hj / (1 - param[:alpha])) .^ (1 - param[:alpha])) .^ (1 - param[:rho]) / (1 - param[:rho])

    # Working population 
    Ljn = (param[:Zjn] .> 0) .* repeat(Lj, 1, param[:N])

    # Production
    Yjn = param[:Zjn] .* (repeat(Lj, 1, param[:N]) .^ param[:a])

    # Domestic absorption of good n
    Djn = max.(0, reshape(x[1+graph.J+1:1+graph.J+graph.J*param[:N]], (graph.J, param[:N])))

    # Total availability of final good 
    Dj = sum(Djn .^ ((param[:sigma] - 1) / param[:sigma]), dims=2) .^ (param[:sigma] / (param[:sigma] - 1)) # total availability of final good, not consumption!

    # Trade flows
    Qin_direct = reshape(x[1+graph.J+graph.J*param[:N]+1:1+graph.J+graph.J*param[:N]+graph.ndeg*param[:N]], (graph.ndeg, param[:N]))
    Qin_indirect = reshape(x[1+graph.J+graph.J*param[:N]+graph.ndeg*param[:N]+1:end], (graph.ndeg, param[:N]))

    Qin = zeros(graph.ndeg, param[:N])

    for i = 1:param[:N]*graph.ndeg
        if Qin_direct[i] > Qin_indirect[i]
            Qin[i] = Qin_direct[i] - Qin_indirect[i]
        else
            Qin[i] = Qin_direct[i] - Qin_indirect[i]
        end
    end
    Qin = reshape(Qin, (graph.ndeg, param[:N]))

    # recover the Q's
    Qjkn = zeros(graph.J, graph.J, param[:N])
    id = 1
    for i = 1:graph.J
        for j in graph.nodes[i].neighbors
            if j > i
                Qjkn[i, j, :] = max.(Qin[id, :], 0)
                Qjkn[j, i, :] = max.(-Qin[id, :], 0)
                id += 1
            end
        end
    end

    # Return a named tuple with the results
    return (welfare=welfare, Cj=Cj, Lj=Lj, cj=cj, hj=hj, uj=uj, Ljn=Ljn, Yjn=Yjn, Djn=Djn, Dj=Dj, Qin=Qin, Qjkn=Qjkn)
end
    


