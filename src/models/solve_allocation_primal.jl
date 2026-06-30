# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4
#
# Direct-Ipopt port of MATLAB solve_allocation_primal.m: fixed labor, no
# cross-good congestion, Armington (<=1 good per location). Used when the dual
# is unavailable (beta > 1, where the dual is not twice differentiable, or
# duality disabled). Single signed flow Qin; labor fixed at graph.Lj.
#
# Variable layout (x):
#   x = [Cjn(J*N); Qin(ndeg*N)]
# Constraint layout (g):
#   g = [cons_Q(J*N)]   (balanced flows only)

"""
    solve_allocation_primal(x0, auxdata, verbose=true) -> (results::Dict, status, x)

Solve the fixed-labor, no-congestion allocation (Armington) via direct Ipopt
(primal). Used when the dual approach is not available.
"""
function solve_allocation_primal(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg

    if any(vec(sum(graph.Zjn .> 0, dims=2)) .> 1)
        error("solve_allocation_primal only supports one good at most per location.")
    end

    n = J * N + ndeg * N
    if x0 === nothing || isempty(x0)
        x0 = vcat(fill(1e-6, J * N), zeros(ndeg * N))
    end

    cache = get(auxdata, :struct_cache, nothing)
    js, hs = get_structs(auxdata, () -> jacobian_structure_primal(auxdata), () -> hessian_structure_primal(auxdata))
    saux = (auxdata..., jac_struct = js, hess_struct = hs)
    nnz_jac = length(js[1])
    nnz_hess = length(hs[1])

    obj = (x) -> objective_primal(x, saux)
    grad = (x, grad_f) -> gradient_primal(x, grad_f, saux)
    cons = (x, g) -> constraints_primal(x, g, saux)
    jac = (x, rows, cols, values) -> jacobian_primal(x, rows, cols, values, saux)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_primal(x, rows, cols, obj_factor, lambda, values, saux)

    m = J * N
    x_L = vcat(fill(1e-8, J * N), fill(-Inf, ndeg * N))
    x_U = vcat(fill(Inf, J * N), fill(Inf, ndeg * N))
    g_L = fill(-Inf, J * N)
    g_U = fill(0.0, J * N)

    x, status, mult_g = run_ipopt_primal(cache, :primal, saux, n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                                         obj, cons, grad, jac, hess, x0, param, verbose)
    results = recover_allocation_primal(x, mult_g, auxdata)
    return namedtuple_to_dict(results), status, x
end

# ----- index helpers -----
@inline n_primal(J, N, ndeg) = J * N + ndeg * N
@inline col_C_pr(j, n, J) = (n - 1) * J + j
@inline col_Q_pr(i, n, J, N, ndeg) = J * N + (n - 1) * ndeg + i
@inline row_Q_pr(j, n, J) = (n - 1) * J + j

# helper: aggregate consumption Cj and per-capita cj
function _cj_primal(Cjn, graph, psigma)
    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)
    cj = ifelse.(graph.Lj .== 0, 0.0, Cj ./ graph.Lj)
    return Cj, cj
end

# ----- objective (nonlinear): f = -sum omegaj*Lj*u(cj,hj) -----
function objective_primal(x, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N = graph.J, param.N
    psigma = (param.sigma - 1) / param.sigma
    Cjn = reshape(view(x, 1:J * N), J, N)
    _, cj = _cj_primal(Cjn, graph, psigma)
    return -sum(graph.omegaj .* graph.Lj .* param.u.(cj, graph.hj))
end

function gradient_primal(x::Vector{Float64}, grad_f::Vector{Float64}, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N = graph.J, param.N
    sigma = param.sigma
    psigma = (sigma - 1) / sigma
    Cjn = reshape(view(x, 1:J * N), J, N)
    Cj, cj = _cj_primal(Cjn, graph, psigma)
    fill!(grad_f, 0.0)
    @inbounds for nn in 1:N, j in 1:J
        grad_f[col_C_pr(j, nn, J)] = -graph.omegaj[j] * param.uprime(cj[j], graph.hj[j]) *
                                     Cjn[j, nn]^(-1 / sigma) * Cj[j]^(1 / sigma)
    end
    return
end

# ----- constraints (single signed Qin, cost charged at origin) -----
function constraints_primal(x::Vector{Float64}, g::Vector{Float64}, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    psigma = (param.sigma - 1) / param.sigma
    beta, a = param.beta, param.a
    Cjn = reshape(view(x, 1:J * N), J, N)
    Qin = reshape(view(x, J * N + 1:J * N + ndeg * N), ndeg, N)

    @inbounds for nn in 1:N
        q = Qin[:, nn]
        cost = abs.(q) .^ (1 + beta) ./ kappa_ex
        pos = q .> 0
        cost_at = Apos * (cost .* pos) + Aneg * (cost .* .!pos)
        flow = A * q + cost_at
        for j in 1:J
            g[row_Q_pr(j, nn, J)] = Cjn[j, nn] + flow[j] - graph.Zjn[j, nn] * graph.Lj[j]^a
        end
    end
    return
end

# ----- Jacobian structure -----
function jacobian_structure_primal(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    for nn in 1:N, j in 1:J
        r = row_Q_pr(j, nn, J)
        push_rc!(r, col_C_pr(j, nn, J))
        for i in 1:ndeg
            A[j, i] != 0 && push_rc!(r, col_Q_pr(i, nn, J, N, ndeg))
        end
    end
    return struct_from_triplets(rows, cols, J * N, n_primal(J, N, ndeg))
end

# ----- Jacobian values -----
function jacobian_primal(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                         values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.jac_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    beta = param.beta
    Qin = reshape(view(x, J * N + 1:J * N + ndeg * N), ndeg, N)

    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (push!(Ir, r); push!(Jc, c); push!(V, v))
    @inbounds for nn in 1:N, j in 1:J
        r = row_Q_pr(j, nn, J)
        addv!(r, col_C_pr(j, nn, J), 1.0)
        for i in 1:ndeg
            if A[j, i] != 0
                q = Qin[i, nn]
                Morig = q > 0 ? Apos[j, i] : Aneg[j, i]
                dcost = (1 + beta) * Morig * sign(q) * abs(q)^beta / kappa_ex[i]
                addv!(r, col_Q_pr(i, nn, J, N, ndeg), A[j, i] + dcost)
            end
        end
    end
    fill_values_from_triplets!(values, auxdata.jac_struct, Ir, Jc, V, J * N, n_primal(J, N, ndeg))
    return
end

# ----- Hessian structure -----
function hessian_structure_primal(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    # Cjn block: same-location coupling across goods (lower triangle)
    for j in 1:J, nn in 1:N, nd in 1:N
        r = col_C_pr(j, nn, J); c = col_C_pr(j, nd, J)
        r >= c && push_rc!(r, c)
    end
    # Qin diagonal
    for nn in 1:N, i in 1:ndeg
        cc = col_Q_pr(i, nn, J, N, ndeg); push_rc!(cc, cc)
    end
    nt = n_primal(J, N, ndeg)
    return struct_from_triplets(rows, cols, nt, nt)
end

# ----- Hessian values: objective part (obj_factor) + constraint part (lambda) -----
function hessian_primal(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                        obj_factor::Float64, lambda::Vector{Float64},
                        values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.hess_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    sigma, beta = param.sigma, param.beta
    psigma = (sigma - 1) / sigma
    Cjn = reshape(view(x, 1:J * N), J, N)
    Qin = reshape(view(x, J * N + 1:J * N + ndeg * N), ndeg, N)
    Cj, cj = _cj_primal(Cjn, graph, psigma)
    Pjn = reshape(lambda[1:J * N], J, N)

    nt = n_primal(J, N, ndeg)
    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (r >= c && (push!(Ir, r); push!(Jc, c); push!(V, v)))

    # Cjn block (objective, weighted by obj_factor)
    @inbounds for j in 1:J
        up = param.uprime(cj[j], graph.hj[j])
        us = param.usecond(cj[j], graph.hj[j])
        # off-diagonal prefactor (same location, across goods)
        pref = -obj_factor * graph.omegaj[j] *
               ((1 / sigma) * up * Cj[j]^(2 / sigma - 1) + (1 / graph.Lj[j]) * us * Cj[j]^(2 / sigma))
        for nn in 1:N, nd in 1:N
            r = col_C_pr(j, nn, J); c = col_C_pr(j, nd, J)
            r >= c || continue
            val = pref * Cjn[j, nn]^(-1 / sigma) * Cjn[j, nd]^(-1 / sigma)
            if nn == nd
                val += obj_factor * (1 / sigma) * Cjn[j, nn]^(-1 / sigma - 1) *
                       graph.omegaj[j] * Cj[j]^(1 / sigma) * up
            end
            addv!(r, c, val)
        end
    end
    # Qin diagonal (constraint, weighted by lambda=Pjn; only if beta>0)
    if beta > 0
        @inbounds for nn in 1:N
            ApP = Apos' * Pjn[:, nn]; AnP = Aneg' * Pjn[:, nn]
            for i in 1:ndeg
                q = Qin[i, nn]; aq = abs(q)
                wgt = q > 0 ? ApP[i] : AnP[i]
                hpow = aq == 0.0 ? (beta == 1.0 ? 1.0 : 0.0) : aq^(beta - 1)
                cc = col_Q_pr(i, nn, J, N, ndeg)
                addv!(cc, cc, (1 + beta) * beta * hpow / kappa_ex[i] * wgt)
            end
        end
    end
    fill_values_from_triplets!(values, auxdata.hess_struct, Ir, Jc, V, nt, nt)
    return
end

# ----- recover allocation -----
function recover_allocation_primal(x, mult_g, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    psigma = (param.sigma - 1) / param.sigma

    Cjn = reshape(x[1:J * N], J, N)
    Qin = reshape(x[J * N + 1:J * N + ndeg * N], ndeg, N)
    Cj, cj = _cj_primal(Cjn, graph, psigma)
    Lj = graph.Lj
    Ljn = (graph.Zjn .> 0) .* Lj
    Yjn = graph.Zjn .* Lj .^ param.a
    hj = ifelse.(Lj .== 0, 0.0, graph.Hj ./ Lj)
    uj = param.u.(cj, hj)
    Qjkn = gen_network_flows(Qin, graph, N)
    Pjn = reshape(mult_g[1:J * N], J, N)
    PCj = dropdims(sum(Pjn .^ (1 - param.sigma), dims=2), dims=2) .^ (1 / (1 - param.sigma))

    return (
        welfare = sum(graph.omegaj .* Lj .* param.u.(cj, graph.hj)),
        Yjn = Yjn, Yj = dropdims(sum(Yjn, dims=2), dims=2),
        Cjn = Cjn, Cj = Cj, Ljn = Ljn, Lj = Lj,
        cj = cj, hj = hj, uj = uj,
        Pjn = Pjn, PCj = PCj, Qin = Qin, Qjkn = Qjkn,
    )
end
