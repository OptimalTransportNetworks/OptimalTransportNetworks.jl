# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4
#
# Direct-Ipopt port of MATLAB solve_allocation_cgc.m: fixed labor, cross-good
# congestion, Armington (<=1 good per location). Used when the dual is
# unavailable (beta > 1 or duality off). Split directional flows are required
# because the congestion cost aggregates goods per direction.
#
# Variable layout (x):
#   x = [cj(J); Djn(J*N); Qin_direct(ndeg*N); Qin_indirect(ndeg*N)]
# Constraint layout (g):
#   g = [cons_C(J); cons_Q(J*N)]   (final-good availability; balanced flows)

"""
    solve_allocation_cgc(x0, auxdata, verbose=true) -> (results::Dict, status, x)

Solve the fixed-labor, cross-good-congestion allocation (Armington) via direct
Ipopt (primal). Used when the dual approach is not available.
"""
function solve_allocation_cgc(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg

    if any(vec(sum(graph.Zjn .> 0, dims=2)) .> 1)
        error("solve_allocation_cgc only supports one good at most per location.")
    end

    n = J + J * N + 2 * ndeg * N
    if x0 === nothing || isempty(x0)
        x0 = vcat(fill(1e-6 / graph.Lj[1], J), fill(1e-6, J * N), fill(1e-8, 2 * ndeg * N))
    end

    cache = get(auxdata, :struct_cache, nothing)
    js, hs = get_structs(auxdata, () -> jacobian_structure_cgc(auxdata), () -> hessian_structure_cgc(auxdata))
    saux = (auxdata..., jac_struct = js, hess_struct = hs)
    nnz_jac = length(js[1])
    nnz_hess = length(hs[1])

    obj = (x) -> objective_cgc(x, saux)
    grad = (x, grad_f) -> gradient_cgc(x, grad_f, saux)
    cons = (x, g) -> constraints_cgc(x, g, saux)
    jac = (x, rows, cols, values) -> jacobian_cgc(x, rows, cols, values, saux)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_cgc(x, rows, cols, obj_factor, lambda, values, saux)

    m = J + J * N
    x_L = vcat(fill(1e-8, J), fill(1e-6, J * N), fill(1e-8, 2 * ndeg * N))
    x_U = vcat(fill(Inf, J), fill(Inf, J * N), fill(Inf, 2 * ndeg * N))
    g_L = fill(-Inf, J + J * N)
    g_U = fill(0.0, J + J * N)

    x, status, mult_g = run_ipopt_primal(cache, :cgc, saux, n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                                         obj, cons, grad, jac, hess, x0, param, verbose)
    results = recover_allocation_cgc(x, mult_g, auxdata)
    return namedtuple_to_dict(results), status, x
end

# ----- index helpers -----
@inline n_cgc(J, N, ndeg) = J + J * N + 2 * ndeg * N
@inline col_cj_cgc(j) = j
@inline col_D_cgc(j, n, J) = J + (n - 1) * J + j
@inline col_Qd_cgc(i, n, J, N, ndeg) = J + J * N + (n - 1) * ndeg + i
@inline col_Qi_cgc(i, n, J, N, ndeg) = J + J * N + ndeg * N + (n - 1) * ndeg + i
@inline row_C_cgc(j) = j
@inline row_Q_cgc(j, n, J) = J + (n - 1) * J + j

# aggregate availability Dj per location
function _Dj_cgc(Djn, psigma)
    return dropdims(sum(Djn .^ psigma, dims=2), dims=2) .^ (1 / psigma)
end

# ----- objective (nonlinear in cj): f = -sum omegaj*Lj*u(cj,hj) -----
function objective_cgc(x, auxdata)
    graph = auxdata.graph; param = auxdata.param
    cj = view(x, 1:graph.J)
    return -sum(graph.omegaj .* graph.Lj .* param.u.(cj, graph.hj))
end

function gradient_cgc(x::Vector{Float64}, grad_f::Vector{Float64}, auxdata)
    graph = auxdata.graph; param = auxdata.param
    cj = view(x, 1:graph.J)
    fill!(grad_f, 0.0)
    @inbounds for j in 1:graph.J
        grad_f[j] = -graph.omegaj[j] * graph.Lj[j] * param.uprime(cj[j], graph.hj[j])
    end
    return
end

# ----- constraints -----
function constraints_cgc(x::Vector{Float64}, g::Vector{Float64}, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, a = param.m, param.nu, param.beta, param.a
    psigma = (param.sigma - 1) / param.sigma
    costfac = (beta + 1) / nu

    cj = view(x, 1:J)
    Djn = reshape(view(x, J + 1:J + J * N), J, N)
    Qd = reshape(view(x, J + J * N + 1:J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, J + J * N + ndeg * N + 1:J + J * N + 2 * ndeg * N), ndeg, N)
    Dj = _Dj_cgc(Djn, psigma)

    Sd = vec(sum((Qd .^ nu) .* m', dims=2))   # ndeg
    Si = vec(sum((Qi .^ nu) .* m', dims=2))
    cost_d = Apos * (Sd .^ costfac ./ kappa_ex)
    cost_i = Aneg * (Si .^ costfac ./ kappa_ex)
    @inbounds for j in 1:J
        g[row_C_cgc(j)] = cj[j] * graph.Lj[j] + cost_d[j] + cost_i[j] - Dj[j]
    end
    @inbounds for nn in 1:N
        netflow = A * Qd[:, nn] - A * Qi[:, nn]
        for j in 1:J
            g[row_Q_cgc(j, nn, J)] = Djn[j, nn] + netflow[j] - graph.Zjn[j, nn] * graph.Lj[j]^a
        end
    end
    return
end

# ----- Jacobian structure -----
function jacobian_structure_cgc(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    # cons_C row j
    for j in 1:J
        push_rc!(j, col_cj_cgc(j))
        for nn in 1:N; push_rc!(j, col_D_cgc(j, nn, J)); end
        for nn in 1:N, i in 1:ndeg
            if A[j, i] > 0; push_rc!(j, col_Qd_cgc(i, nn, J, N, ndeg)); end
            if A[j, i] < 0; push_rc!(j, col_Qi_cgc(i, nn, J, N, ndeg)); end
        end
    end
    # cons_Q row (j,n)
    for nn in 1:N, j in 1:J
        r = row_Q_cgc(j, nn, J)
        push_rc!(r, col_D_cgc(j, nn, J))
        for i in 1:ndeg
            if A[j, i] != 0
                push_rc!(r, col_Qd_cgc(i, nn, J, N, ndeg))
                push_rc!(r, col_Qi_cgc(i, nn, J, N, ndeg))
            end
        end
    end
    return struct_from_triplets(rows, cols, J + J * N, n_cgc(J, N, ndeg))
end

# ----- Jacobian values -----
function jacobian_cgc(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                      values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.jac_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, sigma = param.m, param.nu, param.beta, param.sigma
    psigma = (sigma - 1) / sigma
    costfac = (beta + 1) / nu

    Djn = reshape(view(x, J + 1:J + J * N), J, N)
    Qd = reshape(view(x, J + J * N + 1:J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, J + J * N + ndeg * N + 1:J + J * N + 2 * ndeg * N), ndeg, N)
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2))
    Si = vec(sum((Qi .^ nu) .* m', dims=2))
    cpos = Sd .^ (costfac - 1) ./ kappa_ex   # ndeg
    cneg = Si .^ (costfac - 1) ./ kappa_ex

    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (push!(Ir, r); push!(Jc, c); push!(V, v))
    # cons_C
    @inbounds for j in 1:J
        addv!(j, col_cj_cgc(j), graph.Lj[j])
        for nn in 1:N
            addv!(j, col_D_cgc(j, nn, J), -Dj[j]^(1 / sigma) * Djn[j, nn]^(-1 / sigma))
        end
        for nn in 1:N, i in 1:ndeg
            if A[j, i] > 0
                addv!(j, col_Qd_cgc(i, nn, J, N, ndeg), Apos[j, i] * (1 + beta) * cpos[i] * m[nn] * Qd[i, nn]^(nu - 1))
            end
            if A[j, i] < 0
                addv!(j, col_Qi_cgc(i, nn, J, N, ndeg), Aneg[j, i] * (1 + beta) * cneg[i] * m[nn] * Qi[i, nn]^(nu - 1))
            end
        end
    end
    # cons_Q
    @inbounds for nn in 1:N, j in 1:J
        r = row_Q_cgc(j, nn, J)
        addv!(r, col_D_cgc(j, nn, J), 1.0)
        for i in 1:ndeg
            if A[j, i] != 0
                addv!(r, col_Qd_cgc(i, nn, J, N, ndeg), A[j, i])
                addv!(r, col_Qi_cgc(i, nn, J, N, ndeg), -A[j, i])
            end
        end
    end
    fill_values_from_triplets!(values, auxdata.jac_struct, Ir, Jc, V, J + J * N, n_cgc(J, N, ndeg))
    return
end

# ----- Hessian structure -----
function hessian_structure_cgc(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    for j in 1:J; push_rc!(col_cj_cgc(j), col_cj_cgc(j)); end
    for j in 1:J, nn in 1:N, nd in 1:N
        r = col_D_cgc(j, nn, J); c = col_D_cgc(j, nd, J); r >= c && push_rc!(r, c)
    end
    for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qd_cgc(i, nn, J, N, ndeg); c = col_Qd_cgc(i, nd, J, N, ndeg); r >= c && push_rc!(r, c)
    end
    for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qi_cgc(i, nn, J, N, ndeg); c = col_Qi_cgc(i, nd, J, N, ndeg); r >= c && push_rc!(r, c)
    end
    nt = n_cgc(J, N, ndeg)
    return struct_from_triplets(rows, cols, nt, nt)
end

# ----- Hessian values -----
function hessian_cgc(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                     obj_factor::Float64, lambda::Vector{Float64},
                     values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.hess_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, sigma = param.m, param.nu, param.beta, param.sigma
    psigma = (sigma - 1) / sigma
    costfac = (beta + 1) / nu

    cj = view(x, 1:J)
    Djn = reshape(view(x, J + 1:J + J * N), J, N)
    Qd = reshape(view(x, J + J * N + 1:J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, J + J * N + ndeg * N + 1:J + J * N + 2 * ndeg * N), ndeg, N)
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2))
    Si = vec(sum((Qi .^ nu) .* m', dims=2))

    lamC = lambda[1:J]               # final-good (cons_C) multipliers
    lamApos = Apos' * lamC           # ndeg (origin-node multiplier, direct)
    lamAneg = Aneg' * lamC           # ndeg (origin-node multiplier, indirect)

    nt = n_cgc(J, N, ndeg)
    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (r >= c && (push!(Ir, r); push!(Jc, c); push!(V, v)))

    # cj diagonal (objective)
    @inbounds for j in 1:J
        addv!(col_cj_cgc(j), col_cj_cgc(j), -obj_factor * graph.omegaj[j] * graph.Lj[j] * param.usecond(cj[j], graph.hj[j]))
    end
    # Djn block per location (constraint cons_C, weighted by lamC)
    @inbounds for j in 1:J
        for nn in 1:N, nd in 1:N
            r = col_D_cgc(j, nn, J); c = col_D_cgc(j, nd, J)
            r >= c || continue
            val = -lamC[j] / sigma * Dj[j]^(-(sigma - 2) / sigma) * Djn[j, nn]^(-1 / sigma) * Djn[j, nd]^(-1 / sigma)
            if nn == nd
                val += lamC[j] / sigma * Dj[j]^(1 / sigma) * Djn[j, nn]^(-1 / sigma - 1)
            end
            addv!(r, c, val)
        end
    end
    # Qin_direct block per edge (constraint cons_C, weighted by lamApos)
    @inbounds for i in 1:ndeg
        for nn in 1:N, nd in 1:N
            r = col_Qd_cgc(i, nn, J, N, ndeg); c = col_Qd_cgc(i, nd, J, N, ndeg)
            r >= c || continue
            val = (1 + beta) * (costfac - 1) * nu * lamApos[i] * Sd[i]^(costfac - 2) / kappa_ex[i] *
                  m[nn] * Qd[i, nn]^(nu - 1) * m[nd] * Qd[i, nd]^(nu - 1)
            if nn == nd && nu > 1
                val += (1 + beta) * (nu - 1) * lamApos[i] * Sd[i]^(costfac - 1) / kappa_ex[i] * m[nn] * Qd[i, nn]^(nu - 2)
            end
            addv!(r, c, val)
        end
    end
    # Qin_indirect block per edge (constraint cons_C, weighted by lamAneg)
    @inbounds for i in 1:ndeg
        for nn in 1:N, nd in 1:N
            r = col_Qi_cgc(i, nn, J, N, ndeg); c = col_Qi_cgc(i, nd, J, N, ndeg)
            r >= c || continue
            val = (1 + beta) * (costfac - 1) * nu * lamAneg[i] * Si[i]^(costfac - 2) / kappa_ex[i] *
                  m[nn] * Qi[i, nn]^(nu - 1) * m[nd] * Qi[i, nd]^(nu - 1)
            if nn == nd && nu > 1
                val += (1 + beta) * (nu - 1) * lamAneg[i] * Si[i]^(costfac - 1) / kappa_ex[i] * m[nn] * Qi[i, nn]^(nu - 2)
            end
            addv!(r, c, val)
        end
    end
    fill_values_from_triplets!(values, auxdata.hess_struct, Ir, Jc, V, nt, nt)
    return
end

# ----- recover allocation -----
function recover_allocation_cgc(x, mult_g, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    psigma = (param.sigma - 1) / param.sigma

    cj = x[1:J]
    Djn = max.(0.0, reshape(x[J + 1:J + J * N], J, N))
    Qd = reshape(x[J + J * N + 1:J + J * N + ndeg * N], ndeg, N)
    Qi = reshape(x[J + J * N + ndeg * N + 1:J + J * N + 2 * ndeg * N], ndeg, N)
    Dj = _Dj_cgc(Djn, psigma)
    Lj = graph.Lj
    Cj = cj .* Lj
    Ljn = (graph.Zjn .> 0) .* Lj
    Yjn = graph.Zjn .* Lj .^ param.a
    uj = param.u.(cj, graph.hj)
    Qin = Qd - Qi
    Qjkn = gen_network_flows(Qin, graph, N)
    Pjn = reshape(mult_g[J + 1:J + J * N], J, N)
    PCj = dropdims(sum(Pjn .^ (1 - param.sigma), dims=2), dims=2) .^ (1 / (1 - param.sigma))

    return (
        welfare = sum(graph.omegaj .* Lj .* param.u.(cj, graph.hj)),
        Yjn = Yjn, Yj = dropdims(sum(Yjn, dims=2), dims=2),
        Cjn = Djn, Cj = Cj, Djn = Djn, Dj = Dj, Ljn = Ljn, Lj = Lj,
        cj = cj, hj = graph.hj, uj = uj,
        Pjn = Pjn, PCj = PCj, Qin = Qin, Qjkn = Qjkn,
    )
end
