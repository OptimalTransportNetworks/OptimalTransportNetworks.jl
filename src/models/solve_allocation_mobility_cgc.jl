# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4
#
# Direct-Ipopt port of MATLAB solve_allocation_mobility_cgc.m: full labor
# mobility, cross-good congestion, Armington (<=1 good per location). Combines
# the cgc final-good/flow structure with the mobility wrapper (utility level u,
# aggregate consumption Cj, labor Lj, total-labor constraint).
#
# Variable layout (x):
#   x = [u; Cj(J); Djn(J*N); Qin_direct(ndeg*N); Qin_indirect(ndeg*N); Lj(J)]
# Constraint layout (g):
#   g = [cons_u(J); cons_C(J); cons_Q(J*N); cons_L(1)]

"""
    solve_allocation_mobility_cgc(x0, auxdata, verbose=true) -> (results::Dict, status, x)

Solve the full-mobility, cross-good-congestion allocation (Armington) via direct Ipopt.
"""
function solve_allocation_mobility_cgc(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg

    if any(vec(sum(graph.Zjn .> 0, dims=2)) .> 1)
        error("solve_allocation_mobility_cgc only supports one good at most per location.")
    end

    n = 1 + J + J * N + 2 * ndeg * N + J
    if x0 === nothing || isempty(x0)
        x0 = vcat(0.0, fill(1e-6 * J, J), fill(1e-6, J * N), fill(1e-6, 2 * ndeg * N), fill(1.0 / J, J))
    end

    cache = get(auxdata, :struct_cache, nothing)
    js, hs = get_structs(auxdata, () -> jacobian_structure_mcgc(auxdata), () -> hessian_structure_mcgc(auxdata))
    saux = (auxdata..., jac_struct = js, hess_struct = hs)
    nnz_jac = length(js[1])
    nnz_hess = length(hs[1])

    obj = (x) -> -x[1]
    grad = (x, grad_f) -> (fill!(grad_f, 0.0); grad_f[1] = -1.0; nothing)
    cons = (x, g) -> constraints_mcgc(x, g, saux)
    jac = (x, rows, cols, values) -> jacobian_mcgc(x, rows, cols, values, saux)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_mcgc(x, rows, cols, obj_factor, lambda, values, saux)

    m = J + J + J * N + 1
    x_L = vcat(-Inf, fill(1e-6, J), fill(1e-6, J * N), fill(1e-6, 2 * ndeg * N), fill(1e-8, J))
    x_U = vcat(Inf, fill(Inf, J), fill(Inf, J * N), fill(Inf, 2 * ndeg * N), fill(Inf, J))
    g_L = vcat(fill(-Inf, J + J + J * N), 0.0)
    g_U = vcat(fill(0.0, J + J + J * N), 0.0)

    x, status, mult_g = run_ipopt_primal(cache, :mobility_cgc, saux, n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                                         obj, cons, grad, jac, hess, x0, param, verbose)
    results = recover_allocation_mcgc(x, mult_g, auxdata)
    return namedtuple_to_dict(results), status, x
end

# ----- index helpers -----
@inline n_mcgc(J, N, ndeg) = 1 + J + J * N + 2 * ndeg * N + J
@inline col_Cj_mc(j, J) = 1 + j
@inline col_D_mc(j, n, J) = 1 + J + (n - 1) * J + j
@inline col_Qd_mc(i, n, J, N, ndeg) = 1 + J + J * N + (n - 1) * ndeg + i
@inline col_Qi_mc(i, n, J, N, ndeg) = 1 + J + J * N + ndeg * N + (n - 1) * ndeg + i
@inline col_L_mc(j, J, N, ndeg) = 1 + J + J * N + 2 * ndeg * N + j
@inline row_u_mc(j) = j
@inline row_C_mc(j, J) = J + j
@inline row_Q_mc(j, n, J) = 2 * J + (n - 1) * J + j
@inline row_L_mc(J, N) = 2 * J + J * N + 1

# ----- constraints -----
function constraints_mcgc(x::Vector{Float64}, g::Vector{Float64}, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, a, alpha = param.m, param.nu, param.beta, param.a, param.alpha
    psigma = (param.sigma - 1) / param.sigma
    costfac = (beta + 1) / nu

    u = x[1]
    Cj = view(x, 2:1 + J)
    Djn = reshape(view(x, 2 + J:1 + J + J * N), J, N)
    Qd = reshape(view(x, 2 + J + J * N:1 + J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, 2 + J + J * N + ndeg * N:1 + J + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, 2 + J + J * N + 2 * ndeg * N:n_mcgc(J, N, ndeg))
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2))
    Si = vec(sum((Qi .^ nu) .* m', dims=2))
    cost_d = Apos * (Sd .^ costfac ./ kappa_ex)
    cost_i = Aneg * (Si .^ costfac ./ kappa_ex)

    @inbounds for j in 1:J
        g[row_u_mc(j)] = u * Lj[j] - felicity(Cj[j], graph.Hj[j], alpha)
        g[row_C_mc(j, J)] = Cj[j] + cost_d[j] + cost_i[j] - Dj[j]
    end
    @inbounds for nn in 1:N
        netflow = A * Qd[:, nn] - A * Qi[:, nn]
        for j in 1:J
            g[row_Q_mc(j, nn, J)] = Djn[j, nn] + netflow[j] - graph.Zjn[j, nn] * Lj[j]^a
        end
    end
    g[row_L_mc(J, N)] = sum(Lj) - 1.0
    return
end

# ----- Jacobian structure -----
function jacobian_structure_mcgc(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    # cons_u row j: u, Cj[j], Lj[j]
    for j in 1:J
        push_rc!(j, 1)
        push_rc!(j, col_Cj_mc(j, J))
        push_rc!(j, col_L_mc(j, J, N, ndeg))
    end
    # cons_C row j: Cj[j], Djn[j,n], Qd/Qi at incident edges
    for j in 1:J
        r = row_C_mc(j, J)
        push_rc!(r, col_Cj_mc(j, J))
        for nn in 1:N; push_rc!(r, col_D_mc(j, nn, J)); end
        for nn in 1:N, i in 1:ndeg
            if A[j, i] > 0; push_rc!(r, col_Qd_mc(i, nn, J, N, ndeg)); end
            if A[j, i] < 0; push_rc!(r, col_Qi_mc(i, nn, J, N, ndeg)); end
        end
    end
    # cons_Q row (j,n): Djn[j,n], Qd/Qi, Lj[j]
    for nn in 1:N, j in 1:J
        r = row_Q_mc(j, nn, J)
        push_rc!(r, col_D_mc(j, nn, J))
        for i in 1:ndeg
            if A[j, i] != 0
                push_rc!(r, col_Qd_mc(i, nn, J, N, ndeg))
                push_rc!(r, col_Qi_mc(i, nn, J, N, ndeg))
            end
        end
        push_rc!(r, col_L_mc(j, J, N, ndeg))
    end
    # cons_L row: all Lj
    for j in 1:J; push_rc!(row_L_mc(J, N), col_L_mc(j, J, N, ndeg)); end
    return struct_from_triplets(rows, cols, J + J + J * N + 1, n_mcgc(J, N, ndeg))
end

# ----- Jacobian values -----
function jacobian_mcgc(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                       values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.jac_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, a, sigma, alpha = param.m, param.nu, param.beta, param.a, param.sigma, param.alpha
    psigma = (sigma - 1) / sigma
    costfac = (beta + 1) / nu

    u = x[1]
    Cj = view(x, 2:1 + J)
    Djn = reshape(view(x, 2 + J:1 + J + J * N), J, N)
    Qd = reshape(view(x, 2 + J + J * N:1 + J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, 2 + J + J * N + ndeg * N:1 + J + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, 2 + J + J * N + 2 * ndeg * N:n_mcgc(J, N, ndeg))
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2)); Si = vec(sum((Qi .^ nu) .* m', dims=2))
    cpos = Sd .^ (costfac - 1) ./ kappa_ex; cneg = Si .^ (costfac - 1) ./ kappa_ex

    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (push!(Ir, r); push!(Jc, c); push!(V, v))
    # cons_u
    @inbounds for j in 1:J
        addv!(j, 1, Lj[j])
        addv!(j, col_Cj_mc(j, J), -felicity_c(Cj[j], graph.Hj[j], alpha))
        addv!(j, col_L_mc(j, J, N, ndeg), u)
    end
    # cons_C
    @inbounds for j in 1:J
        r = row_C_mc(j, J)
        addv!(r, col_Cj_mc(j, J), 1.0)
        for nn in 1:N
            addv!(r, col_D_mc(j, nn, J), -Dj[j]^(1 / sigma) * Djn[j, nn]^(-1 / sigma))
        end
        for nn in 1:N, i in 1:ndeg
            if A[j, i] > 0
                addv!(r, col_Qd_mc(i, nn, J, N, ndeg), Apos[j, i] * (1 + beta) * cpos[i] * m[nn] * Qd[i, nn]^(nu - 1))
            end
            if A[j, i] < 0
                addv!(r, col_Qi_mc(i, nn, J, N, ndeg), Aneg[j, i] * (1 + beta) * cneg[i] * m[nn] * Qi[i, nn]^(nu - 1))
            end
        end
    end
    # cons_Q
    @inbounds for nn in 1:N, j in 1:J
        r = row_Q_mc(j, nn, J)
        addv!(r, col_D_mc(j, nn, J), 1.0)
        for i in 1:ndeg
            if A[j, i] != 0
                addv!(r, col_Qd_mc(i, nn, J, N, ndeg), A[j, i])
                addv!(r, col_Qi_mc(i, nn, J, N, ndeg), -A[j, i])
            end
        end
        addv!(r, col_L_mc(j, J, N, ndeg), -a * graph.Zjn[j, nn] * Lj[j]^(a - 1))
    end
    # cons_L
    @inbounds for j in 1:J; addv!(row_L_mc(J, N), col_L_mc(j, J, N, ndeg), 1.0); end

    fill_values_from_triplets!(values, auxdata.jac_struct, Ir, Jc, V, J + J + J * N + 1, n_mcgc(J, N, ndeg))
    return
end

# ----- Hessian structure -----
function hessian_structure_mcgc(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    for j in 1:J; push_rc!(col_Cj_mc(j, J), col_Cj_mc(j, J)); end             # (Cj,Cj)
    for j in 1:J, nn in 1:N, nd in 1:N
        r = col_D_mc(j, nn, J); c = col_D_mc(j, nd, J); r >= c && push_rc!(r, c)
    end
    for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qd_mc(i, nn, J, N, ndeg); c = col_Qd_mc(i, nd, J, N, ndeg); r >= c && push_rc!(r, c)
    end
    for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qi_mc(i, nn, J, N, ndeg); c = col_Qi_mc(i, nd, J, N, ndeg); r >= c && push_rc!(r, c)
    end
    for j in 1:J
        cL = col_L_mc(j, J, N, ndeg)
        push_rc!(cL, 1)        # (Lj, u)
        push_rc!(cL, cL)       # (Lj, Lj)
    end
    nt = n_mcgc(J, N, ndeg)
    return struct_from_triplets(rows, cols, nt, nt)
end

# ----- Hessian values (objective linear; obj_factor unused) -----
function hessian_mcgc(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                      obj_factor::Float64, lambda::Vector{Float64},
                      values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.hess_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, a, sigma, alpha = param.m, param.nu, param.beta, param.a, param.sigma, param.alpha
    psigma = (sigma - 1) / sigma
    costfac = (beta + 1) / nu

    Cj = view(x, 2:1 + J)
    Djn = reshape(view(x, 2 + J:1 + J + J * N), J, N)
    Qd = reshape(view(x, 2 + J + J * N:1 + J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, 2 + J + J * N + ndeg * N:1 + J + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, 2 + J + J * N + 2 * ndeg * N:n_mcgc(J, N, ndeg))
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2)); Si = vec(sum((Qi .^ nu) .* m', dims=2))

    omega = lambda[1:J]              # cons_u multipliers
    lamC = lambda[J + 1:2 * J]       # cons_C multipliers
    Pjn = reshape(lambda[2 * J + 1:2 * J + J * N], J, N)  # cons_Q multipliers
    lamApos = Apos' * lamC; lamAneg = Aneg' * lamC

    nt = n_mcgc(J, N, ndeg)
    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (r >= c && (push!(Ir, r); push!(Jc, c); push!(V, v)))

    # (Cj,Cj) from cons_u (omega-weighted)
    @inbounds for j in 1:J
        addv!(col_Cj_mc(j, J), col_Cj_mc(j, J), -omega[j] * felicity_cc(Cj[j], graph.Hj[j], alpha))
    end
    # Djn block (cons_C, lamC)
    @inbounds for j in 1:J, nn in 1:N, nd in 1:N
        r = col_D_mc(j, nn, J); c = col_D_mc(j, nd, J); r >= c || continue
        val = -lamC[j] / sigma * Dj[j]^(-(sigma - 2) / sigma) * Djn[j, nn]^(-1 / sigma) * Djn[j, nd]^(-1 / sigma)
        nn == nd && (val += lamC[j] / sigma * Dj[j]^(1 / sigma) * Djn[j, nn]^(-1 / sigma - 1))
        addv!(r, c, val)
    end
    # Qd block (cons_C, lamApos)
    @inbounds for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qd_mc(i, nn, J, N, ndeg); c = col_Qd_mc(i, nd, J, N, ndeg); r >= c || continue
        val = (1 + beta) * (costfac - 1) * nu * lamApos[i] * Sd[i]^(costfac - 2) / kappa_ex[i] *
              m[nn] * Qd[i, nn]^(nu - 1) * m[nd] * Qd[i, nd]^(nu - 1)
        (nn == nd && nu > 1) && (val += (1 + beta) * (nu - 1) * lamApos[i] * Sd[i]^(costfac - 1) / kappa_ex[i] * m[nn] * Qd[i, nn]^(nu - 2))
        addv!(r, c, val)
    end
    # Qi block (cons_C, lamAneg)
    @inbounds for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qi_mc(i, nn, J, N, ndeg); c = col_Qi_mc(i, nd, J, N, ndeg); r >= c || continue
        val = (1 + beta) * (costfac - 1) * nu * lamAneg[i] * Si[i]^(costfac - 2) / kappa_ex[i] *
              m[nn] * Qi[i, nn]^(nu - 1) * m[nd] * Qi[i, nd]^(nu - 1)
        (nn == nd && nu > 1) && (val += (1 + beta) * (nu - 1) * lamAneg[i] * Si[i]^(costfac - 1) / kappa_ex[i] * m[nn] * Qi[i, nn]^(nu - 2))
        addv!(r, c, val)
    end
    # (Lj,u) and (Lj,Lj) from cons_u and cons_Q (production)
    @inbounds for j in 1:J
        cL = col_L_mc(j, J, N, ndeg)
        addv!(cL, 1, omega[j])
        hll = 0.0
        for nn in 1:N
            hll += -a * (a - 1) * Pjn[j, nn] * graph.Zjn[j, nn] * Lj[j]^(a - 2)
        end
        addv!(cL, cL, hll)
    end

    fill_values_from_triplets!(values, auxdata.hess_struct, Ir, Jc, V, nt, nt)
    return
end

# ----- recover allocation -----
function recover_allocation_mcgc(x, mult_g, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    psigma = (param.sigma - 1) / param.sigma

    u = x[1]
    Cj = x[2:1 + J]
    Djn = max.(0.0, reshape(x[2 + J:1 + J + J * N], J, N))
    Qd = reshape(x[2 + J + J * N:1 + J + J * N + ndeg * N], ndeg, N)
    Qi = reshape(x[2 + J + J * N + ndeg * N:1 + J + J * N + 2 * ndeg * N], ndeg, N)
    Lj = x[2 + J + J * N + 2 * ndeg * N:n_mcgc(J, N, ndeg)]
    Dj = _Dj_cgc(Djn, psigma)
    cj = ifelse.(Lj .== 0, 0.0, Cj ./ Lj)
    hj = ifelse.(Lj .== 0, 0.0, graph.Hj ./ Lj)
    uj = param.u.(cj, hj)
    Ljn = (graph.Zjn .> 0) .* Lj
    Yjn = graph.Zjn .* Lj .^ param.a
    Qin = Qd - Qi
    Qjkn = gen_network_flows(Qin, graph, N)
    Pjn = reshape(mult_g[2 * J + 1:2 * J + J * N], J, N)
    PCj = dropdims(sum(Pjn .^ (1 - param.sigma), dims=2), dims=2) .^ (1 / (1 - param.sigma))

    return (
        welfare = u,
        Yjn = Yjn, Yj = dropdims(sum(Yjn, dims=2), dims=2),
        Cjn = Djn, Cj = Cj, Djn = Djn, Dj = Dj, Ljn = Ljn, Lj = Lj,
        cj = cj, hj = hj, uj = uj,
        Pjn = Pjn, PCj = PCj, Qin = Qin, Qjkn = Qjkn,
    )
end
