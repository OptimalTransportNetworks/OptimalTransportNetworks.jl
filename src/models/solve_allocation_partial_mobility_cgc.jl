# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4
#
# Direct-Ipopt port of MATLAB solve_allocation_partial_mobility_cgc.m: labor
# mobile within regions, cross-good congestion, Armington (<=1 good per
# location). As solve_allocation_mobility_cgc but with per-region utility ur and
# regional labor-resource constraints.
#
# Variable layout (x):
#   x = [ur(R); Cj(J); Djn(J*N); Qin_direct(ndeg*N); Qin_indirect(ndeg*N); Lj(J)]
# Constraint layout (g):
#   g = [cons_u(J); cons_C(J); cons_Q(J*N); cons_L(R)]   (R = nregions)

"""
    solve_allocation_partial_mobility_cgc(x0, auxdata, verbose=true) -> (results::Dict, status, x)

Solve the partial-mobility, cross-good-congestion allocation (Armington) via direct Ipopt.
"""
function solve_allocation_partial_mobility_cgc(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions

    if any(vec(sum(graph.Zjn .> 0, dims=2)) .> 1)
        error("solve_allocation_partial_mobility_cgc only supports one good at most per location.")
    end

    n = R + J + J * N + 2 * ndeg * N + J
    if x0 === nothing || isempty(x0)
        counts = zeros(R); for j in 1:J; counts[graph.region[j]] += 1; end
        Lj0 = [graph.Lr[graph.region[j]] / counts[graph.region[j]] for j in 1:J]
        x0 = vcat(zeros(R), fill(1e-6 * J, J), fill(1e-6, J * N), fill(1e-6, 2 * ndeg * N), Lj0)
    end

    cache = get(auxdata, :struct_cache, nothing)
    js, hs = get_structs(auxdata, () -> jacobian_structure_pmcgc(auxdata), () -> hessian_structure_pmcgc(auxdata))
    saux = (auxdata..., jac_struct = js, hess_struct = hs)
    nnz_jac = length(js[1])
    nnz_hess = length(hs[1])

    obj = (x) -> -sum(graph.omegar[r] * graph.Lr[r] * x[r] for r in 1:R)
    grad = (x, grad_f) -> (fill!(grad_f, 0.0); for r in 1:R; grad_f[r] = -graph.omegar[r] * graph.Lr[r]; end; nothing)
    cons = (x, g) -> constraints_pmcgc(x, g, saux)
    jac = (x, rows, cols, values) -> jacobian_pmcgc(x, rows, cols, values, saux)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_pmcgc(x, rows, cols, obj_factor, lambda, values, saux)

    m = J + J + J * N + R
    x_L = vcat(fill(-Inf, R), fill(1e-6, J), fill(1e-6, J * N), fill(1e-6, 2 * ndeg * N), fill(1e-8, J))
    x_U = vcat(fill(Inf, R), fill(Inf, J), fill(Inf, J * N), fill(Inf, 2 * ndeg * N), fill(Inf, J))
    g_L = vcat(fill(-Inf, J + J + J * N), zeros(R))
    g_U = vcat(fill(0.0, J + J + J * N), zeros(R))

    x, status, mult_g = run_ipopt_primal(cache, :partial_mobility_cgc, saux, n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                                         obj, cons, grad, jac, hess, x0, param, verbose)
    results = recover_allocation_pmcgc(x, mult_g, auxdata)
    return namedtuple_to_dict(results), status, x
end

# ----- index helpers -----
@inline n_pmcgc(J, N, ndeg, R) = R + J + J * N + 2 * ndeg * N + J
@inline col_ur_pmc(r) = r
@inline col_Cj_pmc(j, R) = R + j
@inline col_D_pmc(j, n, J, R) = R + J + (n - 1) * J + j
@inline col_Qd_pmc(i, n, J, N, ndeg, R) = R + J + J * N + (n - 1) * ndeg + i
@inline col_Qi_pmc(i, n, J, N, ndeg, R) = R + J + J * N + ndeg * N + (n - 1) * ndeg + i
@inline col_L_pmc(j, J, N, ndeg, R) = R + J + J * N + 2 * ndeg * N + j
@inline row_u_pmc(j) = j
@inline row_C_pmc(j, J) = J + j
@inline row_Q_pmc(j, n, J) = 2 * J + (n - 1) * J + j
@inline row_L_pmc(r, J, N) = 2 * J + J * N + r

# ----- constraints -----
function constraints_pmcgc(x::Vector{Float64}, g::Vector{Float64}, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, a, alpha = param.m, param.nu, param.beta, param.a, param.alpha
    psigma = (param.sigma - 1) / param.sigma
    costfac = (beta + 1) / nu
    region = graph.region

    ur = view(x, 1:R)
    Cj = view(x, R + 1:R + J)
    Djn = reshape(view(x, R + J + 1:R + J + J * N), J, N)
    Qd = reshape(view(x, R + J + J * N + 1:R + J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, R + J + J * N + ndeg * N + 1:R + J + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, R + J + J * N + 2 * ndeg * N + 1:n_pmcgc(J, N, ndeg, R))
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2)); Si = vec(sum((Qi .^ nu) .* m', dims=2))
    cost_d = Apos * (Sd .^ costfac ./ kappa_ex); cost_i = Aneg * (Si .^ costfac ./ kappa_ex)

    @inbounds for j in 1:J
        g[row_u_pmc(j)] = Lj[j] * ur[region[j]] - felicity(Cj[j], graph.Hj[j], alpha)
        g[row_C_pmc(j, J)] = Cj[j] + cost_d[j] + cost_i[j] - Dj[j]
    end
    @inbounds for nn in 1:N
        netflow = A * Qd[:, nn] - A * Qi[:, nn]
        for j in 1:J
            g[row_Q_pmc(j, nn, J)] = Djn[j, nn] + netflow[j] - graph.Zjn[j, nn] * Lj[j]^a
        end
    end
    @inbounds for r in 1:R; g[row_L_pmc(r, J, N)] = -graph.Lr[r]; end
    @inbounds for j in 1:J; g[row_L_pmc(region[j], J, N)] += Lj[j]; end
    return
end

# ----- Jacobian structure -----
function jacobian_structure_pmcgc(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    A = auxdata.edges.A; region = graph.region
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    for j in 1:J
        push_rc!(j, col_ur_pmc(region[j]))
        push_rc!(j, col_Cj_pmc(j, R))
        push_rc!(j, col_L_pmc(j, J, N, ndeg, R))
    end
    for j in 1:J
        r = row_C_pmc(j, J)
        push_rc!(r, col_Cj_pmc(j, R))
        for nn in 1:N; push_rc!(r, col_D_pmc(j, nn, J, R)); end
        for nn in 1:N, i in 1:ndeg
            if A[j, i] > 0; push_rc!(r, col_Qd_pmc(i, nn, J, N, ndeg, R)); end
            if A[j, i] < 0; push_rc!(r, col_Qi_pmc(i, nn, J, N, ndeg, R)); end
        end
    end
    for nn in 1:N, j in 1:J
        r = row_Q_pmc(j, nn, J)
        push_rc!(r, col_D_pmc(j, nn, J, R))
        for i in 1:ndeg
            if A[j, i] != 0
                push_rc!(r, col_Qd_pmc(i, nn, J, N, ndeg, R))
                push_rc!(r, col_Qi_pmc(i, nn, J, N, ndeg, R))
            end
        end
        push_rc!(r, col_L_pmc(j, J, N, ndeg, R))
    end
    for j in 1:J; push_rc!(row_L_pmc(region[j], J, N), col_L_pmc(j, J, N, ndeg, R)); end
    return struct_from_triplets(rows, cols, J + J + J * N + R, n_pmcgc(J, N, ndeg, R))
end

# ----- Jacobian values -----
function jacobian_pmcgc(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                        values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.jac_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, a, sigma, alpha = param.m, param.nu, param.beta, param.a, param.sigma, param.alpha
    psigma = (sigma - 1) / sigma
    costfac = (beta + 1) / nu
    region = graph.region

    ur = view(x, 1:R)
    Cj = view(x, R + 1:R + J)
    Djn = reshape(view(x, R + J + 1:R + J + J * N), J, N)
    Qd = reshape(view(x, R + J + J * N + 1:R + J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, R + J + J * N + ndeg * N + 1:R + J + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, R + J + J * N + 2 * ndeg * N + 1:n_pmcgc(J, N, ndeg, R))
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2)); Si = vec(sum((Qi .^ nu) .* m', dims=2))
    cpos = Sd .^ (costfac - 1) ./ kappa_ex; cneg = Si .^ (costfac - 1) ./ kappa_ex

    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (push!(Ir, r); push!(Jc, c); push!(V, v))
    @inbounds for j in 1:J
        addv!(j, col_ur_pmc(region[j]), Lj[j])
        addv!(j, col_Cj_pmc(j, R), -felicity_c(Cj[j], graph.Hj[j], alpha))
        addv!(j, col_L_pmc(j, J, N, ndeg, R), ur[region[j]])
    end
    @inbounds for j in 1:J
        r = row_C_pmc(j, J)
        addv!(r, col_Cj_pmc(j, R), 1.0)
        for nn in 1:N
            addv!(r, col_D_pmc(j, nn, J, R), -Dj[j]^(1 / sigma) * Djn[j, nn]^(-1 / sigma))
        end
        for nn in 1:N, i in 1:ndeg
            if A[j, i] > 0
                addv!(r, col_Qd_pmc(i, nn, J, N, ndeg, R), Apos[j, i] * (1 + beta) * cpos[i] * m[nn] * Qd[i, nn]^(nu - 1))
            end
            if A[j, i] < 0
                addv!(r, col_Qi_pmc(i, nn, J, N, ndeg, R), Aneg[j, i] * (1 + beta) * cneg[i] * m[nn] * Qi[i, nn]^(nu - 1))
            end
        end
    end
    @inbounds for nn in 1:N, j in 1:J
        r = row_Q_pmc(j, nn, J)
        addv!(r, col_D_pmc(j, nn, J, R), 1.0)
        for i in 1:ndeg
            if A[j, i] != 0
                addv!(r, col_Qd_pmc(i, nn, J, N, ndeg, R), A[j, i])
                addv!(r, col_Qi_pmc(i, nn, J, N, ndeg, R), -A[j, i])
            end
        end
        addv!(r, col_L_pmc(j, J, N, ndeg, R), -a * graph.Zjn[j, nn] * Lj[j]^(a - 1))
    end
    @inbounds for j in 1:J; addv!(row_L_pmc(region[j], J, N), col_L_pmc(j, J, N, ndeg, R), 1.0); end

    fill_values_from_triplets!(values, auxdata.jac_struct, Ir, Jc, V, J + J + J * N + R, n_pmcgc(J, N, ndeg, R))
    return
end

# ----- Hessian structure -----
function hessian_structure_pmcgc(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    region = graph.region
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))
    for j in 1:J; push_rc!(col_Cj_pmc(j, R), col_Cj_pmc(j, R)); end
    for j in 1:J, nn in 1:N, nd in 1:N
        r = col_D_pmc(j, nn, J, R); c = col_D_pmc(j, nd, J, R); r >= c && push_rc!(r, c)
    end
    for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qd_pmc(i, nn, J, N, ndeg, R); c = col_Qd_pmc(i, nd, J, N, ndeg, R); r >= c && push_rc!(r, c)
    end
    for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qi_pmc(i, nn, J, N, ndeg, R); c = col_Qi_pmc(i, nd, J, N, ndeg, R); r >= c && push_rc!(r, c)
    end
    for j in 1:J
        cL = col_L_pmc(j, J, N, ndeg, R)
        push_rc!(cL, col_ur_pmc(region[j]))   # (Lj, ur[region])
        push_rc!(cL, cL)                        # (Lj, Lj)
    end
    nt = n_pmcgc(J, N, ndeg, R)
    return struct_from_triplets(rows, cols, nt, nt)
end

# ----- Hessian values (objective linear) -----
function hessian_pmcgc(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                       obj_factor::Float64, lambda::Vector{Float64},
                       values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.hess_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    m, nu, beta, a, sigma, alpha = param.m, param.nu, param.beta, param.a, param.sigma, param.alpha
    psigma = (sigma - 1) / sigma
    costfac = (beta + 1) / nu
    region = graph.region

    Cj = view(x, R + 1:R + J)
    Djn = reshape(view(x, R + J + 1:R + J + J * N), J, N)
    Qd = reshape(view(x, R + J + J * N + 1:R + J + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, R + J + J * N + ndeg * N + 1:R + J + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, R + J + J * N + 2 * ndeg * N + 1:n_pmcgc(J, N, ndeg, R))
    Dj = _Dj_cgc(Djn, psigma)
    Sd = vec(sum((Qd .^ nu) .* m', dims=2)); Si = vec(sum((Qi .^ nu) .* m', dims=2))

    omega = lambda[1:J]
    lamC = lambda[J + 1:2 * J]
    Pjn = reshape(lambda[2 * J + 1:2 * J + J * N], J, N)
    lamApos = Apos' * lamC; lamAneg = Aneg' * lamC

    nt = n_pmcgc(J, N, ndeg, R)
    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (r >= c && (push!(Ir, r); push!(Jc, c); push!(V, v)))

    @inbounds for j in 1:J
        addv!(col_Cj_pmc(j, R), col_Cj_pmc(j, R), -omega[j] * felicity_cc(Cj[j], graph.Hj[j], alpha))
    end
    @inbounds for j in 1:J, nn in 1:N, nd in 1:N
        r = col_D_pmc(j, nn, J, R); c = col_D_pmc(j, nd, J, R); r >= c || continue
        val = -lamC[j] / sigma * Dj[j]^(-(sigma - 2) / sigma) * Djn[j, nn]^(-1 / sigma) * Djn[j, nd]^(-1 / sigma)
        nn == nd && (val += lamC[j] / sigma * Dj[j]^(1 / sigma) * Djn[j, nn]^(-1 / sigma - 1))
        addv!(r, c, val)
    end
    @inbounds for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qd_pmc(i, nn, J, N, ndeg, R); c = col_Qd_pmc(i, nd, J, N, ndeg, R); r >= c || continue
        val = (1 + beta) * (costfac - 1) * nu * lamApos[i] * Sd[i]^(costfac - 2) / kappa_ex[i] *
              m[nn] * Qd[i, nn]^(nu - 1) * m[nd] * Qd[i, nd]^(nu - 1)
        (nn == nd && nu > 1) && (val += (1 + beta) * (nu - 1) * lamApos[i] * Sd[i]^(costfac - 1) / kappa_ex[i] * m[nn] * Qd[i, nn]^(nu - 2))
        addv!(r, c, val)
    end
    @inbounds for i in 1:ndeg, nn in 1:N, nd in 1:N
        r = col_Qi_pmc(i, nn, J, N, ndeg, R); c = col_Qi_pmc(i, nd, J, N, ndeg, R); r >= c || continue
        val = (1 + beta) * (costfac - 1) * nu * lamAneg[i] * Si[i]^(costfac - 2) / kappa_ex[i] *
              m[nn] * Qi[i, nn]^(nu - 1) * m[nd] * Qi[i, nd]^(nu - 1)
        (nn == nd && nu > 1) && (val += (1 + beta) * (nu - 1) * lamAneg[i] * Si[i]^(costfac - 1) / kappa_ex[i] * m[nn] * Qi[i, nn]^(nu - 2))
        addv!(r, c, val)
    end
    @inbounds for j in 1:J
        cL = col_L_pmc(j, J, N, ndeg, R)
        addv!(cL, col_ur_pmc(region[j]), omega[j])
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
function recover_allocation_pmcgc(x, mult_g, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    psigma = (param.sigma - 1) / param.sigma

    ur = x[1:R]
    Cj = x[R + 1:R + J]
    Djn = max.(0.0, reshape(x[R + J + 1:R + J + J * N], J, N))
    Qd = reshape(x[R + J + J * N + 1:R + J + J * N + ndeg * N], ndeg, N)
    Qi = reshape(x[R + J + J * N + ndeg * N + 1:R + J + J * N + 2 * ndeg * N], ndeg, N)
    Lj = x[R + J + J * N + 2 * ndeg * N + 1:n_pmcgc(J, N, ndeg, R)]
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
        welfare = sum(graph.omegar .* graph.Lr .* ur),
        reg_pc_welfare = ur,
        Yjn = Yjn, Yj = dropdims(sum(Yjn, dims=2), dims=2),
        Cjn = Djn, Cj = Cj, Djn = Djn, Dj = Dj, Ljn = Ljn, Lj = Lj,
        cj = cj, hj = hj, uj = uj,
        Pjn = Pjn, PCj = PCj, Qin = Qin, Qjkn = Qjkn,
    )
end
