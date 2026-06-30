# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4
#
# Direct-Ipopt port of MATLAB solve_allocation_mobility.m: the primal problem
# with full labor mobility, no cross-good congestion, Armington (<=1 good per
# location). Calls Ipopt.jl's C interface directly with hand-coded sparse
# gradient/Jacobian/Hessian, mirroring the existing dual solvers.
#
# Variable layout (x):
#   x = [u; Cjn(J*N); Qin_direct(ndeg*N); Qin_indirect(ndeg*N); Lj(J)]
# Constraint layout (g):
#   g = [cons_u(J); cons_Q(J*N); cons_L(1)]

"""
    solve_allocation_mobility(x0, auxdata, verbose=true) -> (results::Dict, status, x)

Solve the full-mobility, no-congestion allocation (Armington) given a matrix of
kappa (= I^gamma / delta_tau) using a direct Ipopt primal approach.

- `x0`: initial seed for the solver (empty `[]` for the default)
- `auxdata`: model parameters (param, graph, kappa, kappa_ex, edges)
- `verbose`: tell Ipopt to display output or not

Returns the results Dict, the Ipopt status code, and the solution vector `x`
(useful for warm starts).
"""
function solve_allocation_mobility(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg

    # This hand-coded primal supports at most one good per location (Armington)
    if any(vec(sum(graph.Zjn .> 0, dims=2)) .> 1)
        error("solve_allocation_mobility only supports one good at most per location.")
    end

    # Variable count and default seed
    n = 1 + J * N + 2 * ndeg * N + J
    if x0 === nothing || isempty(x0)
        x0 = vcat(0.0, fill(1e-6, J * N), fill(1e-4, 2 * ndeg * N), fill(1.0 / J, J))
    end

    # Precompute sparsity structures (superset patterns, x-independent)
    cache = get(auxdata, :struct_cache, nothing)
    js, hs = get_structs(auxdata, () -> jacobian_structure_mobility(auxdata), () -> hessian_structure_mobility(auxdata))
    saux = (auxdata..., jac_struct = js, hess_struct = hs)
    nnz_jac = length(js[1])
    nnz_hess = length(hs[1])

    # Callbacks (capture the stable auxdata `saux` whose kappa_ex the cache updates in place)
    obj = (x) -> objective_mobility(x, saux)
    grad = (x, grad_f) -> gradient_mobility(x, grad_f, saux)
    cons = (x, g) -> constraints_mobility(x, g, saux)
    jac = (x, rows, cols, values) -> jacobian_mobility(x, rows, cols, values, saux)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_mobility(x, rows, cols, obj_factor, lambda, values, saux)

    # Bounds
    m = J + J * N + 1
    x_L = vcat(-Inf, fill(1e-6, J * N), fill(1e-6, 2 * ndeg * N), fill(1e-8, J))
    x_U = vcat(Inf, fill(Inf, J * N), fill(Inf, 2 * ndeg * N), fill(1.0, J))
    # cons_u <= 0, cons_Q <= 0 (balanced flow), cons_L == 0 (labor). NB: the MATLAB
    # reference uses a loose 1e-3 upper bound on cons_Q, which inflates welfare here
    # (prices are O(1), not rescaled as in MATLAB), so we keep the exact <= 0 bound.
    g_L = vcat(fill(-Inf, J), fill(-Inf, J * N), 0.0)
    g_U = vcat(fill(0.0, J), fill(0.0, J * N), 0.0)

    x, status, mult_g = run_ipopt_primal(cache, :mobility, saux, n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                                         obj, cons, grad, jac, hess, x0, param, verbose)

    results = recover_allocation_mobility(x, mult_g, auxdata)
    return namedtuple_to_dict(results), status, x
end

# ----- column / row index helpers -----
# u:                col 1
# Cjn[j,n]:         col 1 + (n-1)*J + j
# Qin_direct[i,n]:  col 1 + J*N + (n-1)*ndeg + i
# Qin_indirect[i,n]:col 1 + J*N + ndeg*N + (n-1)*ndeg + i
# Lj[j]:            col 1 + J*N + 2*ndeg*N + j
# cons_u[j]:        row j
# cons_Q[j,n]:      row J + (n-1)*J + j
# cons_L:           row J + J*N + 1
@inline col_C_mob(j, n, J) = 1 + (n - 1) * J + j
@inline col_Qd_mob(i, n, J, N, ndeg) = 1 + J * N + (n - 1) * ndeg + i
@inline col_Qi_mob(i, n, J, N, ndeg) = 1 + J * N + ndeg * N + (n - 1) * ndeg + i
@inline col_L_mob(j, J, N, ndeg) = 1 + J * N + 2 * ndeg * N + j
@inline row_Q_mob(j, n, J) = J + (n - 1) * J + j

# ----- objective and gradient (objective is linear: f = -u) -----
function objective_mobility(x, auxdata)
    return -x[1]
end

function gradient_mobility(x::Vector{Float64}, grad_f::Vector{Float64}, auxdata)
    fill!(grad_f, 0.0)
    grad_f[1] = -1.0
    return
end

# ----- constraints -----
function constraints_mobility(x::Vector{Float64}, g::Vector{Float64}, auxdata)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    psigma = (param.sigma - 1) / param.sigma
    beta = param.beta

    u = x[1]
    Cjn = reshape(view(x, 2:1 + J * N), J, N)
    Qd = reshape(view(x, 2 + J * N:1 + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, 2 + J * N + ndeg * N:1 + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, 2 + J * N + 2 * ndeg * N:n_mobility(J, N, ndeg))

    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)

    # cons_u (J): Lj*u - U(Cj, Hj)
    @inbounds for j in 1:J
        g[j] = Lj[j] * u - (Cj[j] / param.alpha)^param.alpha * (graph.Hj[j] / (1 - param.alpha))^(1 - param.alpha)
    end

    # cons_Q (J*N): Cjn + Apos*(Qd^(1+beta)/kappa) + Aneg*(Qi^(1+beta)/kappa) + A*Qd - A*Qi - Yjn
    @inbounds for nn in 1:N
        cd = Qd[:, nn] .^ (1 + beta) ./ kappa_ex
        ci = Qi[:, nn] .^ (1 + beta) ./ kappa_ex
        flow = Apos * cd + Aneg * ci + A * Qd[:, nn] - A * Qi[:, nn]
        for j in 1:J
            g[row_Q_mob(j, nn, J)] = Cjn[j, nn] + flow[j] - graph.Zjn[j, nn] * Lj[j]^param.a
        end
    end

    # cons_L (1): sum(Lj) - 1
    g[J + J * N + 1] = sum(Lj) - 1.0
    return
end

@inline n_mobility(J, N, ndeg) = 1 + J * N + 2 * ndeg * N + J

# ----- Jacobian structure (superset, matches MATLAB jacobianstructure) -----
function jacobian_structure_mobility(auxdata)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A
    rows = Int[]
    cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))

    # cons_u row j
    for j in 1:J
        push_rc!(j, 1)                              # u
        for nn in 1:N
            push_rc!(j, col_C_mob(j, nn, J))        # Cjn[j,n]
        end
        push_rc!(j, col_L_mob(j, J, N, ndeg))       # Lj[j]
    end
    # cons_Q row (j,n)
    for nn in 1:N, j in 1:J
        r = row_Q_mob(j, nn, J)
        push_rc!(r, col_C_mob(j, nn, J))            # Cjn eye
        for i in 1:ndeg
            if A[j, i] != 0
                push_rc!(r, col_Qd_mob(i, nn, J, N, ndeg))
                push_rc!(r, col_Qi_mob(i, nn, J, N, ndeg))
            end
        end
        push_rc!(r, col_L_mob(j, J, N, ndeg))       # Lj (repmat eye)
    end
    # cons_L row
    rL = J + J * N + 1
    for j in 1:J
        push_rc!(rL, col_L_mob(j, J, N, ndeg))
    end

    return struct_from_triplets(rows, cols, J + J * N + 1, n_mobility(J, N, ndeg))
end

# ----- Jacobian values -----
function jacobian_mobility(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                           values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.jac_struct
        rows .= r
        cols .= c
        return
    end
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    A = auxdata.edges.A
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    sigma, alpha, beta, a = param.sigma, param.alpha, param.beta, param.a
    psigma = (sigma - 1) / sigma

    u = x[1]
    Cjn = reshape(view(x, 2:1 + J * N), J, N)
    Qd = reshape(view(x, 2 + J * N:1 + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, 2 + J * N + ndeg * N:1 + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, 2 + J * N + 2 * ndeg * N:n_mobility(J, N, ndeg))
    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)

    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (push!(Ir, r); push!(Jc, c); push!(V, v))

    # cons_u
    @inbounds for j in 1:J
        addv!(j, 1, Lj[j])                                       # d/du
        cu = felicity_c(Cj[j], graph.Hj[j], alpha)
        for nn in 1:N
            addv!(j, col_C_mob(j, nn, J), -Cjn[j, nn]^(-1 / sigma) * Cj[j]^(1 / sigma) * cu)
        end
        addv!(j, col_L_mob(j, J, N, ndeg), u)                   # d/dLj
    end
    # cons_Q
    @inbounds for nn in 1:N, j in 1:J
        r = row_Q_mob(j, nn, J)
        addv!(r, col_C_mob(j, nn, J), 1.0)                      # d/dCjn
        for i in 1:ndeg
            if A[j, i] != 0
                addv!(r, col_Qd_mob(i, nn, J, N, ndeg), A[j, i] + (1 + beta) * Apos[j, i] * Qd[i, nn]^beta / kappa_ex[i])
                addv!(r, col_Qi_mob(i, nn, J, N, ndeg), -A[j, i] + (1 + beta) * Aneg[j, i] * Qi[i, nn]^beta / kappa_ex[i])
            end
        end
        addv!(r, col_L_mob(j, J, N, ndeg), -a * graph.Zjn[j, nn] * Lj[j]^(a - 1))  # d/dLj
    end
    # cons_L
    rL = J + J * N + 1
    @inbounds for j in 1:J
        addv!(rL, col_L_mob(j, J, N, ndeg), 1.0)
    end

    fill_values_from_triplets!(values, auxdata.jac_struct, Ir, Jc, V, J + J * N + 1, n_mobility(J, N, ndeg))
    return
end

# ----- Hessian structure (superset, lower triangle, matches MATLAB) -----
function hessian_structure_mobility(auxdata)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))

    # (Cjn, Cjn) full N*N block per location j (lower triangle)
    for j in 1:J, nn in 1:N, nd in 1:N
        r = col_C_mob(j, nn, J)
        c = col_C_mob(j, nd, J)
        if r >= c
            push_rc!(r, c)
        end
    end
    # (Qdir, Qdir) diagonal
    for nn in 1:N, i in 1:ndeg
        cc = col_Qd_mob(i, nn, J, N, ndeg); push_rc!(cc, cc)
    end
    # (Qindir, Qindir) diagonal
    for nn in 1:N, i in 1:ndeg
        cc = col_Qi_mob(i, nn, J, N, ndeg); push_rc!(cc, cc)
    end
    # (Lj, u) and (Lj, Lj) diagonal
    for j in 1:J
        cL = col_L_mob(j, J, N, ndeg)
        push_rc!(cL, 1)        # (Lj, u)
        push_rc!(cL, cL)       # (Lj, Lj)
    end

    nn_tot = n_mobility(J, N, ndeg)
    return struct_from_triplets(rows, cols, nn_tot, nn_tot)
end

# ----- Hessian values (Lagrangian; objective linear so obj_factor unused) -----
function hessian_mobility(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                          obj_factor::Float64, lambda::Vector{Float64},
                          values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.hess_struct
        rows .= r
        cols .= c
        return
    end
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    sigma, beta, a, alpha = param.sigma, param.beta, param.a, param.alpha
    psigma = (sigma - 1) / sigma

    Cjn = reshape(view(x, 2:1 + J * N), J, N)
    Qd = reshape(view(x, 2 + J * N:1 + J * N + ndeg * N), ndeg, N)
    Qi = reshape(view(x, 2 + J * N + ndeg * N:1 + J * N + 2 * ndeg * N), ndeg, N)
    Lj = view(x, 2 + J * N + 2 * ndeg * N:n_mobility(J, N, ndeg))
    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)

    omegaj = lambda[1:J]
    Pjn = reshape(lambda[J + 1:J + J * N], J, N)

    nn_tot = n_mobility(J, N, ndeg)
    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (r >= c && (push!(Ir, r); push!(Jc, c); push!(V, v)))

    # (Cjn, Cjn) block per location j: diag(Hcdiag) + Hcnondiag
    @inbounds for j in 1:J
        up = felicity_c(Cj[j], graph.Hj[j], alpha)
        us = felicity_cc(Cj[j], graph.Hj[j], alpha)
        pref = -(omegaj[j] * us * Cj[j]^(2 / sigma) + (1 / sigma) * omegaj[j] * up * Cj[j]^(2 / sigma - 1))
        for nn in 1:N, nd in 1:N
            r = col_C_mob(j, nn, J)
            c = col_C_mob(j, nd, J)
            r >= c || continue
            val = pref * Cjn[j, nn]^(-1 / sigma) * Cjn[j, nd]^(-1 / sigma)   # Hcnondiag
            if nn == nd
                val += (1 / sigma) * omegaj[j] * up * Cj[j]^(1 / sigma) * Cjn[j, nn]^(-1 / sigma - 1)  # Hcdiag
            end
            addv!(r, c, val)
        end
    end
    # (Qdir, Qdir) and (Qindir, Qindir) diagonals
    @inbounds for nn in 1:N
        ApP = Apos' * Pjn[:, nn]
        AnP = Aneg' * Pjn[:, nn]
        for i in 1:ndeg
            cd = col_Qd_mob(i, nn, J, N, ndeg)
            addv!(cd, cd, (1 + beta) * beta * Qd[i, nn]^(beta - 1) / kappa_ex[i] * ApP[i])
            ci = col_Qi_mob(i, nn, J, N, ndeg)
            addv!(ci, ci, (1 + beta) * beta * Qi[i, nn]^(beta - 1) / kappa_ex[i] * AnP[i])
        end
    end
    # (Lj, u) = omegaj  and (Lj, Lj) = Hl
    @inbounds for j in 1:J
        cL = col_L_mob(j, J, N, ndeg)
        addv!(cL, 1, omegaj[j])
        hl = 0.0
        for nn in 1:N
            # NB: Zjn factor is required (production Yjn = Zjn*Lj^a). The MATLAB
            # reference omits it in Hl, which is a latent bug for Zjn != 1.
            hl += -a * (a - 1) * graph.Zjn[j, nn] * Pjn[j, nn] * Lj[j]^(a - 2)
        end
        addv!(cL, cL, hl)
    end

    fill_values_from_triplets!(values, auxdata.hess_struct, Ir, Jc, V, nn_tot, nn_tot)
    return
end

# ----- recover allocation -----
function recover_allocation_mobility(x, mult_g, auxdata)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg = graph.J, param.N, graph.ndeg
    psigma = (param.sigma - 1) / param.sigma

    u = x[1]
    Cjn = reshape(x[2:1 + J * N], J, N)
    Qd = reshape(x[2 + J * N:1 + J * N + ndeg * N], ndeg, N)
    Qi = reshape(x[2 + J * N + ndeg * N:1 + J * N + 2 * ndeg * N], ndeg, N)
    Lj = x[2 + J * N + 2 * ndeg * N:n_mobility(J, N, ndeg)]

    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)
    Ljn = (graph.Zjn .> 0) .* Lj
    Yjn = graph.Zjn .* Lj .^ param.a
    cj = ifelse.(Lj .== 0, 0.0, Cj ./ Lj)
    hj = ifelse.(Lj .== 0, 0.0, graph.Hj ./ Lj)
    uj = param.u.(cj, hj)
    Qin = Qd - Qi
    Qjkn = gen_network_flows(Qin, graph, N)

    Pjn = reshape(mult_g[J + 1:J + J * N], J, N)
    PCj = dropdims(sum(Pjn .^ (1 - param.sigma), dims=2), dims=2) .^ (1 / (1 - param.sigma))

    results = (
        welfare = u,
        Yjn = Yjn,
        Yj = dropdims(sum(Yjn, dims=2), dims=2),
        Cjn = Cjn,
        Cj = Cj,
        Ljn = Ljn,
        Lj = Lj,
        cj = cj,
        hj = hj,
        uj = uj,
        Pjn = Pjn,
        PCj = PCj,
        Qin = Qin,
        Qjkn = Qjkn,
    )
    return results
end
