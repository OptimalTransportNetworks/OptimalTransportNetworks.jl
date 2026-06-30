# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4
#
# Direct-Ipopt port of MATLAB solve_allocation_partial_mobility.m: the primal
# problem with labor mobile within regions, no cross-good congestion, Armington
# (<=1 good per location). Uses a single signed flow Qin (as the MATLAB and JuMP
# references do for this case). NB: the split-flow formulation used for full
# mobility lands on worse local optima here because partial mobility's bilinear
# Lj*ur term makes the inner problem non-convex.
#
# Variable layout (x):
#   x = [ur(R); Cjn(J*N); Qin(ndeg*N); Lj(J)]   (R = nregions)
# Constraint layout (g):
#   g = [cons_u(J); cons_Q(J*N); cons_L(R)]

"""
    solve_allocation_partial_mobility(x0, auxdata, verbose=true) -> (results::Dict, status, x)

Solve the partial-mobility, no-congestion allocation (Armington) via direct Ipopt.
"""
function solve_allocation_partial_mobility(x0, auxdata, verbose=true)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions

    if any(vec(sum(graph.Zjn .> 0, dims=2)) .> 1)
        error("solve_allocation_partial_mobility only supports one good at most per location.")
    end

    n = R + J * N + ndeg * N + J
    if x0 === nothing || isempty(x0)
        counts = zeros(R)
        for j in 1:J; counts[graph.region[j]] += 1; end
        Lj0 = [graph.Lr[graph.region[j]] / counts[graph.region[j]] for j in 1:J]
        x0 = vcat(zeros(R), fill(1e-6, J * N), zeros(ndeg * N), Lj0)
    end

    cache = get(auxdata, :struct_cache, nothing)
    js, hs = get_structs(auxdata, () -> jacobian_structure_pmob(auxdata), () -> hessian_structure_pmob(auxdata))
    saux = (auxdata..., jac_struct = js, hess_struct = hs)
    nnz_jac = length(js[1])
    nnz_hess = length(hs[1])

    obj = (x) -> objective_pmob(x, saux)
    grad = (x, grad_f) -> gradient_pmob(x, grad_f, saux)
    cons = (x, g) -> constraints_pmob(x, g, saux)
    jac = (x, rows, cols, values) -> jacobian_pmob(x, rows, cols, values, saux)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_pmob(x, rows, cols, obj_factor, lambda, values, saux)

    m = J + J * N + R
    x_L = vcat(fill(-Inf, R), fill(1e-6, J * N), fill(-Inf, ndeg * N), fill(1e-8, J))
    x_U = vcat(fill(Inf, R), fill(Inf, J * N), fill(Inf, ndeg * N), fill(Inf, J))
    g_L = vcat(fill(-Inf, J), fill(-Inf, J * N), zeros(R))
    g_U = vcat(fill(0.0, J), fill(0.0, J * N), zeros(R))

    x, status, mult_g = run_ipopt_primal(cache, :partial_mobility, saux, n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                                         obj, cons, grad, jac, hess, x0, param, verbose)
    results = recover_allocation_pmob(x, mult_g, auxdata)
    return namedtuple_to_dict(results), status, x
end

# ----- index helpers (R = nregions offset, single flow Qin) -----
@inline n_pmob(J, N, ndeg, R) = R + J * N + ndeg * N + J
@inline col_ur_pm(r) = r
@inline col_C_pm(j, n, J, R) = R + (n - 1) * J + j
@inline col_Q_pm(i, n, J, N, ndeg, R) = R + J * N + (n - 1) * ndeg + i
@inline col_L_pm(j, J, N, ndeg, R) = R + J * N + ndeg * N + j
@inline row_Q_pm(j, n, J) = J + (n - 1) * J + j
@inline row_L_pm(r, J, N) = J + J * N + r

# ----- objective (linear in ur) -----
function objective_pmob(x, auxdata)
    graph = auxdata.graph
    return -sum(graph.omegar[r] * graph.Lr[r] * x[r] for r in 1:graph.nregions)
end

function gradient_pmob(x::Vector{Float64}, grad_f::Vector{Float64}, auxdata)
    graph = auxdata.graph
    fill!(grad_f, 0.0)
    @inbounds for r in 1:graph.nregions
        grad_f[r] = -graph.omegar[r] * graph.Lr[r]
    end
    return
end

# ----- constraints -----
function constraints_pmob(x::Vector{Float64}, g::Vector{Float64}, auxdata)
    graph = auxdata.graph
    param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    psigma = (param.sigma - 1) / param.sigma
    beta, alpha, a = param.beta, param.alpha, param.a
    region = graph.region

    ur = view(x, 1:R)
    Cjn = reshape(view(x, R + 1:R + J * N), J, N)
    Qin = reshape(view(x, R + J * N + 1:R + J * N + ndeg * N), ndeg, N)
    Lj = view(x, R + J * N + ndeg * N + 1:n_pmob(J, N, ndeg, R))
    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)

    @inbounds for j in 1:J
        g[j] = Lj[j] * ur[region[j]] - felicity(Cj[j], graph.Hj[j], alpha)
    end
    @inbounds for nn in 1:N
        q = Qin[:, nn]
        cost = abs.(q) .^ (1 + beta) ./ kappa_ex
        pos = q .> 0
        cost_at = Apos * (cost .* pos) + Aneg * (cost .* .!pos)   # charged at origin node
        flow = A * q + cost_at
        for j in 1:J
            g[row_Q_pm(j, nn, J)] = Cjn[j, nn] + flow[j] - graph.Zjn[j, nn] * Lj[j]^a
        end
    end
    @inbounds for r in 1:R
        g[row_L_pm(r, J, N)] = -graph.Lr[r]
    end
    @inbounds for j in 1:J
        g[row_L_pm(region[j], J, N)] += Lj[j]
    end
    return
end

# ----- Jacobian structure -----
function jacobian_structure_pmob(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    A = auxdata.edges.A; region = graph.region
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))

    for j in 1:J
        push_rc!(j, col_ur_pm(region[j]))
        for nn in 1:N; push_rc!(j, col_C_pm(j, nn, J, R)); end
        push_rc!(j, col_L_pm(j, J, N, ndeg, R))
    end
    for nn in 1:N, j in 1:J
        r = row_Q_pm(j, nn, J)
        push_rc!(r, col_C_pm(j, nn, J, R))
        for i in 1:ndeg
            if A[j, i] != 0
                push_rc!(r, col_Q_pm(i, nn, J, N, ndeg, R))
            end
        end
        push_rc!(r, col_L_pm(j, J, N, ndeg, R))
    end
    for j in 1:J
        push_rc!(row_L_pm(region[j], J, N), col_L_pm(j, J, N, ndeg, R))
    end
    return struct_from_triplets(rows, cols, J + J * N + R, n_pmob(J, N, ndeg, R))
end

# ----- Jacobian values -----
function jacobian_pmob(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                       values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.jac_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    sigma, alpha, beta, a = param.sigma, param.alpha, param.beta, param.a
    psigma = (sigma - 1) / sigma
    region = graph.region

    ur = view(x, 1:R)
    Cjn = reshape(view(x, R + 1:R + J * N), J, N)
    Qin = reshape(view(x, R + J * N + 1:R + J * N + ndeg * N), ndeg, N)
    Lj = view(x, R + J * N + ndeg * N + 1:n_pmob(J, N, ndeg, R))
    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)

    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (push!(Ir, r); push!(Jc, c); push!(V, v))

    @inbounds for j in 1:J
        addv!(j, col_ur_pm(region[j]), Lj[j])
        cu = felicity_c(Cj[j], graph.Hj[j], alpha)
        for nn in 1:N
            addv!(j, col_C_pm(j, nn, J, R), -Cjn[j, nn]^(-1 / sigma) * Cj[j]^(1 / sigma) * cu)
        end
        addv!(j, col_L_pm(j, J, N, ndeg, R), ur[region[j]])
    end
    @inbounds for nn in 1:N, j in 1:J
        r = row_Q_pm(j, nn, J)
        addv!(r, col_C_pm(j, nn, J, R), 1.0)
        for i in 1:ndeg
            if A[j, i] != 0
                q = Qin[i, nn]
                Morig = q > 0 ? Apos[j, i] : Aneg[j, i]
                dcost = (1 + beta) * Morig * sign(q) * abs(q)^beta / kappa_ex[i]
                addv!(r, col_Q_pm(i, nn, J, N, ndeg, R), A[j, i] + dcost)
            end
        end
        addv!(r, col_L_pm(j, J, N, ndeg, R), -a * graph.Zjn[j, nn] * Lj[j]^(a - 1))
    end
    @inbounds for j in 1:J
        addv!(row_L_pm(region[j], J, N), col_L_pm(j, J, N, ndeg, R), 1.0)
    end

    fill_values_from_triplets!(values, auxdata.jac_struct, Ir, Jc, V, J + J * N + R, n_pmob(J, N, ndeg, R))
    return
end

# ----- Hessian structure -----
function hessian_structure_pmob(auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    region = graph.region
    rows = Int[]; cols = Int[]
    push_rc!(r, c) = (push!(rows, r); push!(cols, c))

    for j in 1:J, nn in 1:N, nd in 1:N
        r = col_C_pm(j, nn, J, R); c = col_C_pm(j, nd, J, R)
        r >= c && push_rc!(r, c)
    end
    for nn in 1:N, i in 1:ndeg
        cc = col_Q_pm(i, nn, J, N, ndeg, R); push_rc!(cc, cc)
    end
    for j in 1:J
        cL = col_L_pm(j, J, N, ndeg, R)
        push_rc!(cL, col_ur_pm(region[j]))   # (Lj, ur[region])
        push_rc!(cL, cL)                      # (Lj, Lj)
    end
    nt = n_pmob(J, N, ndeg, R)
    return struct_from_triplets(rows, cols, nt, nt)
end

# ----- Hessian values (objective linear; obj_factor unused) -----
function hessian_pmob(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32},
                      obj_factor::Float64, lambda::Vector{Float64},
                      values::Union{Nothing,Vector{Float64}}, auxdata)
    if values === nothing
        r, c = auxdata.hess_struct; rows .= r; cols .= c; return
    end
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    A = auxdata.edges.A; Apos = auxdata.edges.Apos; Aneg = auxdata.edges.Aneg
    kappa_ex = auxdata.kappa_ex
    sigma, beta, a, alpha = param.sigma, param.beta, param.a, param.alpha
    psigma = (sigma - 1) / sigma
    region = graph.region

    Cjn = reshape(view(x, R + 1:R + J * N), J, N)
    Qin = reshape(view(x, R + J * N + 1:R + J * N + ndeg * N), ndeg, N)
    Lj = view(x, R + J * N + ndeg * N + 1:n_pmob(J, N, ndeg, R))
    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)

    omegaj = lambda[1:J]
    Pjn = reshape(lambda[J + 1:J + J * N], J, N)

    nt = n_pmob(J, N, ndeg, R)
    Ir = Int[]; Jc = Int[]; V = Float64[]
    addv!(r, c, v) = (r >= c && (push!(Ir, r); push!(Jc, c); push!(V, v)))

    @inbounds for j in 1:J
        up = felicity_c(Cj[j], graph.Hj[j], alpha)
        us = felicity_cc(Cj[j], graph.Hj[j], alpha)
        pref = -(omegaj[j] * us * Cj[j]^(2 / sigma) + (1 / sigma) * omegaj[j] * up * Cj[j]^(2 / sigma - 1))
        for nn in 1:N, nd in 1:N
            r = col_C_pm(j, nn, J, R); c = col_C_pm(j, nd, J, R)
            r >= c || continue
            val = pref * Cjn[j, nn]^(-1 / sigma) * Cjn[j, nd]^(-1 / sigma)
            if nn == nd
                val += (1 / sigma) * omegaj[j] * up * Cj[j]^(1 / sigma) * Cjn[j, nn]^(-1 / sigma - 1)
            end
            addv!(r, c, val)
        end
    end
    # (Qin, Qin) diagonal: transport-cost curvature, weighted by origin-node price
    @inbounds for nn in 1:N
        ApP = Apos' * Pjn[:, nn]; AnP = Aneg' * Pjn[:, nn]
        for i in 1:ndeg
            q = Qin[i, nn]; aq = abs(q)
            wgt = q > 0 ? ApP[i] : AnP[i]
            hpow = aq == 0.0 ? (beta == 1.0 ? 1.0 : 0.0) : aq^(beta - 1)
            cc = col_Q_pm(i, nn, J, N, ndeg, R)
            addv!(cc, cc, (1 + beta) * beta * hpow / kappa_ex[i] * wgt)
        end
    end
    @inbounds for j in 1:J
        cL = col_L_pm(j, J, N, ndeg, R)
        addv!(cL, col_ur_pm(region[j]), omegaj[j])
        hl = 0.0
        for nn in 1:N
            hl += -a * (a - 1) * graph.Zjn[j, nn] * Pjn[j, nn] * Lj[j]^(a - 2)
        end
        addv!(cL, cL, hl)
    end

    fill_values_from_triplets!(values, auxdata.hess_struct, Ir, Jc, V, nt, nt)
    return
end

# ----- recover allocation -----
function recover_allocation_pmob(x, mult_g, auxdata)
    graph = auxdata.graph; param = auxdata.param
    J, N, ndeg, R = graph.J, param.N, graph.ndeg, graph.nregions
    psigma = (param.sigma - 1) / param.sigma

    ur = x[1:R]
    Cjn = reshape(x[R + 1:R + J * N], J, N)
    Qin = reshape(x[R + J * N + 1:R + J * N + ndeg * N], ndeg, N)
    Lj = x[R + J * N + ndeg * N + 1:n_pmob(J, N, ndeg, R)]

    Cj = dropdims(sum(Cjn .^ psigma, dims=2), dims=2) .^ (1 / psigma)
    Ljn = (graph.Zjn .> 0) .* Lj
    Yjn = graph.Zjn .* Lj .^ param.a
    cj = ifelse.(Lj .== 0, 0.0, Cj ./ Lj)
    hj = ifelse.(Lj .== 0, 0.0, graph.Hj ./ Lj)
    uj = param.u.(cj, hj)
    Qjkn = gen_network_flows(Qin, graph, N)
    Pjn = reshape(mult_g[J + 1:J + J * N], J, N)
    PCj = dropdims(sum(Pjn .^ (1 - param.sigma), dims=2), dims=2) .^ (1 / (1 - param.sigma))

    return (
        welfare = sum(graph.omegar .* graph.Lr .* ur),
        reg_pc_welfare = ur,
        Yjn = Yjn, Yj = dropdims(sum(Yjn, dims=2), dims=2),
        Cjn = Cjn, Cj = Cj, Ljn = Ljn, Lj = Lj,
        cj = cj, hj = hj, uj = uj,
        Pjn = Pjn, PCj = PCj, Qin = Qin, Qjkn = Qjkn,
    )
end
