# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# Shared helpers for the direct-Ipopt primal solvers
# ==============================================================
#
# These back the hand-coded primal allocation solvers (solve_allocation_mobility,
# _cgc, _partial_mobility, _primal, ...) that call Ipopt's C interface directly,
# mirroring the dual solvers but with constraints.

# Cobb-Douglas felicity W(c,h) = (c/alpha)^alpha * (h/(1-alpha))^(1-alpha) and its
# first/second derivatives w.r.t. c. The mobility and partial-mobility utility
# constraints use this felicity directly (the rho = 0 form), independent of
# param.rho, so the solvers call these rather than param.uprime / param.usecond.
@inline felicity(c, h, alpha) = (c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha)
@inline felicity_c(c, h, alpha) = (c / alpha)^(alpha - 1) * (h / (1 - alpha))^(1 - alpha)
@inline felicity_cc(c, h, alpha) = (alpha - 1) / alpha * (c / alpha)^(alpha - 2) * (h / (1 - alpha))^(1 - alpha)

# Fetch the Jacobian/Hessian sparsity structures, computing them once and caching them when
# optimal_network supplies a per-run `struct_cache`. The structures depend only on the graph
# (not on kappa or x), so this avoids rebuilding them on every outer iteration's solve.
# `jacf`/`hessf` are zero-arg closures that compute the respective structure.
function get_structs(auxdata, jacf, hessf)
    if haskey(auxdata, :struct_cache)
        c = auxdata.struct_cache
        return get!(jacf, c, :jac), get!(hessf, c, :hess)
    end
    return jacf(), hessf()
end

# Build the deduplicated, column-major (rows, cols) sparsity structure from triplet
# lists, plus a map from column-major linear index -> position in (rows, cols). The
# map lets the value callbacks scatter their entries onto the declared structure
# WITHOUT building a sparse matrix inside the Ipopt C callback (allocating a sparse
# matrix from within the @cfunction trampoline corrupts the heap). Returns
# (rows::Vector{Int32}, cols::Vector{Int32}, lin2pos::Dict{Int,Int}).
function struct_from_triplets(rows, cols, m, n)
    nt = length(rows)
    lin = Vector{Int}(undef, nt)
    @inbounds for t in 1:nt
        lin[t] = (Int(cols[t]) - 1) * m + Int(rows[t])
    end
    order = sortperm(lin)   # column-major order
    r_out = Int32[]; c_out = Int32[]
    lin2pos = Dict{Int,Int}()
    last = -1
    @inbounds for t in order
        if lin[t] != last
            push!(r_out, rows[t]); push!(c_out, cols[t])
            lin2pos[lin[t]] = length(r_out)
            last = lin[t]
        end
    end
    return (r_out, c_out, lin2pos)
end

# Scatter triplet values (Ir, Jc, V; overlapping entries sum) onto the values buffer
# in the order of the precomputed structure, using the linear-index -> position map.
# No sparse-matrix allocation (heap-safe inside the Ipopt callback).
function fill_values_from_triplets!(values, struct_rc, Ir, Jc, V, m, n)
    lin2pos = struct_rc[3]
    fill!(values, 0.0)
    @inbounds for t in eachindex(V)
        values[lin2pos[(Int(Jc[t]) - 1) * m + Int(Ir[t])]] += V[t]
    end
    return
end

# Create, configure, and solve an Ipopt problem via the C interface, REUSING the
# problem object across outer iterations when a per-run `cache` is supplied (keyed by
# `key`). On the first call the problem is built and stored together with its stable
# auxdata `saux` (whose `kappa_ex` the callbacks captured); on later calls only
# `saux.kappa_ex` is updated in place and the solver re-runs warm-started from the
# previous primal+dual solution. This avoids rebuilding the problem and the cold dual
# restart each iteration — the same trick JuMP uses by reusing its model.
#
# `saux` is the stable auxdata the callbacks (obj/cons/grad/jac/hess) capture. Returns
# (x, status, mult_g); the recover functions only need graph/param/x/mult_g, so callers
# pass their original auxdata to recover (its kappa is irrelevant there).
function run_ipopt_primal(cache, key, saux, n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                          obj, cons, grad, jac, hess, x0, param, verbose; max_iter=2000)
    if cache !== nothing && haskey(cache, key)
        prob, saux0 = cache[key]
        saux0.kappa_ex .= saux.kappa_ex      # propagate the updated network to the cached callbacks
        if isempty(x0)
            Ipopt.AddIpoptStrOption(prob, "warm_start_init_point", "no")  # cold restart requested
        else
            prob.x = x0
            Ipopt.AddIpoptStrOption(prob, "warm_start_init_point", "yes") # reuse previous primal+dual point
        end
        status = Ipopt.IpoptSolve(prob)
        return prob.x, status, prob.mult_g
    end

    prob = Ipopt.CreateIpoptProblem(
        n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
        obj, cons, grad, jac, hess
    )
    Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "exact")
    Ipopt.AddIpoptIntOption(prob, "max_iter", max_iter)
    Ipopt.AddIpoptIntOption(prob, "print_level", verbose ? 5 : 0)
    # Tight warm-start pushes so reused solves start essentially at the previous point.
    Ipopt.AddIpoptNumOption(prob, "warm_start_bound_push", 1e-9)
    Ipopt.AddIpoptNumOption(prob, "warm_start_mult_bound_push", 1e-9)
    Ipopt.AddIpoptNumOption(prob, "warm_start_bound_frac", 1e-9)

    if haskey(param, :optimizer_attr)
        for (k, value) in param.optimizer_attr
            if value isa String
                Ipopt.AddIpoptStrOption(prob, String(k), value)
            elseif value isa Float64
                Ipopt.AddIpoptNumOption(prob, String(k), value)
            elseif value isa Integer
                Ipopt.AddIpoptIntOption(prob, String(k), value)
            end
        end
    end

    prob.x = x0
    status = Ipopt.IpoptSolve(prob)
    if cache !== nothing
        cache[key] = (prob, saux)
    end
    return prob.x, status, prob.mult_g
end
