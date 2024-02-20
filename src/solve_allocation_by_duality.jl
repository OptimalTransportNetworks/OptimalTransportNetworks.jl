
using Ipopt
using LinearAlgebra

function solve_allocation_by_duality(x0, auxdata, verbose=true)

    # Recover parameters
    graph = auxdata.graph
    param = auxdata.param

    if isempty(x0)
        x0 = vec(reshape(1:2, graph.J, param.N) * reshape(1:2, graph.J, param.N)')
    end

    # Init functions
    funcs = Dict(
        :objective => (x) -> objective(x, auxdata),
        :gradient => (x) -> gradient(x, auxdata),
        :hessian => (x, sigma_hess, lambda_hess) -> hessian(x, auxdata, sigma_hess, lambda_hess),
        :hessianstructure => () -> sparse(tril(repeat(I, param.N, param.N) + kron(I, graph.adjacency)))
    )

    # Options
    options = Dict(
        :lb => 1e-3 * ones(graph.J * param.N),
        :ub => Inf * ones(graph.J * param.N),
        :print_level => verbose ? 5 : 0
    )

    # Run Ipopt
    x, info = ipopt(x0, funcs, options)

    # Return results
    flag = info
    results = recover_allocation(x, auxdata)

    # Compute missing fields
    results.hj = param.hj
    results.Lj = param.Lj
    results.welfare = funcs[:objective](x)
    results.uj = param.u(results.cj, results.hj)

    return results, flag, x
end

function objective(x, auxdata)
    # Recover parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa = auxdata.kappa

    res = recover_allocation(x, auxdata)

    x = reshape(x, graph.J, param.N)

    cost = res.Qjkn .^ (1 + param.beta) ./ repeat(kappa, outer=(1, 1, param.N))
    cost[res.Qjkn .== 0] .= 0  # Deal with cases Qjkn=kappa=0

    cons = sum(x .* (res.cjn .* param.Lj .+ squeeze(sum(res.Qjkn + cost - permutedims(res.Qjkn, (2, 1, 3)), dims=2)) - param.Zjn .* param.F(res.Ljn, param.a)), dims=2)

    return sum(param.omegaj .* param.Lj .* param.u(res.cj, param.hj) - cons)
end

function gradient(x, auxdata)
    # Recover parameters
    param = auxdata.param
    kappa = auxdata.kappa

    res = recover_allocation(x, auxdata)

    cost = res.Qjkn .^ (1 + param.beta) ./ repeat(kappa, outer=(1, 1, param.N))
    cost[res.Qjkn .== 0] .= 0  # Deal with cases Qjkn=kappa=0

    cons = res.cjn .* param.Lj .+ squeeze(sum(res.Qjkn + cost - permutedims(res.Qjkn, (2, 1, 3)), dims=2)) - param.Zjn .* param.F(res.Ljn, param.a)

    return -cons[:]
end

function hessian(x, auxdata, sigma_hess, lambda_hess)
    # Recover parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa = auxdata.kappa

    # Recover allocation and format
    res = recover_allocation(x, auxdata)
    Lambda = repeat(x, 1, graph.J * param.N)
    lambda = reshape(x, graph.J, param.N)

    # Precalculations
    P = (sum(lambda .^ (1 - param.sigma), dims=2)) .^ (1 / (1 - param.sigma))  # Price index
    mat_P = repeat(P, param.N, graph.J * param.N)  # Matrix of price indices of size (JxN,JxN)
    Iij = repeat(I, param.N, param.N)  # Mask selecting only the (i,n;j,m) such that i=j
    Inm = kron(I, graph.adjacency)  # Mask selecting only the (i,n;j,m) such that j is in N(i) and n=m

    # Compute first diagonal terms coming from C^n_j: numerator
    termA = -param.sigma * (repeat(P, param.N) .^ param.sigma .* lambda[:] .^ -(param.sigma + 1) .* repeat(res.Cj, param.N))  # Diagonal vector of size JxN

    # Compute the non-diagonal terms from C^n_j: denominator
    termB = param.sigma * Iij .* Lambda .^ -param.sigma .* (Lambda') .^ -param.sigma .* mat_P .^ (2 * param.sigma - 1) .* repeat(res.Cj, param.N, param.N * graph.J)

    # Compute the non-diagonal terms from C^n_j: term in L_i c_i
    termC = Iij .* Lambda .^ -param.sigma .* (Lambda') .^ -param.sigma .* mat_P .^ (2 * param.sigma) .* repeat(param.Lj ./ (param.omegaj .* param.usecond(res.cj, param.hj)), param.N, graph.J * param.N)

    # Compute the non-diagonal terms from the constraint
    diff = Lambda' - Lambda
    mat_kappa = repeat(kappa, param.N, param.N)
    termD = 1 / (param.beta * (1 + param.beta) ^ (1 / param.beta)) * Inm .* mat_kappa .^ (1 / param.beta) .* abs(diff) .^ (1 / param.beta - 1) .* ((diff .> 0) .* Lambda' ./ Lambda .^ (1 + 1 / param.beta) + (diff .< 0) .* Lambda ./ (Lambda') .^ (1 + 1 / param.beta))

    # Compute the diagonal terms from the constraint
    termE = -1 / (param.beta * (1 + param.beta) ^ (1 / param.beta)) * Inm .* mat_kappa .^ (1 / param.beta) .* abs(diff) .^ (1 / param.beta - 1) .* ((diff .> 0) .* (Lambda') .^ 2 ./ Lambda .^ (2 + 1 / param.beta) + (diff .< 0) .* 1 ./ (Lambda') .^ (1 / param.beta))
    termE = sum(termE, dims=2)

    # Compute the term coming from Lj
    if param.a == 1
        X = 0
    else
        denom = sum((lambda .* param.Zjn) .^ (1 / (1 - param.a)), dims=2)
        Lambdaz = repeat(lambda[:] .* param.Zjn[:], 1, graph.J * param.N)
        X_nondiag = param.a / (1 - param.a) * Iij .* repeat(param.Zjn[:], 1, graph.J * param.N) .* repeat(param.Zjn[:]', graph.J * param.N, 1) .* repeat((param.Lj ./ denom) .^ param.a, param.N, graph.J * param.N) .* Lambdaz .^ (param.a / (1 - param.a)) .* (Lambdaz') .^ (param.a / (1 - param.a))
        X_diag = -param.a / (1 - param.a) * repeat((param.Lj ./ denom) .^ param.a, param.N) .* param.Zjn[:] ./ lambda[:] .* (lambda[:] .* param.Zjn[:]) .^ (param.a / (1 - param.a))
        X = X_nondiag + Diagonal(X_diag)
    end

    # Return hessian
    return -sigma_hess * (Diagonal(termA) + termB + termC + termD + Diagonal(termE) + X)
end

function recover_allocation(x, auxdata)
    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    omegaj = param.omegaj
    kappa = auxdata.kappa
    Lj = param.Lj

    # Extract price vectors
    Pjn = reshape(x, graph.J, param.N)
    results = Dict(:Pjn => Pjn, :PCj => (sum(Pjn .^ (1 - param.sigma), dims=2)) .^ (1 / (1 - param.sigma)))

    # Calculate labor allocation
    if param.a < 1
        results[:Ljn] = ((Pjn .* param.Zjn) .^ (1 / (1 - param.a))) ./ repeat(sum((Pjn .* param.Zjn) .^ (1 / (1 - param.a)), dims=2), 1, param.N) .* repeat(Lj, 1, param.N)
        results[:Ljn][param.Zjn .== 0] .= 0
    else
        max_id = argmax(Pjn .* param.Zjn, dims=2)
        results[:Ljn] = zeros(graph.J, param.N)
        results[:Ljn][sub2ind(size(results[:Ljn]), 1:graph.J, max_id)] = param.Lj
    end
    results[:Yjn] = param.Zjn .* (results[:Ljn] .^ param.a)

    # Calculate consumption
    results[:cj] = param.alpha * (sum(Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma)) ./ omegaj) .^ (-1 / (1 + param.alpha * (param.rho - 1))) .* (param.hj / (1 - param.alpha)) .^ -((1 - param.alpha) * (param.rho - 1) / (1 + param.alpha * (param.rho - 1)))
    zeta = omegaj .* ((results[:cj] / param.alpha) .^ param.alpha .* (param.hj / (1 - param.alpha)) .^ (1 - param.alpha)) .^ -param.rho .* ((results[:cj] / param.alpha) .^ (param.alpha - 1) .* (param.hj / (1 - param.alpha)) .^ (1 - param.alpha))
    results[:cjn] = (Pjn ./ repeat(zeta, 1, param.N)) .^ -param.sigma .* repeat(results[:cj], 1, param.N)
    results[:Cj] = results[:cj] .* param.Lj
    results[:Cjn] = results[:cjn] .* repeat(param.Lj, 1, param.N)

    # Calculate the flows Qjkn of dimension [J,J,N]
    results[:Qjkn] = zeros(graph.J, graph.J, param.N)
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = max.(Lambda' .- Lambda, 0)
        LL[.!graph.adjacency] .= 0
        results[:Qjkn][:, :, n] = (1 / (1 + param.beta) * kappa .* LL ./ Lambda) .^ (1 / param.beta)
    end

    return results
end


# Please note that the `ipopt` function used in the original Matlab code is from the Ipopt library, which is a solver for large scale nonlinear optimization problems. The equivalent function in Julia is also called `ipopt` and is provided by the Ipopt.jl package. The syntax and usage are similar but not identical, so the code may need to be adjusted accordingly.