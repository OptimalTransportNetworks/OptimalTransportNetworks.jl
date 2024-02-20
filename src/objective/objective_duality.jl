
function objective_duality(x, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex = auxdata.kappa_ex
    A = auxdata.A
    omegaj = param.omegaj
    Lj = param.Lj

    # Extract price vector
    Pjn = reshape(x, (graph.J, param.N))

    # Calculate consumption
    cj = param.alpha * (sum(Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma)) ./ omegaj) .^ (-1 / (1 + param.alpha * (param.rho - 1))) .* (param.hj / (1 - param.alpha)) .^ (-((1 - param.alpha) * (param.rho - 1) / (1 + param.alpha * (param.rho - 1))))
    zeta = omegaj .* ((cj / param.alpha) .^ param.alpha .* (param.hj / (1 - param.alpha)) .^ (1 - param.alpha)) .^ (-param.rho) .* ((cj / param.alpha) .^ (param.alpha - 1) .* (param.hj / (1 - param.alpha)) .^ (1 - param.alpha))
    cjn = (Pjn ./ zeta) .^ (-param.sigma) .* cj

    # Calculate Q, Qin_direct (which is the flow in the direction of the edge)
    # and Qin_indirect (flow in edge opposite direction)
    Qin_direct = zeros(graph.ndeg, param.N)
    Qin_indirect = zeros(graph.ndeg, param.N)
    for n = 1:param.N
        Qin_direct[:, n] = max.(1 / (1 + param.beta) * kappa_ex .* (-A' * Pjn[:, n] ./ (max.(A', 0) * Pjn[:, n])), 0) .^ (1 / param.beta)
    end
    for n = 1:param.N
        Qin_indirect[:, n] = max.(1 / (1 + param.beta) * kappa_ex .* (A' * Pjn[:, n] ./ (max.(-A', 0) * Pjn[:, n])), 0) .^ (1 / param.beta)
    end

    # Calculate labor allocation
    Ljn = ((Pjn .* param.Zjn) .^ (1 / (1 - param.a))) ./ repeat(sum((Pjn .* param.Zjn) .^ (1 / (1 - param.a)), dims=2), outer=[1, param.N]) .* repeat(Lj, outer=[1, param.N])
    Ljn[param.Zjn .== 0] = 0
    Yjn = param.Zjn .* (Ljn .^ param.a)

    # Create flow constraint
    cons = cjn .* repeat(param.Lj, outer=[1, param.N]) + A * Qin_direct - A * Qin_indirect - Yjn + max.(A, 0) * (Qin_direct .^ (1 + param.beta) ./ repeat(kappa_ex, outer=[1, param.N])) + max.(-A, 0) * (Qin_indirect .^ (1 + param.beta) ./ repeat(kappa_ex, outer=[1, param.N]))
    cons = sum(Pjn .* cons, dims=2)

    # Lagrangian
    f = sum(omegaj .* param.Lj .* ((cj / param.alpha) .^ param.alpha .* (param.hj / (1 - param.alpha)) .^ (1 - param.alpha)) .^ (1 - param.rho) / (1 - param.rho) - cons)

    return f
end


# Please note that in Julia, element-wise operations are denoted by `.` before the operator. Also, the `ones` function in MATLAB is replaced by `repeat` in Julia. The `dims` argument is used in Julia functions to specify the dimension along which the operation is performed.