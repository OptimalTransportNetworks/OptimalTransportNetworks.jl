
using LinearAlgebra

function constraints_mobility(x, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex = auxdata.kappa_ex
    A = auxdata.A

    # Extract optimization variables
    u = x[1]
    Cjn = reshape(x[2:graph.J*param.N+1], graph.J, param.N)
    Qin = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)
    Lj = x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+graph.ndeg*param.N+graph.J+1]
    Cj = (sum(Cjn.^((param.sigma-1)/param.sigma), dims=2)).^(param.sigma/(param.sigma-1))
    Ljn = reshape(x[graph.J*param.N+graph.ndeg*param.N+graph.J+2:end], graph.J, param.N)
    Yjn = param.Zjn .* Ljn.^param.a

    # Utility constraint (Lj*u <= ... )
    cons_u = Lj .* u - (Cj / param.alpha).^param.alpha .* (param.Hj / (1 - param.alpha)).^(1 - param.alpha)

    # balanced flow constraints
    cons_Q = zeros(graph.J, param.N)
    for n in 1:param.N
        M = max.(A .* (ones(graph.J, 1) * sign.(Qin[:, n]))', 0)
        cons_Q[:, n] = Cjn[:, n] + A * Qin[:, n] - Yjn[:, n] + M * (abs.(Qin[:, n]).^(1 + param.beta) ./ kappa_ex)
    end

    # labor resource constraint
    cons_L = sum(Lj) - 1

    # Local labor availability constraints ( sum Ljn <= Lj )
    cons_Ljn = sum(Ljn, dims=2) .- Lj

    # return whole vector of constraints
    cons = vcat(cons_u[:], cons_Q[:], cons_L, cons_Ljn)

    return cons
end


# Please note that in Julia, array indexing starts from 1, not 0 as in Matlab. Also, the `end` keyword in Julia is equivalent to Matlab's `end`. The `dims` argument in functions like `sum` and `max` specifies the dimensions to operate over. The `.` before functions like `*` and `^` is used for element-wise operations. The `vcat` function concatenates arrays vertically.