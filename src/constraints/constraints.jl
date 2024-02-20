
"""
    constraints(x, auxdata)

Constraint function used by ADiGator in the primal case, no mobility, 
no cross-good congestion.
"""
function constraints(x, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex = auxdata.kappa_ex
    A = auxdata.A
    Lj = param.Lj

    # Recover variables
    Cjn = reshape(x[1:graph.J*param.N], graph.J, param.N)
    Qin = reshape(x[graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N], graph.ndeg, param.N)
    Ljn = reshape(x[graph.J*param.N+graph.ndeg*param.N+1:end], graph.J, param.N)
    Yjn = param.Zjn .* (Ljn .^ param.a)

    # Balanced flow constraints
    cons_Q = zeros(graph.J, param.N)
    for n in 1:param.N
        M = max.(A .* (ones(graph.J, 1) * sign.(Qin[:, n])'), 0) # Matrix of dimension [J,Ndeg] taking value 1 if node J sends a good through edge Ndeg and 0 else 
        cons_Q[:, n] = Cjn[:, n] + A * Qin[:, n] - Yjn[:, n] + M * (abs.(Qin[:, n]) .^ (1 + param.beta) ./ kappa_ex)
    end

    # Local labor constraints
    cons_Ljn = sum(Ljn, dims=2) .- Lj

    # return whole vector of constraints
    return vcat(cons_Q[:], cons_Ljn)
end


# Please note that in Julia, array indexing starts from 1, not 0 as in Matlab. Also, element-wise operations are denoted by a dot (.) before the operator. The `end` keyword is used to denote the end of a block of code, such as a for loop or a function. The `reshape` function in Julia works the same way as in Matlab, reshaping an array into a specified size. The `vcat` function concatenates arrays vertically.