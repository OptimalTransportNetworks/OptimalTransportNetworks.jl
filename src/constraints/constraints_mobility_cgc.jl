
using LinearAlgebra

function constraints_mobility_cgc(x, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex = auxdata.kappa_ex
    A = auxdata.A
    m = param.m

    # Extract optimization variables
    u = x[1]
    Djn = reshape(x[2:graph.J*param.N+1], graph.J, param.N) # Consumption per good pre-transport cost
    Dj = (sum(Djn.^((param.sigma-1)/param.sigma), dims=2)).^(param.sigma/(param.sigma-1))  # Aggregate consumption pre-transport cost
    Qin_direct = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N) # Flow in the direction of the edge
    Qin_indirect = reshape(x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+2*graph.ndeg*param.N+1], graph.ndeg, param.N) # Flow in edge opposite direction
    Lj = x[graph.J*param.N+2*graph.ndeg*param.N+2:graph.J*param.N+2*graph.ndeg*param.N+graph.J+1]
    cj = x[graph.J*param.N+2*graph.ndeg*param.N+graph.J+2:graph.J*param.N+2*graph.ndeg*param.N+2*graph.J+1]
    Ljn = reshape(x[graph.J*param.N+2*graph.ndeg*param.N+2*graph.J+2:end], graph.J, param.N)
    Yjn = param.Zjn.*(Ljn.^param.a)

    # Utility constraint (Lj*u <= ... )
    cons_u = Lj.*u .- (cj.*Lj/param.alpha).^param.alpha .* (param.Hj/(1-param.alpha)).^(1-param.alpha)

    # balanced flow constraint
    cons_Q = Djn + A*Qin_direct - A*Qin_indirect - Yjn

    # labor resource constraint
    cons_L = sum(Lj) - 1

    # Final good constraint

    # Create the matrix B_direct (resp. B_indirect) of transport cost along the direction of the edge 
    # (resp. in edge opposite direction)
    B_direct = (sum(repeat(m', graph.ndeg).*Qin_direct.^param.nu, dims=2)).^((param.beta+1)/param.nu)./kappa_ex
    B_indirect = (sum(repeat(m', graph.ndeg).*Qin_indirect.^param.nu, dims=2)).^((param.beta+1)/param.nu)./kappa_ex
    # Write final good constraint
    cons_c = cj.*Lj + max.(A, 0)*B_direct + max.(-A, 0)*B_indirect - Dj

    # Local labor constraint
    cons_Ljn = sum(Ljn, dims=2) - Lj 

    # return whole vector of constraints
    cons = vcat(cons_u[:], cons_Q[:], cons_L, cons_c, cons_Ljn)

    return cons
end


# Please note that in Julia, array indexing starts from 1, not 0 as in Matlab. Also, element-wise operations are denoted by `.` before the operator. The `reshape` function in Julia takes the new dimensions as separate arguments, not as a vector. The `max` function is replaced by `max.` for element-wise operation. The `sum` function takes `dims` as a keyword argument to specify the dimensions to operate over. The `repeat` function is used instead of `repmat`. The `vcat` function is used to concatenate arrays vertically.