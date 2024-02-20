
"""
constraints_cgc(x, auxdata)

Constraint function used by ADiGator in the primal case, no mobility, 
with cross-good congestion.
"""
function constraints_cgc(x, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex = auxdata.kappa_ex
    A = auxdata.A
    m = param.m

    # Recover labor allocation
    Ljn = reshape(x[graph.J*param.N+2*graph.ndeg*param.N+graph.J+1:end], graph.J, param.N)
    Yjn = param.Zjn .* (Ljn .^ param.a)

    # Extract optimization variables
    Qin_direct = reshape(x[graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N], graph.ndeg, param.N) # Flow in the direction of the edge
    Qin_indirect = reshape(x[graph.J*param.N+graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N], graph.ndeg, param.N) # Flow in edge opposite direction
    Djn = reshape(x[1:graph.J*param.N], graph.J, param.N) # Consumption per good pre-transport cost
    Dj = sum(Djn .^ ((param.sigma-1)/param.sigma), dims=2) .^ (param.sigma/(param.sigma-1)) # Aggregate consumption pre-transport cost
    cj = x[graph.J*param.N+2*graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N+graph.J]

    # Final good constraint

    # Create the matrix B_direct (resp. B_indirect) of transport cost along the direction of the edge 
    # (resp. in edge opposite direction)
    B_direct = (sum(repeat(m', outer=[graph.ndeg, 1]) .* Qin_direct .^ param.nu, dims=2)) .^ ((param.beta+1)/param.nu) ./ kappa_ex
    B_indirect = (sum(repeat(m', outer=[graph.ndeg, 1]) .* Qin_indirect .^ param.nu, dims=2)) .^ ((param.beta+1)/param.nu) ./ kappa_ex

    # Write final good constraint
    cons_C = cj .* param.Lj + max.(A, 0) * B_direct + max.(-A, 0) * B_indirect - Dj

    # Balanced flow constraint
    cons_Q = Djn + A * Qin_direct - A * Qin_indirect - Yjn

    # Labor allocation constraint
    cons_Ljn = sum(Ljn, dims=2) - param.Lj

    # return whole vector of constraints
    cons = [cons_Q[:]; cons_C; cons_Ljn]

    return cons
end


# Please note that in Julia, the `max` function is not vectorized, so we use `max.` instead. Similarly, other mathematical operations are also vectorized using the dot (.) operator. The `repeat` function in Julia is used instead of `repmat` in MATLAB. The `dims` argument is used in the `sum` function to specify the dimension along which the sum is computed.