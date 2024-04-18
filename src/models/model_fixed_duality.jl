# Note: See their supplementary material for details

function model_fixed_duality(optimizer, auxdata)
    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex_init = auxdata.kappa_ex
    A = auxdata.edges.A
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    es = auxdata.edges.edge_start
    ee = auxdata.edges.edge_end
    omegaj = param.omegaj
    Lj = param.Lj
    alpha = param.alpha
    sigma = param.sigma
    rho = param.rho
    hj = param.hj
    hj1malpha = (hj / (1-alpha)) .^ (1-alpha)
    Zjn = param.Zjn
    beta = param.beta
    a = param.a

    model = Model(optimizer; add_bridges = false)
    set_string_names_on_creation(model, false)

    # Define price vector variable Pjn
    @variable(model, Pjn[1:graph.J, 1:param.N] >= 1e-6, container=Array)
    # Generate starting values
    v1 = range(1, 2, length=graph.J)
    v2 = param.N == 1 ? 2.0 : range(1, 2, length=param.N)
    x0 = vec(v1 * v2')
    set_start_value.(Pjn, x0)

    # Parameters: to be updated between solves
    @variable(model, kappa_ex[i = 1:graph.ndeg] in Parameter(i), container=Array)
    set_parameter_value.(kappa_ex, kappa_ex_init)

    # Calculate consumption cj
    @expression(model, cj[j=1:graph.J],
        alpha * (sum(Pjn[j, n]^(1-sigma) for n=1:param.N)^(1/(1-sigma)) / omegaj[j])^(-1/(1+alpha*(rho-1))) * hj1malpha[j]^(-((rho-1)/(1+alpha*(rho-1))))
    )
    # Utility per worker in location j
    @expression(model, uj[j=1:graph.J], ((cj[j]/alpha)^alpha * hj1malpha[j])^(1-rho)/(1-rho))
    # zeta (auxiliarly variable)
    zeta = @expression(model, [j=1:graph.J], omegaj[j] * uj[j] * (1-rho) * alpha / cj[j])
    # Calculate consumption c(j,n)
    @expression(model, cjn[j=1:graph.J, n=1:param.N], (Pjn[j, n] / zeta[j])^(-sigma) * cj[j])

    # No cross-good congestion: Eq. 11 in the main paper
    # Calculate Q, Qin_direct and Qin_indirect
    # F&S: Condition (8) implies that goods in each sector flow in only one direction, although a link may have flows in opposite directions corresponding to different sectors.
    @expression(model, Qin_direct[i=1:graph.ndeg, n=1:param.N],  # Flow in the direction of the edge
        max(1/(1+beta) * kappa_ex[i] * (Pjn[ee[i],n]/Pjn[es[i],n] - 1), 0)^(1/beta)
    )
    @expression(model, Qin_indirect[i=1:graph.ndeg, n=1:param.N], # Flow in edge opposite direction
        max(1/(1+beta) * kappa_ex[i] * (Pjn[es[i],n]/Pjn[ee[i],n] - 1), 0)^(1/beta)
    )
    # -> Seems here we let the size of the flow decide the direction NOT NEEDED!
    @expression(model, Qin[i=1:graph.ndeg, n=1:param.N],
        ifelse(Qin_direct[i,n] > Qin_indirect[i,n], Qin_direct[i,n], -Qin_indirect[i,n])
    )
    # Calculate labor allocation Ljn
    PZ = @expression(model, (Pjn .* Zjn) .^ (1/(1-a)))
    @expression(model, Ljn[j=1:graph.J, n=1:param.N],
        ifelse(Zjn[j,n] == 0, 0, PZ[j, n] / sum(PZ[j, n] for n=1:param.N) * Lj[j])
    )
    # Calculate production Yjn
    @expression(model, Yjn, Zjn .* Ljn .^ a)
    # Create flow constraint expression
    @expression(model, cons[j=1:graph.J, n=1:param.N],
        cjn[j,n] * Lj[j] - Yjn[j,n] + sum(A[j,i] * (Qin_direct[i,n] - Qin_indirect[i,n]) + 
            (Apos[j,i] * Qin_direct[i,n]^(1+beta) + Aneg[j,i] * Qin_indirect[i,n]^(1+beta)) / kappa_ex[i] for i=1:graph.ndeg)
        # sum(A[j,i] * Qin_direct[i,n] for i=1:graph.ndeg) -
        # sum(A[j,i] * Qin_indirect[i,n] for i=1:graph.ndeg) +
        # sum(Apos[j,i] * Qin_direct[i,n]^(1+beta) / kappa_ex[i] for i=1:graph.ndeg) +
        # sum(Aneg[j,i] * Qin_indirect[i,n]^(1+beta) / kappa_ex[i] for i=1:graph.ndeg)
    )
    # Define the Lagrangian objective
    @expression(model, U, sum(omegaj .* Lj .* uj) - sum(Pjn .* cons))
    @objective(model, Max, U)
    
    return model
end

function recover_allocation_fixed_duality(model, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:U])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = dropdims(sum(results[:Yjn], dims=2), dims = 2)
    results[:Cjn] = value.(model_dict[:cjn]) .* param.Lj'
    results[:cj] = value.(model_dict[:cj])
    results[:Cj] = results[:cj] .* param.Lj
    results[:Ljn] = value.(model_dict[:Ljn])
    results[:Lj] = param.Lj
    results[:hj] = param.hj
    results[:uj] = value.(model_dict[:uj]) # param.u.(results[:cj], results[:hj])
    # Prices
    results[:Pjn] = value.(model_dict[:Pjn])
    results[:PCj] = dropdims(sum(results[:Pjn] .^ (1-param.sigma), dims=2), dims = 2) .^ (1/(1-param.sigma))    
    # Network flows
    results[:Qin] = value.(model_dict[:Qin])
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end