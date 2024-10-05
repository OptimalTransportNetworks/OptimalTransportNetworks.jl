# Note: See their supplementary material for details

function model_fixed_duality_cgc(optimizer, auxdata)
    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex_init = auxdata.kappa_ex
    A = auxdata.edges.A
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    es = auxdata.edges.edge_start
    ee = auxdata.edges.edge_end
    omegaj = graph.omegaj
    Lj = graph.Lj
    alpha = param.alpha
    sigma = param.sigma
    rho = param.rho
    hj = graph.hj
    hj1malpha = (hj / (1-alpha)) .^ (1-alpha)
    Zjn = graph.Zjn
    beta = param.beta
    a = param.a

    model = Model(optimizer) # add_bridges = false
    set_string_names_on_creation(model, false)

    # Define price vector variable Pjn
    @variable(model, Pjn[1:graph.J, 1:param.N] >= 1e-6, container=Array)
    # Generate starting values
    v1 = range(1, 2, length=graph.J)
    v2 = param.N == 1 ? 2.0 : range(1, 2, length=param.N)
    x0 = v1 * v2' # vec()
    set_start_value.(Pjn, x0)

    # Parameters: to be updated between solves
    @variable(model, kappa_ex[i = 1:graph.ndeg] in Parameter(i), container=Array)
    set_parameter_value.(kappa_ex, kappa_ex_init)

    # Calculate aggregate price P^D(j)
    @expression(model, PDj[j=1:graph.J], sum(Pjn[j, n]^(1-sigma) for n=1:param.N)^(1/(1-sigma)))

    # Calculate consumption cj
    @expression(model, cj[j=1:graph.J],
        alpha * (PDj[j] / omegaj[j])^(-1/(1+alpha*(rho-1))) * hj1malpha[j]^(-((rho-1)/(1+alpha*(rho-1))))
    )
    # Utility per worker in location j
    @expression(model, uj[j=1:graph.J], ((cj[j]/alpha)^alpha * hj1malpha[j])^(1-rho)/(1-rho))
    # zeta (auxiliarly variable)
    zeta = @expression(model, [j=1:graph.J], omegaj[j] * uj[j] * (1-rho) * alpha / cj[j])

    # cross-good congestion: Eq. 11 in the main paper
    # Calculate Q, Qi_direct and Qi_indirect
    # F&S: Condition (8) implies that goods in each sector flow in only one direction, although a link may have flows in opposite directions corresponding to different sectors.
    @expression(model, Qi_direct[i=1:graph.ndeg],  # Flow in the direction of the edge
        maximum(((Pjn[ee[i],:]-Pjn[es[i],:]) ./ param.m) ./ ((1+beta) * PDj[es[i]] / kappa_ex[i]))^(1/beta)
    )
    @expression(model, Qi_indirect[i=1:graph.ndeg], # Flow in edge opposite direction
        maximum(((Pjn[es[i],:]-Pjn[ee[i],:]) ./ param.m) ./ ((1+beta) * PDj[ee[i]] / kappa_ex[i]))^(1/beta)
    )

    B_direct = @expression(model, Apos * (Qi_direct .^ (1+beta) ./ kappa_ex))     # 1:J vector
    B_indirect = @expression(model, Aneg * (Qi_indirect .^ (1+beta) ./ kappa_ex)) # 1:J vector
    # Calculate consumption pre transport cost D(j,n)
    @expression(model, Djn[j=1:graph.J, n=1:param.N], (Pjn[j, n] / zeta[j])^(-sigma) * (cj[j] * Lj[j] + B_direct[j] + B_indirect[j]))

    # Calculate labor allocation Ljn
    PZ = @expression(model, (Pjn .* Zjn) .^ (1/(1-a)))
    @expression(model, Ljn[j=1:graph.J, n=1:param.N],
        ifelse(Zjn[j,n] == 0, 0, PZ[j, n] / sum(PZ[j, n] for n=1:param.N) * Lj[j])
    )
    # Calculate production Yjn
    @expression(model, Yjn, Zjn .* Ljn .^ a)
    # # Create flow constraint expression
    # @expression(model, cons[j=1:graph.J], # , n=1:param.N
    #     sum(Djn[j,n] - Yjn[j,n] for n=1:param.N) + sum(A[j,i] * (Qi_direct[i] - Qi_indirect[i]) for i=1:graph.ndeg)
    # )

    # Better: add a 1:J constraint that satisfies the linear system, incorporating the flow constraint
    # @constraint(model, A * (Qi_direct - Qi_indirect) - ((Yjn - Djn) .^ param.nu * param.m) .^ (1/param.nu) .<= -1e-8)

    # Implicit definition 
    # @expression(model, [i=1:graph.ndeg], Qi_direct[i] = sum(param.m[n] * Qin_direct[i,n]^param.nu for n=1:param.N)^(1/param.nu))
    # @expression(model, [i=1:graph.ndeg], Qi_indirect[i] = sum(param.m[n] * Qin_indirect[i,n]^param.nu for n=1:param.N)^(1/param.nu))
    # @expression(model, cons[j=1:graph.J, n=1:param.N],
    #     Djn[j,n] - Yjn[j,n] + sum(A[j,i] * (Qin_direct[i,n] - Qin_indirect[i,n]) for i=1:graph.ndeg)
    # )
    @variable(model, Qin[1:graph.ndeg, 1:param.N], container=Array)
    @constraint(model, [i=1:graph.ndeg], Qi_direct[i] - Qi_indirect[i] == sum(param.m[n] * Qin[i,n]^param.nu for n=1:param.N)^(1/param.nu))
    @expression(model, cons[j=1:graph.J, n=1:param.N],
        Djn[j,n] - Yjn[j,n] + sum(A[j,i] * Qin[i, n] for i=1:graph.ndeg)
    )

    # Define the Lagrangian objective
    @expression(model, U, sum(omegaj .* Lj .* uj) - sum(Pjn .* cons))
    @objective(model, Min, U)
    
    return model
end

function recover_allocation_fixed_duality_cgc(model, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:U])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = dropdims(sum(results[:Yjn], dims=2), dims = 2)
    results[:Djn] = value.(model_dict[:Djn]) 
    results[:Dj] = dropdims(sum(results[:Djn], dims=2), dims = 2)
    results[:cj] = value.(model_dict[:cj])
    results[:Cj] = results[:cj] .* graph.Lj
    results[:Ljn] = value.(model_dict[:Ljn])
    results[:Lj] = graph.Lj
    results[:hj] = graph.hj
    results[:uj] = value.(model_dict[:uj]) # param.u.(results[:cj], results[:hj])
    # Prices
    results[:Pjn] = value.(model_dict[:Pjn])
    results[:PCj] = dvalue.(model_dict[:PDj])
    # Network flows
    results[:Qin] = value.(model_dict[:Qin])
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end