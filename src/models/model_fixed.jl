# The primal case, no mobility, no cross-good congestion

function model_fixed(optimizer, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex_init = auxdata.kappa_ex
    A = auxdata.edges.A
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    psigma = (param.sigma - 1) / param.sigma
    Lj = graph.Lj

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variables
    @variable(model, Cjn[1:graph.J, 1:param.N] >= 1e-8, container=Array, start = 1e-6)
    @variable(model, Qin[1:graph.ndeg, 1:param.N], container=Array, start = 0.0)
    @variable(model, Ljn[1:graph.J, 1:param.N] >= 1e-8, container=Array, start = sum(Lj)/(graph.J*param.N))

    # Parameters: to be updated between solves
    @variable(model, kappa_ex[i = 1:graph.ndeg] in Parameter(i), container=Array)
    set_parameter_value.(kappa_ex, kappa_ex_init)

    # Defining Utility Funcion: from Cjn + parameters (by operator overloading)
    @expression(model, Cj, sum(Cjn .^ psigma, dims=2) .^ (1 / psigma))
    @expression(model, cj, ifelse.(graph.Lj .== 0, 0.0, Cj ./ graph.Lj))
    @expression(model, uj, ((cj / param.alpha) .^ param.alpha .* (graph.hj / (1-param.alpha)) .^ (1-param.alpha)) .^ (1-param.rho) / (1-param.rho))
    @expression(model, U, sum(graph.omegaj .* graph.Lj .* uj))
    @objective(model, Max, U)

    # Define Yjn (production) as expression
    @expression(model, Yjn[j=1:graph.J, n=1:param.N], graph.Zjn[j, n] * Ljn[j, n]^param.a)
    # Balanced flow constraints
    @constraint(model, Pjn[j in 1:param.J, n in 1:param.N],
        Cjn[j, n] + sum(A[j, i] * Qin[i, n] for i in 1:graph.ndeg) -
        Yjn[j, n] + sum(
            ifelse(Qin[i, n] > 0, Apos[j, i], Aneg[j, i]) *
            abs(Qin[i, n])^(1 + param.beta) / kappa_ex[i]
            for i in 1:graph.ndeg
        ) <= 0
    )

    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, sum(Ljn, dims=2) .- Lj .<= 0)

    return model
end

function recover_allocation_fixed(model, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:U])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = dropdims(sum(results[:Yjn], dims=2), dims = 2)
    results[:Cjn] = value.(model_dict[:Cjn])
    results[:Cj] = dropdims(value.(model_dict[:Cj]), dims = 2)
    results[:Ljn] = value.(model_dict[:Ljn])
    results[:Lj] = graph.Lj
    results[:cj] = dropdims(value.(model_dict[:cj]), dims = 2)
    results[:hj] = ifelse.(results[:Lj] .== 0, 0.0, graph.Hj ./ results[:Lj])
    results[:uj] = dropdims(value.(model_dict[:uj]), dims = 2)
    # Prices
    results[:Pjn] = shadow_price.(model_dict[:Pjn])
    results[:PCj] = dropdims(sum(results[:Pjn] .^ (1-param.sigma), dims=2), dims = 2) .^ (1/(1-param.sigma))    
    # Network flows
    results[:Qin] = value.(model_dict[:Qin])
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end