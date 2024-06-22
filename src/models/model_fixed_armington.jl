# The primal case, no mobility, no cross-good congestion, and an Armington (1969) world where each location produces only one good

function model_fixed_armington(optimizer, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex_init = auxdata.kappa_ex
    A = auxdata.edges.A
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    psigma = (param.sigma - 1) / param.sigma
    Lj = param.Lj
    # Production: Fixed because just one good is produced by each location
    Yj = dropdims(sum(param.Zjn, dims = 2) .* Lj .^ param.a, dims = 2)

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variables
    @variable(model, Cjn[1:graph.J, 1:param.N] >= 1e-8, container=Array, start = 1e-6)
    @variable(model, Qin[1:graph.ndeg, 1:param.N], container=Array, start = 0.0)

    # Parameters: to be updated between solves
    @variable(model, kappa_ex[i = 1:graph.ndeg] in Parameter(i), container=Array)
    set_parameter_value.(kappa_ex, kappa_ex_init)

    # Defining Utility Funcion: from Cjn + parameters (by operator overloading)
    @expression(model, Cj, sum(Cjn .^ psigma, dims=2) .^ (1 / psigma))
    @expression(model, cj, ifelse.(Lj .== 0, 0.0, Cj ./ Lj))
    @expression(model, uj, ((cj / param.alpha) .^ param.alpha .* (param.hj / (1-param.alpha)) .^ (1-param.alpha)) .^ (1-param.rho) / (1-param.rho))
    @expression(model, U, sum(param.omegaj .* Lj .* uj))
    @objective(model, Max, U)

    # Balanced flow constraints
    @constraint(model, Pjn[j in 1:param.J, n in 1:param.N],
        Cjn[j, n] + sum(A[j, i] * Qin[i, n] for i in 1:graph.ndeg) -
        Yj[j] + sum(
            ifelse(Qin[i, n] > 0, Apos[j, i], Aneg[j, i]) *
            abs(Qin[i, n])^(1 + param.beta) / kappa_ex[i]
            for i in 1:graph.ndeg
        ) <= 0
    )
    return model
end

function recover_allocation_fixed_armington(model, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    model_dict = model.obj_dict
    results = Dict()
    # Production: Fixed because just one good is produced by each location
    Yj = dropdims(sum(param.Zjn, dims = 2) .* param.Lj .^ param.a, dims = 2)
    WY = param.Zjn .> 0

    results[:welfare] = value(model_dict[:U])
    results[:Yjn] = WY .* Yj
    results[:Yj] = Yj
    results[:Cjn] = value.(model_dict[:Cjn])
    results[:Cj] = dropdims(value.(model_dict[:Cj]), dims = 2)
    results[:Ljn] = WY .* param.Lj
    results[:Lj] = param.Lj
    results[:cj] = dropdims(value.(model_dict[:cj]), dims = 2)
    results[:hj] = ifelse.(results[:Lj] .== 0, 0.0, param.Hj ./ results[:Lj])
    results[:uj] = dropdims(value.(model_dict[:uj]), dims = 2)
    # Prices
    results[:Pjn] = shadow_price.(model_dict[:Pjn])
    results[:PCj] = dropdims(sum(results[:Pjn] .^ (1-param.sigma), dims=2), dims = 2) .^ (1/(1-param.sigma))    
    # Network flows
    results[:Qin] = value.(model_dict[:Qin])
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end