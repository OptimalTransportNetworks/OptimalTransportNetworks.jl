# The primal case, with labor mobility, no cross-good congestion, and an Armington (1969) world where each location produces only one good

function model_mobility_armington(optimizer, auxdata)
    
    # Parameters and data
    param = auxdata.param
    graph = auxdata.graph
    kappa_ex_init = auxdata.kappa_ex
    A = auxdata.edges.A
    Apos = auxdata.edges.Apos
    Aneg = auxdata.edges.Aneg
    psigma = (param.sigma - 1) / param.sigma
    Hj = graph.Hj

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variables + bounds
    @variable(model, u, start = 0.0)                                                      # Overall utility
    @variable(model, Cjn[1:graph.J, 1:param.N] >= 1e-8, container=Array, start = 1e-6)    # Good specific consumption
    @variable(model, Qin[1:graph.ndeg, 1:param.N], container=Array, start = 0.0)          # Good specific flow
    @variable(model, 1e-8 <= Lj[1:graph.J] <= 1, container=Array, start = 1 / graph.J)    # Total labour

    # Parameters: to be updated between solves
    @variable(model, kappa_ex[i = 1:graph.ndeg] in Parameter(i), container=Array)
    set_parameter_value.(kappa_ex, kappa_ex_init)

    # Objective
    @objective(model, Max, u)

    # Utility constraint (Lj*u <= ... )
    for j in 1:graph.J
        Cj = sum(Cjn[j, n]^psigma for n in 1:param.N)^(1 / psigma)
        @constraint(model, Lj[j] * u - (Cj / param.alpha)^param.alpha * (Hj[j] / (1 - param.alpha))^(1 - param.alpha) <= -1e-8)
    end

    # balanced flow constraints
     @expression(model, Yjn[j=1:graph.J, n=1:param.N], graph.Zjn[j, n] * Lj[j]^param.a)
    @constraint(model, Pjn[j in 1:param.J, n in 1:param.N],
        Cjn[j, n] + sum(A[j, i] * Qin[i, n] for i in 1:graph.ndeg) -
        Yjn[j, n] + sum(
            ifelse(Qin[i, n] > 0, Apos[j, i], Aneg[j, i]) *
            abs(Qin[i, n])^(1 + param.beta) / kappa_ex[i]
            for i in 1:graph.ndeg
        ) <= -1e-8
    )

    # Labor resource constraint
    @constraint(model, -1e-8 <= sum(Lj) - 1 <= 1e-8)

    return model
end

function recover_allocation_mobility_armington(model, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:u])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = dropdims(sum(results[:Yjn], dims=2), dims = 2) 
    results[:Cjn] = value.(model_dict[:Cjn])
    results[:Cj] = dropdims(sum(results[:Cjn] .^ ((param.sigma-1)/param.sigma), dims=2), dims = 2) .^ (param.sigma/(param.sigma-1))
    results[:Lj] = value.(model_dict[:Lj])
    results[:Ljn] = (graph.Zjn .> 0) .* results[:Lj]
    results[:cj] = ifelse.(results[:Lj] .== 0, 0.0, results[:Cj] ./ results[:Lj])
    results[:hj] = ifelse.(results[:Lj] .== 0, 0.0, graph.Hj ./ results[:Lj])
    results[:uj] = param.u.(results[:cj], results[:hj])
    # Prices
    results[:Pjn] = shadow_price.(model_dict[:Pjn])
    results[:PCj] = dropdims(sum(results[:Pjn] .^ (1-param.sigma), dims=2), dims=2) .^ (1/(1-param.sigma))    
    # Network flows
    results[:Qin] = value.(model_dict[:Qin])
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end
