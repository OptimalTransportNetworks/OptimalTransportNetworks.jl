# The primal case, no mobility, no cross-good congestion

function model_fixed(optimizer, auxdata)

    # Extract parameters
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    kappa_ex = auxdata[:kappa_ex]
    A = auxdata[:A]
    psigma = (param.sigma - 1) / param.sigma
    Lj = param.Lj

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variables
    @variable(model, U)
    @variable(model, Cjn[1:graph.J, 1:param.N])
    @variable(model, Qin[1:graph.ndeg, 1:param.N])
    @variable(model, Ljn[1:graph.J, 1:param.N])

    # Defining Utility Funcion: from Cjn + parameters (by operator overloading)
    Cj = @expression(model, sum(Cjn .^ psigma, dims=2) .^ (1 / psigma))
    cj = @expression(model, ifelse.(param.Lj .== 0, 0.0, Cj ./ param.Lj))
    uj = @expression(model, ((cj / param.alpha) .^ param.alpha .* (param.hj / (1-param.alpha)) .^ (1-param.alpha)) .^ (1-param.rho) / (1-param.rho))
    U = @expression(model, sum(param.omegaj .* param.Lj .* uj))
    @objective(model, Max, U)

    # Define Yjn (production) as expression
    @expression(model, Yjn[j=1:graph.J, n=1:param.N], param.Zjn[j, n] * Ljn[j, n]^param.a)
    # Balanced flow constraints
    @constraint(model, Pjn[j in 1:param.J, n in 1:param.N],
        Cjn[j, n] + sum(A[j, i] * Qin[i, n] for i in 1:graph.ndeg) -
        Yjn[j, n] + sum(
            max(ifelse(Qin[i, n] > 0, A[j, i], -A[j, i]), 0) *
            abs(Qin[i, n])^(1 + param.beta) / kappa_ex[i]
            for i in 1:graph.ndeg
        ) <= -1e-8
    )

    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, -1e-8 .<= sum(Ljn, dims=2) .- Lj .<= 1e-8)

    return model
end

function recover_allocation_fixed(model, auxdata)
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:U])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = sum(results[:Yjn], dims=2) 
    results[:Cjn] = value.(model_dict[:Cjn])
    results[:Cj] = value.(model_dict[:Cj])
    results[:Ljn] = value.(model_dict[:Ljn])
    results[:Lj] = param.Lj
    results[:cj] = value.(model_dict[:cj])
    results[:hj] = ifelse.(results[:Lj] .== 0, 0.0, param.Hj ./ results[:Lj])
    results[:uj] = value.(model_dict[:uj])
    # Prices
    results[:Pjn] = shadow_price.(model_dict[:Pjn])
    results[:PCj] = sum(results[:Pjn] .^ (1-param.sigma), dims=2) .^ (1/(1-param.sigma))    
    # Network flows
    results[:Qin] = value.(model_dict[:Qin])
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end