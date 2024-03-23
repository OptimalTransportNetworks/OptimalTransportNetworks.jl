# The primal case, no mobility, with cross-good congestion

function model_fixed_cgc(optimizer, auxdata)

    # Extract parameters
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    kappa_ex_init = auxdata[:kappa_ex]
    A = auxdata[:A]
    Apos = auxdata[:Apos]
    Aneg = auxdata[:Aneg]
    Lj = param.Lj
    m = param.m # 1:param.N: Vector of weights on each goods flow for aggregate congestion term
    psigma = (param.sigma - 1) / param.sigma
    beta_nu = (param.beta + 1) / param.nu

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variables
    @variable(model, Djn[1:graph.J, 1:param.N] >= 1e-8, container=Array, start = 1e-6)             # Consumption per good pre-transport cost (Dj)
    @variable(model, Qin_direct[1:graph.ndeg, 1:param.N] >= 1e-8, container=Array, start = 0.0)    # Direct aggregate flow
    @variable(model, Qin_indirect[1:graph.ndeg, 1:param.N] >= 1e-8, container=Array, start = 0.0)  # Indirect aggregate flow
    @variable(model, Ljn[1:graph.J, 1:param.N] >= 1e-8, container=Array, start = 1e-6)             # Good specific labour
    @variable(model, cj[1:graph.J] >= 1e-8, container=Array, start = 1e-6)                         # Overall consumption bundle, including transport costs

    # Parameters: to be updated between solves
    @variable(model, kappa_ex[i = 1:graph.ndeg] in Parameter(i), container=Array)
    set_parameter_value.(kappa_ex, kappa_ex_init)

    # Defining Utility Funcion: from cj + parameters (by operator overloading)
    @expression(model, uj, ((cj / param.alpha) .^ param.alpha .* (param.hj / (1-param.alpha)) .^ (1-param.alpha)) .^ (1-param.rho) / (1-param.rho))
    @expression(model, U, sum(param.omegaj .* param.Lj .* uj))      # Overall Utility
    @objective(model, Max, U)

    # Create the matrix B_direct (resp. B_indirect) of transport cost along the direction of the edge (resp. in edge opposite direction)
    B_direct = @expression(model, ((Qin_direct .^ param.nu) * m) .^ beta_nu ./ kappa_ex)
    B_indirect = @expression(model, ((Qin_indirect .^ param.nu) * m) .^ beta_nu ./ kappa_ex)
    # Final good constraints
    @expression(model, Dj, dropdims(sum(Djn .^ psigma, dims=2) .^ (1 / psigma), dims=2))
    @constraint(model, cj .* param.Lj + Apos * B_direct + Aneg * B_indirect - Dj .<= -1e-8)

    # Balanced flow constraints
    @expression(model, Yjn, param.Zjn .* (Ljn .^ param.a))
    @constraint(model, Pjn, Djn + A * Qin_direct - A * Qin_indirect - Yjn .<= -1e-8)

    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, sum(Ljn, dims=2) .- Lj .<= -1e-8)

    return model
end

function recover_allocation_fixed_cgc(model, auxdata)
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:U])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = dropdims(sum(results[:Yjn], dims=2), dims = 2) 
    results[:Ljn] = value.(model_dict[:Ljn])
    results[:Lj] = param.Lj
    results[:Djn] = value.(model_dict[:Djn]) # Consumption per good pre-transport cost
    results[:Dj] = value.(model_dict[:Dj])
    results[:cj] = value.(model_dict[:cj])
    results[:Cj] = results[:cj] .* param.Lj
    results[:hj] = ifelse.(results[:Lj] .== 0, 0.0, param.Hj ./ results[:Lj])
    results[:uj] = value.(model_dict[:uj])
    # Prices
    results[:Pjn] = shadow_price.(model_dict[:Pjn])
    results[:PCj] = dropdims(sum(results[:Pjn] .^ (1-param.sigma), dims=2), dims = 2) .^ (1/(1-param.sigma))    
    # Network flows
    Qin_direct = value.(model_dict[:Qin_direct])
    Qin_indirect = value.(model_dict[:Qin_indirect])
    results[:Qin] = max.(Qin_direct, Qin_indirect) - min.(Qin_direct, Qin_indirect)
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end
