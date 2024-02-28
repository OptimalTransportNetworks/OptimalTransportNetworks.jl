# The primal case, no mobility, with cross-good congestion

function model_fixed_cgc(optimizer, auxdata)

    # Extract parameters
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    kappa_ex = auxdata[:kappa_ex]
    A = auxdata[:A]
    Lj = param.Lj
    m = param.m # Vector of weights on each goods flow for aggregate congestion term
    psigma = (param.sigma - 1) / param.sigma
    beta_nu = (param.beta + 1) / param.nu

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variables
    @variable(model, U)                                    # Overall utility
    @variable(model, Djn[1:graph.J, 1:param.N])            # Consumption per good pre-transport cost (Dj)
    @variable(model, Qin_direct[1:graph.ndeg, 1:param.N])  # Direct aggregate flow
    @variable(model, Qin_indirect[1:graph.ndeg, 1:param.N])# Indirect aggregate flow
    @variable(model, Ljn[1:graph.J, 1:param.N])            # Good specific labour
    @variable(model, cj[1:graph.J])                        # Overall consumption bundle, including transport costs

    # Defining Utility Funcion: from cj + parameters (by operator overloading)
    uj = @expression(model, ((cj / param.alpha) .^ param.alpha .* (param.hj / (1-param.alpha)) .^ (1-param.alpha)) .^ (1-param.rho) / (1-param.rho))
    U = @expression(model, sum(param.omegaj .* param.Lj .* uj))
    @objective(model, Max, U)

    # Final good constraints
    for i in 1:graph.ndeg
        # Create the matrix B_direct (resp. B_indirect) of transport cost along the direction of the edge (resp. in edge opposite direction)
        B_direct = sum(m[n] * Qin_direct[i, n] ^ param.nu for n in 1:param.N) ^ beta_nu / kappa_ex[i]
        B_indirect = sum(m[n] * Qin_indirect[i, n] ^ param.nu for n in 1:param.N) ^ beta_nu / kappa_ex[i]
        @constraint(model, [j in 1:param.J],
                    cj[j] * param.Lj[j] + 
                    max(A[j, i], 0) * B_direct + 
                    max(-A[j, i], 0) * B_indirect - 
                    sum(Djn[j, n] ^ psigma for n in 1:param.N) ^ (1 / psigma) <= 1e-8
        )
    end

    # Balanced flow constraints
    Yjn = @expression(model, param.Zjn .* (Ljn .^ param.a))
    @constraint(model, Pjn, Djn + A * Qin_direct - A * Qin_indirect - Yjn .<= -1e-8)

    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, -1e-8 .<= sum(Ljn, dims=2) .- Lj .<= 1e-8)

    return model
end

function recover_allocation_fixed_cgc(model, auxdata)
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:U])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = sum(results[:Yjn], dims=2) 
    results[:Ljn] = value.(model_dict[:Ljn])
    results[:Lj] = param.Lj
    results[:Djn] = value.(model_dict[:Djn]) # Consumption per good pre-transport cost
    results[:Dj] = sum(results[:Djn] .^ ((param.sigma-1)/param.sigma), dims=2) .^ (param.sigma/(param.sigma-1))
    results[:cj] = value.(model_dict[:cj])
    results[:Cj] = results[:cj] .* param.Lj
    results[:hj] = ifelse.(results[:Lj] .== 0, 0.0, param.Hj ./ results[:Lj])
    results[:uj] = value.(model_dict[:uj])
    # Prices
    results[:Pjn] = shadow_price.(model_dict[:Pjn])
    results[:PCj] = sum(results[:Pjn] .^ (1-param.sigma), dims=2) .^ (1/(1-param.sigma))    
    # Network flows
    Qin_direct = value.(model_dict[:Qin_direct])
    Qin_indirect = value.(model_dict[:Qin_indirect])
    results[:Qin] = max.(Qin_direct, Qin_indirect) - min.(Qin_direct, Qin_indirect)
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end
