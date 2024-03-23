# The primal case, with mobility within regions and cross-good congestion

function model_partial_mobility_cgc(optimizer, auxdata)

    # Extract parameters
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    region = graph.region
    if length(region) != graph.J
        error("length(region) = $(length(region)) does not match number of nodes = $(graph.J)")
    end
    kappa_ex_init = auxdata[:kappa_ex]
    A = auxdata[:A]
    Apos = auxdata[:Apos]
    Aneg = auxdata[:Aneg]
    m = param.m # Vector of weights on each goods flow for aggregate congestion term
    psigma = (param.sigma - 1) / param.sigma
    beta_nu = (param.beta + 1) / param.nu
    Lr = param.Lr
    if length(param.omegar) != param.nregions
        error("length(param.omegar) = $(length(param.omegar)) does not match number of regions = $(param.nregions)")
    end
    if length(Lr) != param.nregions
        error("Populations Lr = $(length(Lr)) does not match number of regions = $(param.nregions)")
    end

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variable declarations
    @variable(model, ur[1:param.nregions], container=Array, start = 0.0)                          # Utility per capita in each region
    @variable(model, Djn[1:graph.J, 1:param.N] >= 1e-8, container=Array, start = 1e-6)            # Consumption per good pre-transport cost (Dj)
    @variable(model, cj[1:graph.J] >= 1e-8, container=Array, start = 1e-6)                        # Overall consumption bundle, including transport costs
    @variable(model, Qin_direct[1:graph.ndeg, 1:param.N] >= 1e-8, container=Array, start = 0.0)   # Direct aggregate flow
    @variable(model, Qin_indirect[1:graph.ndeg, 1:param.N] >= 1e-8, container=Array, start = 0.0) # Indirect aggregate flow
    # NOTE: Fajgelbaum et al (2019) only optimize Lj and distribute it equally for goods with positive productivity
    @variable(model, Ljn[1:graph.J, 1:param.N] >= 1e-8, container=Array)            # Good specific labour
    @variable(model, Lj[1:graph.J] >= 1e-8, container=Array)                        # Overall labour
    # Calculate start values for Lj and Ljn
    pop_start = (Lr ./ gsum(ones(param.nregions), param.nregions, region))[region]
    set_start_value.(Lj, pop_start)
    pop_start_goods = repeat(pop_start / param.N, 1, param.N)
    set_start_value.(Ljn, pop_start_goods)

    # Parameters: to be updated between solves
    @variable(model, kappa_ex[i = 1:graph.ndeg] in Parameter(i), container=Array)
    set_parameter_value.(kappa_ex, kappa_ex_init)

    # Objective
    @expression(model, U, sum(param.omegar .* Lr .* ur)) # Overall Utility
    @objective(model, Max, U)

    # Utility constraint (Lj * ur <= ... )
    @constraint(model, Lj .* ur[region] - (cj .* Lj ./ param.alpha) .^ param.alpha .* (param.Hj ./ (1 - param.alpha)) .^ (1 - param.alpha) .<= -1e-8)

    # Create the matrix B_direct (resp. B_indirect) of transport cost along the direction of the edge (resp. in edge opposite direction)
    B_direct = @expression(model, ((Qin_direct .^ param.nu) * m) .^ beta_nu ./ kappa_ex)
    B_indirect = @expression(model, ((Qin_indirect .^ param.nu) * m) .^ beta_nu ./ kappa_ex)
    # Final good constraints
    @expression(model, Dj, sum(Djn .^ psigma, dims=2) .^ (1 / psigma))
    @constraint(model, cj .* Lj + Apos * B_direct + Aneg * B_indirect - Dj .<= -1e-8)

    # Balanced flow constraints: same as full mobility
    @expression(model, Yjn, param.Zjn .* (Ljn .^ param.a))
    @constraint(model, Pjn, Djn + A * Qin_direct - A * Qin_indirect - Yjn .<= -1e-8)

    # Labor resource constraints (within each region)
    @constraint(model, -1e-8 .<= gsum(Lj, param.nregions, region) - Lr .<= 1e-8)

    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, -1e-8 .<= sum(Ljn, dims=2) .- Lj .<= 1e-8)
    
    return model
end

function recover_allocation_partial_mobility_cgc(model, auxdata)
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    model_dict = model.obj_dict
    results = Dict()

    results[:welfare] = value(model_dict[:U])
    results[:reg_pc_welfare] = value.(model_dict[:ur])
    results[:Yjn] = value.(model_dict[:Yjn])
    results[:Yj] = dropdims(sum(results[:Yjn], dims=2), dims = 2) 
    results[:Ljn] = value.(model_dict[:Ljn])
    results[:Lj] = value.(model_dict[:Lj])
    results[:Djn] = value.(model_dict[:Djn]) # Consumption per good pre-transport cost
    results[:Dj] = dropdims(value.(model_dict[:Dj]), dims=2)
    results[:cj] = value.(model_dict[:cj])
    results[:Cj] = results[:cj] .* results[:Lj]
    results[:hj] = ifelse.(results[:Lj] .== 0, 0.0, param.Hj ./ results[:Lj])
    results[:uj] = param.u.(results[:cj], results[:hj])
    # Prices
    results[:Pjn] = shadow_price.(model_dict[:Pjn])
    results[:PCj] = dropdims(sum(results[:Pjn] .^ (1-param.sigma), dims=2), dims=2) .^ (1/(1-param.sigma))    
    # Network flows
    Qin_direct = value.(model_dict[:Qin_direct])
    Qin_indirect = value.(model_dict[:Qin_indirect])
    results[:Qin] = ifelse.(Qin_direct .> Qin_indirect, Qin_direct, -Qin_indirect)
    results[:Qjkn] = gen_network_flows(results[:Qin], graph, param.N)
    return results
end