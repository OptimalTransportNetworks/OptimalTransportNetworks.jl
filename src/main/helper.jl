
function dict_to_namedtuple(dict)
    if dict isa NamedTuple
        return dict
    else
        return NamedTuple{Tuple(keys(dict))}(values(dict))
    end
end

function namedtuple_to_dict(namedtuple)
    if namedtuple isa Dict
        return namedtuple
    else
        return Dict(pairs(namedtuple))
    end
end

function gen_network_flows(Qin, graph, N)
    Qjkn = zeros(graph.J, graph.J, N)
    nodes = graph.nodes
    id = 1
    for i in 1:graph.J
        for j in 1:length(nodes[i])
            if nodes[i][j] > i
                Qjkn[i, nodes[i][j], :] = max.(Qin[id, :], 0)
                Qjkn[nodes[i][j], i, :] = max.(-Qin[id, :], 0)
                id += 1
            end
        end
    end
    return Qjkn
end

"""
    kappa_extract(graph, kappa)

Description: auxiliary function that converts kappa_jk into kappa_i
"""
function kappa_extract(graph, kappa)
    kappa_ex = zeros(graph.ndeg)
    nodes = graph.nodes
    id = 1
    for i in 1:graph.J
        for j in nodes[i]
            if j > i
                kappa_ex[id] = kappa[i, j]
                id += 1
            end
        end
    end
    return kappa_ex
end

# Function to check if the input is a color
function is_color(input)
    # TODO: what if directly passing RGBA color?
    if isa(input, String)
        return haskey(Colors.color_names, input)
    elseif isa(input, Symbol)
        return haskey(Colors.color_names, string(input))
    else
        return false
    end
end

function gsum(x, ng, g)
    n = length(x)
    if length(g) != n
        error("length(region) = $(length(g)) must match number of edges (graph.J) = $n")
    end
    # Determine the type for the result based on the type of elements in x
    if eltype(x) <: JuMP.AbstractVariableRef
        # Initialize a vector of affine expressions
        res = Vector{JuMP.GenericAffExpr{Float64, JuMP.VariableRef}}(undef, ng)
        res .= zero(JuMP.GenericAffExpr{Float64, JuMP.VariableRef})
    else
        res = zeros(ng)
    end
    for i in 1:n
        @inbounds res[g[i]] += x[i]
    end
    return res
end

function rowmult(A, v)
    r = similar(A)
    @inbounds for j = 1:size(A,2) 
        @simd for i = 1:size(A,1) 
            r[i,j] = v[j] * A[i,j] 
        end
    end 
    return r
end 

function represent_edges(graph)
    # Create matrix A
    # Note: matrix A is of dimension J*ndeg and takes value 1 when node J is
    # the starting point of edge ndeg, and -1 when it is the ending point. It 
    # is used to make hessian and jacobian calculations using matrix algebra
    A = zeros(Int, graph.J, graph.ndeg)
    edge_start = zeros(Int, graph.ndeg)    
    edge_end = zeros(Int, graph.ndeg) 
    nodes = graph.nodes

    id = 1
    for j in 1:graph.J
        for k in 1:length(nodes[j])
            if nodes[j][k] > j # This enforces that the graph is undirected
                A[j, id] = 1
                edge_start[id] = j
                A[nodes[j][k], id] = -1
                edge_end[id] = nodes[j][k]
                id += 1
            end
        end
    end

    Apos = max.(A, 0)
    Aneg = max.(-A, 0)

    return (A = A, Apos = Apos, Aneg = Aneg, edge_start = edge_start, edge_end = edge_end)
end

"""
    create_auxdata(param, graph, I)

Creates the auxdata structure that contains all the auxiliary parameters for estimation

# Arguments
- `param`: structure that contains the model's parameters
- `graph`: structure that contains the underlying graph (created by create_graph function)
- `I`: provides the current JxJ symmetric matrix of infrastructure investment

# Output
- `auxdata`: structure auxdata to be used by IPOPT bundle.
"""
function create_auxdata(param, graph, edges, I)
    # Make named tuples
    # param = dict_to_namedtuple(param)
    # graph = dict_to_namedtuple(graph)

    # Initialize kappa
    kappa = max.(I.^param.gamma ./ graph.delta_tau, param.kappa_min)
    kappa[.!graph.adjacency] .= 0
    kappa_ex = kappa_extract(graph, kappa)  # extract the ndeg free values of matrix kappa (due to symmetry)

    # Store in auxdata
    auxdata = (
        param = param,
        graph = graph,
        edges = edges,
        kappa = kappa,
        kappa_ex = kappa_ex,
        # Iex = kappa_extract(graph, I),
        # delta_tau_ex = kappa_extract(graph, graph.delta_tau),
    )
    return auxdata
end

"""
    get_model(param, auxdata)

Construct the appropriate model based on the parameters and auxiliary data.

# Arguments
- `param`: A named tuple containing the model parameters.
- `auxdata`: Auxiliary data required for constructing the model.

# Returns
- `model`: The constructed model.
- `recover_allocation`: A function to recover the allocation from the model solution.
"""
function get_model(param, auxdata)
     optimizer = get(param, :optimizer, Ipopt.Optimizer)
     param = dict_to_namedtuple(param)

    if haskey(param, :model)
        model = param.model(optimizer, auxdata)
        if !haskey(param, :recover_allocation)
            error("The custom model does not have the recover_allocation function.")
        end
        recover_allocation = param.recover_allocation 
    elseif param.mobility == 1 && param.cong
        if all(sum(param.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_mobility_cgc_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility_cgc_armington
        else
            model = model_mobility_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility_cgc
        end
    elseif param.mobility == 0.5 && param.cong
        if all(sum(param.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_partial_mobility_cgc_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility_cgc_armington
        else
            model = model_partial_mobility_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility_cgc
        end
    elseif param.mobility == 0 && param.cong
        if all(sum(param.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_fixed_cgc_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_cgc_armington
        else
            model = model_fixed_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_cgc
        end
    elseif param.mobility == 1 && !param.cong
        if all(sum(param.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_mobility_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility_armington
        else
            model = model_mobility(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility
        end
    elseif param.mobility == 0.5 && !param.cong
        if all(sum(param.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_partial_mobility_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility_armington    
        else
            model = model_partial_mobility(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility    
        end
    elseif param.mobility == 0 && !param.cong
        if all(sum(param.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_fixed_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_armington
        elseif param.beta <= 1 && param.a < 1 && param.duality
            model = model_fixed_duality(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_duality
        else
            model = model_fixed(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed
        end
    else
        error("Usupported model configuration with labor_mobility = $(param.mobility) and cross_good_congestion = $(param.cong)")
    end

    # --------------
    # CUSTOMIZATIONS

    if haskey(param, :optimizer_attr)
        for (key, value) in param.optimizer_attr
            set_optimizer_attribute(model, String(key), value)
        end
    end

    if haskey(param, :model_attr) 
        for value in values(param.model_attr)
            if !(value isa Tuple)  
                error("model_attr must be a dict of tuples.")
            end
            set_optimizer_attribute(model, value[1], value[2])
        end
    end
    # E.g.:
    # set_attribute(model,
    #    MOI.AutomaticDifferentiationBackend(),
    #    MathOptSymbolicAD.DefaultBackend(),
    # )
    return model, recover_allocation
end

function all_variables_except_kappa_ex(model)
    all_vars = all_variables(model)
    return setdiff(all_vars, model[:kappa_ex])
end

function rescale_network!(param, graph, I1, Il, Iu; max_iter = 100)
    distance_lb = max(maximum(Il - I1), 0.0)
    distance_ub = max(maximum(I1 - Iu), 0.0)
    counter_rescale = 0
    TOL_I_BOUNDS = min(param.tol, 1e-7)
    
    while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < max_iter
        I1 = max.(min.(I1, Iu), Il)
        I1 *= param.K / sum(graph.delta_i .* I1)
        distance_lb = max(maximum(Il - I1), 0.0)
        distance_ub = max(maximum(I1 - Iu), 0.0)
        counter_rescale += 1
    end
    
    if counter_rescale == 100 && distance_lb + distance_ub > param.tol && param.verbose
        println("Warning! Could not impose bounds on network properly.")
    end

    return I1
end
