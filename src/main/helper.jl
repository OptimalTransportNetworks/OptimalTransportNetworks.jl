
"""
    dict_to_namedtuple(dict)

Convert a dictionary to a NamedTuple.

If the input is already a NamedTuple, it is returned unchanged.
Otherwise, it creates a new NamedTuple from the dictionary's keys and values.

# Arguments
- `dict`: A dictionary or NamedTuple to be converted.

# Returns
- A NamedTuple equivalent to the input dictionary.
"""
function dict_to_namedtuple(dict)
    if dict isa NamedTuple
        return dict
    else
        return NamedTuple{Tuple(keys(dict))}(values(dict))
    end
end

"""
    namedtuple_to_dict(namedtuple)

Convert a NamedTuple to a dictionary.

If the input is already a dictionary, it is returned unchanged.
Otherwise, it creates a new dictionary from the NamedTuple's pairs.

# Arguments
- `namedtuple`: A NamedTuple or dictionary to be converted.

# Returns
- A dictionary equivalent to the input NamedTuple.
"""
function namedtuple_to_dict(namedtuple)
    if namedtuple isa Dict
        return namedtuple
    else
        # return Dict(Symbol(k) => v for (k, v) in pairs(namedtuple))
        return Dict(pairs(namedtuple)) # Can be of fixed type
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

# Description: auxiliary function that converts kappa_jk into kappa_i
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

"""
    represent_edges(graph)

Creates a NamedTuple providing detailed representation of the graph edges. 

# Arguments
- `graph`: NamedTuple that contains the underlying graph (created by `dict_to_namedtuple(create_graph())`)

# Returns
- A NamedTuple with the following fields:
  - `A`: J x ndeg matrix where each column represents an edge. The value is 1 if the edge starts at node J, -1 if it ends at node J, and 0 otherwise.
  - `Apos`: J x ndeg matrix where each column represents an edge. The value is the positive part of the edge flow.
  - `Aneg`: J x ndeg matrix where each column represents an edge. The value is the negative part of the edge flow.
  - `edge_start`: J x ndeg matrix where each column represents an edge. The value is the starting node of the edge.
  - `edge_end`: J x ndeg matrix where each column represents an edge. The value is the ending node of the edge.
"""
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
    create_auxdata(param, graph, edges, I)

Creates the auxdata structure that contains all the auxiliary parameters for estimation

# Arguments
- `param`: NamedTuple that contains the model's parameters (created by `dict_to_namedtuple(init_parameters())`)
- `graph`: NamedTuple that contains the underlying graph (created by `dict_to_namedtuple(create_graph())`)
- `edges`: NamedTuple that contains the edges of the graph (created by `represent_edges()`)
- `I`: J x J symmetric matrix of current infrastructure (investments)

# Returns
- A NamedTuple with the following fields:
  - `param`: The input parameter NamedTuple.
  - `graph`: The input graph NamedTuple.
  - `edges`: The edges of the graph.
  - `kappa`: The kappa matrix: I^gamma / delta_tau
  - `kappa_ex`: The extracted kappa values (ndeg x 1)
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
    get_model(auxdata)

Construct the appropriate JuMP model based on the auxiliary data.

# Arguments
- `auxdata`: Auxiliary data required for constructing the model (created by `create_auxdata()`).

# Returns
- `model`: The constructed JuMP model.
- `recover_allocation`: A function to recover the allocation from the model solution.
"""
function get_model(auxdata)
     param = auxdata.param
     graph = auxdata.graph
     optimizer = get(param, :optimizer, Ipopt.Optimizer)

    if haskey(param, :model)
        model = param.model(optimizer, auxdata)
        if !haskey(param, :recover_allocation)
            error("The custom model does not have the recover_allocation function.")
        end
        recover_allocation = param.recover_allocation 
    elseif param.mobility == 1 && param.cong
        if all(sum(graph.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_mobility_cgc_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility_cgc_armington
        else
            model = model_mobility_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility_cgc
        end
    elseif param.mobility == 0.5 && param.cong
        if all(sum(graph.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_partial_mobility_cgc_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility_cgc_armington
        else
            model = model_partial_mobility_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility_cgc
        end
    elseif param.mobility == 0 && param.cong
        if param.beta <= 1 && param.duality # && param.a < 1
            model = nothing # model_fixed_duality_cgc(optimizer, auxdata)
            recover_allocation = solve_allocation_by_duality_cgc #recover_allocation_fixed_duality_cgc
        elseif all(sum(graph.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_fixed_cgc_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_cgc_armington
        else
            model = model_fixed_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_cgc
        end
    elseif param.mobility == 1 && !param.cong
        if all(sum(graph.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_mobility_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility_armington
        else
            model = model_mobility(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility
        end
    elseif param.mobility == 0.5 && !param.cong
        if all(sum(graph.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_partial_mobility_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility_armington    
        else
            model = model_partial_mobility(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility    
        end
    elseif param.mobility == 0 && !param.cong
        if param.beta <= 1 && param.duality # && param.a < 1 
            model = nothing # model_fixed_duality(optimizer, auxdata)
            recover_allocation = solve_allocation_by_duality # recover_allocation_fixed_duality
        elseif all(sum(graph.Zjn .> 0, dims = 2) .<= 1) # Armington case
            model = model_fixed_armington(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_armington
        else
            model = model_fixed(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed
        end
    else
        error("Usupported model configuration with labor_mobility = $(param.mobility) and cross_good_congestion = $(param.cong)")
    end

    # --------------
    # CUSTOMIZATIONS
    if model !== nothing
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
    end
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

# KNN Version: More robust
function linear_interpolation_2d(vec_x, vec_y, vec_map, xmap, ymap)

    # Ensure input vectors are of the same length
    @assert length(vec_x) == length(vec_y) == length(vec_map) "Input vectors must have the same length"
    
    # Ensure input vectors are Float64
    vec_x = convert(Vector{Float64}, vec_x)
    vec_y = convert(Vector{Float64}, vec_y)
    vec_map = convert(Vector{Float64}, vec_map)
    xmap = convert(Vector{Float64}, xmap)
    ymap = convert(Vector{Float64}, ymap)

    # Initialize the output array
    fmap = zeros(length(xmap), length(ymap))

    # Create a KDTree for efficient nearest neighbor search
    points = hcat(vec_x, vec_y)
    tree = KDTree(points'; leafsize = 5)

    # Determine the number of neighbors to use (k)
    k = min(15, size(points, 1))  # Use 15 or the total number of points, whichever is smaller

    for (ix, x) in enumerate(xmap), (iy, y) in enumerate(ymap)

        # Find the 15 nearest neighbors
        idxs, dists = knn(tree, [x, y], k, true)

        # If the point is exactly on a known point, use that value
        if dists[1] ≈ 0
            fmap[ix, iy] = vec_map[idxs[1]]
            continue
        end

        # Weights
        weights = 1 ./ dists.^2
        weights ./= sum(weights)

        # Interpolate
        fmap[ix, iy] = sum(weights .* vec_map[idxs])
    end

    return fmap
end