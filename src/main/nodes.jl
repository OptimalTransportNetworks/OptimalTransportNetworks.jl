
"""
    add_node(param, graph, x, y, neighbors) -> Dict, NamedTuple

Add a node in position (x,y) and list of neighbors. The new node is given an index J+1.

# Arguments
- `param::Dict`: Dict that contains the model's parameters, or `nothing` to only update graph
- `graph::NamedTuple`: Named tuple that contains the underlying graph (created by create_graph())
- `x::Float64`: x coordinate of the new node (any real number)
- `y::Float64`: y coordinate of the new node (any real number)
- `neighbors::Vector{Int64}`: Vector of nodes to which it is connected (1 x n list of node indices between 1 and J, where n is an arbitrary # of neighbors) 

The cost matrices `delta_tau` and `delta_i` are parametrized as a function of Euclidean distance between nodes.

Returns the updated `graph` and `param` objects (`param` is affected too because the variable `Zjn`, `Lj`, `Hj` and others are reset to a uniform dist.)
"""
function add_node(param, graph, x, y, neighbors)

    # Check validity of neighbors list
    if any(neighbors .!= floor.(neighbors)) || any(neighbors .< 1) || any(neighbors .> graph.J)
        error("neighbors should be a list of integers between 1 and $(graph.J).")
    end
    Jnew = graph.J + 1

    # Add node
    nodes = [graph.nodes; [neighbors]]
    new_x = [graph.x; x]
    new_y = [graph.y; y]

    # Add new node to neighbors' neighbors
    # And update adjacency and cost matrices
    adjacency = zeros(Jnew, Jnew)
    adjacency[1:graph.J, 1:graph.J] = graph.adjacency

    delta_tau = zeros(Jnew, Jnew)
    delta_tau[1:graph.J, 1:graph.J] = graph.delta_tau

    delta_i = zeros(Jnew, Jnew)
    delta_i[1:graph.J, 1:graph.J] = graph.delta_i

    for i in neighbors
        nodes[i] = [nodes[i]; Jnew]
        distance = sqrt((new_x[i] - x)^2 + (new_y[i] - y)^2)

        # Adjacency
        adjacency[i, Jnew] = 1
        adjacency[Jnew, i] = 1

        # Travel cost: delta_tau
        delta_tau[i, Jnew] = distance
        delta_tau[Jnew, i] = distance

        # Building cost: delta_i
        delta_i[i, Jnew] = distance
        delta_i[Jnew, i] = distance
    end

    # Update number of degrees of liberty for Ijk
    ndeg = sum(tril(adjacency)) # nb of degrees of liberty in adjacency matrix

    # Return new graph structure
    graph_new = (
        J = Jnew,
        x = new_x,
        y = new_y,
        nodes = nodes,
        adjacency = adjacency,
        delta_i = delta_i,
        delta_tau = delta_tau,
        ndeg = ndeg
    )

    if param !== nothing
        # Now, update the param structure
        param_new = copy(param)
        if haskey(param, :J)
            param_new[:J] = Jnew
        end
        if haskey(param, :Lj)
            param_new[:Lj] = ones(Jnew) / Jnew
        end
        if haskey(param, :Hj)
            param_new[:Hj] = ones(Jnew)
        end
        if haskey(param, :hj)
            param_new[:hj] = param[:Hj] ./ param[:Lj]
        end
        if haskey(param, :omegaj)
            param_new[:omegaj] = ones(Jnew)
        end
        if haskey(param, :Zjn)
            param_new[:Zjn] = ones(Jnew, param[:N])
        end

        return param_new, graph_new
    end

    return graph_new
end


# Please note that in Julia, we use `Dict` to represent structures as in Matlab. The keys of the `Dict` are strings that correspond to the field names in the Matlab structure. Also, the `tril` function is not built-in in Julia, you may need to use a package like `LinearAlgebra` to use it.

"""
    find_node(graph, x, y) -> Int64

Returns the index of the node closest to the coordinates (x,y) on the graph.

# Arguments
- `graph::NamedTuple`: structure that contains the underlying graph
- `x::Float64`: x coordinate on the graph (between 1 and w)
- `y::Float64`: y coordinate on the graph (between 1 and h)
"""
function find_node(graph, x, y)
    distance = (graph.x .- x).^2 + (graph.y .- y).^2
    _, id = findmin(distance)
    return id
end



"""
    remove_node(param, graph, i) -> updated_param, updated_graph

Removes node i from the graph.

# Arguments
- `param::Dict`: structure that contains the model's parameters, or `nothing` to only update graph
- `graph::NamedTuple`: structure that contains the underlying graph (created by create_graph())
- `i::Int64`: index of the mode to be removed (integer between 1 and graph.J)

Returns the updated graph and param objects (param is affected too because the variable Zjn, Lj, Hj and others are changed).
"""
function remove_node(param, graph, i)

    if i < 1 || i > graph.J || i != floor(i)
        error("remove_node: node i should be an integer between 1 and $(graph.J).")
    end

    Jnew = graph.J - 1
    graph_new = namedtuple_to_dict(graph)

    graph_new[:J] = Jnew
    graph_new[:x] = deleteat!(copy(graph.x), i)
    graph_new[:y] = deleteat!(copy(graph.y), i)
    graph_new[:nodes] = deleteat!(copy(graph.nodes), i)

    nodes = graph_new[:nodes]
    for k in 1:Jnew
        node_k = filter(x -> x != i, nodes[k]) # nodes[k][nodes[k] .!= i]
        nodes[k] = ifelse.(node_k .> i, node_k .- 1, node_k) # reindex nodes k > i to k-1
    end

    # Rebuild adjacency matrix, delta_i and delta_tau
    graph_new[:adjacency] = [graph.adjacency[1:i-1, 1:i-1] graph.adjacency[1:i-1, i+1:end];
                             graph.adjacency[i+1:end, 1:i-1] graph.adjacency[i+1:end, i+1:end]]

    graph_new[:delta_i] = [graph.delta_i[1:i-1, 1:i-1] graph.delta_i[1:i-1, i+1:end];
                           graph.delta_i[i+1:end, 1:i-1] graph.delta_i[i+1:end, i+1:end]]

    graph_new[:delta_tau] = [graph.delta_tau[1:i-1, 1:i-1] graph.delta_tau[1:i-1, i+1:end];
                             graph.delta_tau[i+1:end, 1:i-1] graph.delta_tau[i+1:end, i+1:end]]

    if haskey(graph, :across_obstacle)
        graph_new[:across_obstacle] = [graph.across_obstacle[1:i-1, 1:i-1] graph.across_obstacle[1:i-1, i+1:end];
                                       graph.across_obstacle[i+1:end, 1:i-1] graph.across_obstacle[i+1:end, i+1:end]]
    end

    if haskey(graph, :along_obstacle)
        graph_new[:along_obstacle] = [graph.along_obstacle[1:i-1, 1:i-1] graph.along_obstacle[1:i-1, i+1:end];
                                      graph.along_obstacle[i+1:end, 1:i-1] graph.along_obstacle[i+1:end, i+1:end]]
    end

    graph_new[:ndeg] = sum(tril(graph_new[:adjacency]))

    if graph.region !== nothing
        graph_new[:region] = deleteat!(copy(graph.region), i)
    end

    if param !== nothing 
        # Now, update the param structure
        param_new = copy(param)
        param_new[:J] = Jnew
        if haskey(param, :Lj)
            param_new[:Lj] = deleteat!(copy(param[:Lj]), i)
        end
        if haskey(param, :Hj)
            param_new[:Hj] = deleteat!(copy(param[:Hj]), i)
        end
        if haskey(param, :hj)
            param_new[:hj] = deleteat!(copy(param[:hj]), i)
        end
        if haskey(param, :omegaj)
            param_new[:omegaj] = deleteat!(copy(param[:omegaj]), i)
        end
        if haskey(param, :Zjn)
            param_new[:Zjn] = vcat(param[:Zjn][1:i-1, :], param[:Zjn][i+1:end, :])
        end

        return param_new, dict_to_namedtuple(graph_new)
    end

    return dict_to_namedtuple(graph_new)
end

