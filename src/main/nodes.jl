
"""
    add_node(graph, x, y, neighbors) -> Dict

Add a node in position (x,y) and list of neighbors. The new node is given an index J+1. Returns an updated `graph` object.

# Arguments
- `graph::Dict`: Dict that contains the underlying graph (created by create_graph())
- `x::Float64`: x coordinate of the new node (any real number)
- `y::Float64`: y coordinate of the new node (any real number)
- `neighbors::Vector{Int64}`: Vector of nodes to which it is connected (1 x n list of node indices between 1 and J, where n is an arbitrary # of neighbors) 

# Notes
The cost matrices `delta_tau` and `delta_i` are parametrized as a function of Euclidean distance between nodes. 
The new node is given population 1e-6 and productivity equal to the minimum productivity in the graph.
"""
function add_node(graph, x, y, neighbors)

    if graph isa Dict
        graph_new = copy(dict)
    else
        graph_new = Dict(pairs(namedtuple))
    end

    # Check validity of neighbors list
    if any(neighbors .!= floor.(neighbors)) || any(neighbors .< 1) || any(neighbors .> graph[:J])
        error("neighbors should be a list of integers between 1 and $(graph[:J]).")
    end
    Jnew = graph[:J] + 1

    # Add node
    nodes = [graph[:nodes]; [neighbors]]
    new_x = [graph[:x]; x]
    new_y = [graph[:y]; y]

    # Add new node to neighbors' neighbors
    # And update adjacency and cost matrices
    adjacency = zeros(Jnew, Jnew)
    adjacency[1:graph[:J], 1:graph[:J]] = graph[:adjacency]

    delta_tau = zeros(Jnew, Jnew)
    delta_tau[1:graph[:J], 1:graph[:J]] = graph[:delta_tau]

    delta_i = zeros(Jnew, Jnew)
    delta_i[1:graph[:J], 1:graph[:J]] = graph[:delta_i]

    for i in neighbors
        nodes[i] = [nodes[i]; Jnew]
        distance = sqrt((new_x[i] - x)^2 + (new_y[i] - y)^2)

        # Adjacency
        adjacency[i, Jnew] = adjacency[Jnew, i] = 1

        # Travel cost: delta_tau
        delta_tau[i, Jnew] = delta_tau[Jnew, i] = distance

        # Building cost: delta_i
        delta_i[i, Jnew] = delta_i[Jnew, i] = distance
    end

    # Update number of degrees of liberty for Ijk
    ndeg = sum(tril(adjacency)) # nb of degrees of liberty in adjacency matrix

    # Update graph structure
    graph_new[:J] = Jnew
    graph_new[:x] = new_x
    graph_new[:y] = new_y
    graph_new[:nodes] = nodes
    graph_new[:adjacency] = adjacency
    graph_new[:delta_i] = delta_i
    graph_new[:delta_tau] = delta_tau
    graph_new[:ndeg] = ndeg

    # Update other parameters if they exist
    if haskey(graph, :Lj)
        graph_new[:Lj] = [graph[:Lj]; 1e-6] # ones(Jnew) / Jnew
    end
    if haskey(graph, :Hj)
        graph_new[:Hj] = [graph[:Hj]; 1e-6] # ones(Jnew)
    end
    if haskey(graph, :hj)
        graph_new[:hj] = graph_new[:Hj] ./ graph_new[:Lj]
    end
    if haskey(graph, :omegaj)
        graph_new[:omegaj] = [graph_new[:omegaj]; 1e-6] # ones(Jnew)
    end
    if haskey(graph, :Zjn)
        Zjn = fill(minimum(graph[:Zjn]), Jnew, size(graph[:Zjn], 2))
        Zjn[1:graph[:J], :] = graph[:Zjn]
        graph_new[:Zjn] = Zjn # ones(Jnew, size(graph[:Zjn], 2))
    end
    if haskey(graph, :region)
        closest_node = find_node(graph, x, y)
        closest_region = graph[:region][closest_node]
        graph_new[:region] = [graph[:region]; closest_region]
    end

    return graph_new
end

"""
    find_node(graph, x, y) -> Int64

Returns the index of the node closest to the coordinates (x,y) on the graph.

# Arguments
- `graph::Dict`: Dict that contains the underlying graph (created by create_graph())
- `x::Float64`: x coordinate on the graph (between 1 and w)
- `y::Float64`: y coordinate on the graph (between 1 and h)
"""
function find_node(graph, x, y)
    distance = (graph[:x] .- x).^2 + (graph[:y] .- y).^2
    _, id = findmin(distance)
    return id
end

"""
    remove_node(graph, i) -> Dict

Removes node i from the graph, returning an updated `graph` object.

# Arguments
- `graph::Dict`: Dict that contains the underlying graph (created by create_graph())
- `i::Int64`: index of the mode to be removed (integer between 1 and graph[:J])
"""
function remove_node(graph, i)

    if graph isa Dict
        graph_new = copy(dict)
    else
        graph_new = Dict(pairs(namedtuple))
    end

    if i < 1 || i > graph[:J] || i != floor(i)
        error("remove_node: node i should be an integer between 1 and $(graph[:J]).")
    end

    Jnew = graph[:J] - 1

    graph_new[:J] = Jnew
    graph_new[:x] = deleteat!(copy(graph[:x]), i)
    graph_new[:y] = deleteat!(copy(graph[:y]), i)
    nodes = deleteat!(copy(graph[:nodes]), i)
    for k in 1:Jnew
        node_k = filter(x -> x != i, nodes[k])
        nodes[k] = ifelse.(node_k .> i, node_k .- 1, node_k) # reindex nodes k > i to k-1
    end
    graph_new[:nodes] = nodes  # Assign the modified nodes to graph_new

    # Rebuild adjacency matrix, delta_i and delta_tau
    graph_new[:adjacency] = [graph[:adjacency][1:i-1, 1:i-1] graph[:adjacency][1:i-1, i+1:end];
                             graph[:adjacency][i+1:end, 1:i-1] graph[:adjacency][i+1:end, i+1:end]]

    graph_new[:delta_i] = [graph[:delta_i][1:i-1, 1:i-1] graph[:delta_i][1:i-1, i+1:end];
                           graph[:delta_i][i+1:end, 1:i-1] graph[:delta_i][i+1:end, i+1:end]]

    graph_new[:delta_tau] = [graph[:delta_tau][1:i-1, 1:i-1] graph[:delta_tau][1:i-1, i+1:end];
                             graph[:delta_tau][i+1:end, 1:i-1] graph[:delta_tau][i+1:end, i+1:end]]

    if haskey(graph, :across_obstacle)
        graph_new[:across_obstacle] = [graph[:across_obstacle][1:i-1, 1:i-1] graph[:across_obstacle][1:i-1, i+1:end];
                                       graph[:across_obstacle][i+1:end, 1:i-1] graph[:across_obstacle][i+1:end, i+1:end]]
    end

    if haskey(graph, :along_obstacle)
        graph_new[:along_obstacle] = [graph[:along_obstacle][1:i-1, 1:i-1] graph[:along_obstacle][1:i-1, i+1:end];
                                      graph[:along_obstacle][i+1:end, 1:i-1] graph[:along_obstacle][i+1:end, i+1:end]]
    end

    graph_new[:ndeg] = sum(tril(graph_new[:adjacency]))

    # Update other parameters if they exist
    if haskey(graph, :Lj)
        graph_new[:Lj] = deleteat!(copy(graph[:Lj]), i)
    end
    if haskey(graph, :Hj)
        graph_new[:Hj] = deleteat!(copy(graph[:Hj]), i)
    end
    if haskey(graph, :hj)
        graph_new[:hj] = deleteat!(copy(graph[:hj]), i)
    end
    if haskey(graph, :omegaj)
        graph_new[:omegaj] = deleteat!(copy(graph[:omegaj]), i)
    end
    if haskey(graph, :Zjn)
        graph_new[:Zjn] = vcat(graph[:Zjn][1:i-1, :], graph[:Zjn][i+1:end, :])
    end
    if haskey(graph, :region)
        graph_new[:region] = deleteat!(copy(graph[:region]), i)
    end

    return graph_new
end

