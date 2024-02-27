
"""
    add_node(param, graph, x, y, neighbors)

Add a node in position (x,y) and list of neighbors. The new node is given an index J+1.

# Arguments
- `param`: Dict that contains the model's parameters
- `graph`: Dict that contains the underlying graph (created by create_graph)
- `x`: x-coordinate of the new node (any real number)
- `y`: y-coordinate of the new node (any real number)
- `neighbors`: list of nodes to which it is connected (1 x n list of node indices between 1 and J, where n is an arbitrary # of neighbors) 

The cost matrices delta_tau and delta_i are parametrized as a function of Euclidian distance between nodes.

Returns the updated graph and param Dict (param is affected too because the variable Zjn, Lj, Hj and others are reset to a uniform dist.)
"""
function add_node(param, graph, x, y, neighbors)
    # check validity of neighbors list
    if any(neighbors .!= floor.(neighbors)) || any(neighbors .< 1) || any(neighbors .> graph["J"])
        println("neighbors should be a list of integers between 1 and $(graph["J"]).")
    end

    # Add node
    nodes = graph["nodes"]
    nodes[graph["J"]+1] = Dict("neighbors" => neighbors)
    new_x = [graph["x"]; x]
    new_y = [graph["y"]; y]

    # Add new node to neighbors' neighbors
    # and update adjacency and cost matrices
    adjacency = zeros(graph["J"]+1, graph["J"]+1)
    adjacency[1:graph["J"], 1:graph["J"]] = graph["adjacency"]

    delta_tau = zeros(graph["J"]+1, graph["J"]+1)
    delta_tau[1:graph["J"], 1:graph["J"]] = graph["delta_tau"]

    delta_i = zeros(graph["J"]+1, graph["J"]+1)
    delta_i[1:graph["J"], 1:graph["J"]] = graph["delta_i"]

    for i in neighbors
        nodes[i]["neighbors"] = [nodes[i]["neighbors"]; graph["J"]+1]

        distance = sqrt((new_x[i]-x)^2 + (new_y[i]-y)^2)

        # adjacency
        adjacency[i, graph["J"]+1] = 1
        adjacency[graph["J"]+1, i] = 1

        # travel cost: delta_tau
        delta_tau[i, graph["J"]+1] = distance
        delta_tau[graph["J"]+1, i] = distance

        # building cost: delta_i
        delta_i[i, graph["J"]+1] = distance
        delta_i[graph["J"]+1, i] = distance
    end

    # update number of degrees of liberty for Ijk
    ndeg = sum(tril(adjacency)) # nb of degrees of liberty in adjacency matrix

    # return new graph structure
    graph = Dict("J" => graph["J"]+1, "x" => new_x, "y" => new_y, "nodes" => nodes, "adjacency" => adjacency, "delta_i" => delta_i, "delta_tau" => delta_tau, "ndeg" => ndeg)

    # now, update the param structure
    param["J"] = graph["J"]
    param["Lj"] = ones(graph["J"]) / graph["J"]
    param["Hj"] = ones(graph["J"])
    param["hj"] = param["Hj"] ./ param["Lj"]
    param["omegaj"] = ones(graph["J"])
    param["Zjn"] = ones(graph["J"], param["N"])

    return param, graph
end


# Please note that in Julia, we use `Dict` to represent structures as in Matlab. The keys of the `Dict` are strings that correspond to the field names in the Matlab structure. Also, the `tril` function is not built-in in Julia, you may need to use a package like `LinearAlgebra` to use it.

"""
    find_node(graph, x, y)

Returns the index of the node closest to the coordinates (x,y) on the graph.

# Arguments
- `graph`: structure that contains the underlying graph (created by
create_map, create_rectangle or create_triangle functions)
- `x`: x coordinate on the graph between 1 and w
- `y`: y coordinate on the graph between 1 and h
"""
function find_node(graph, x, y)
    distance = (graph.x .- x).^2 + (graph.y .- y).^2
    _, id = findmin(distance)
    return id
end



"""
    remove_node(param, graph, i)

Removes node i from the graph.

# Arguments
- `param`: structure that contains the model's parameters
- `graph`: structure that contains the underlying graph (created by create_graph)
- `i`: index of the mode to be removed (integer between 1 and graph.J)

Returns the updated graph and param structure (param is affected too because the variable Zjn, Lj, Hj and others are changed).
"""
function remove_node(param, graph, i)
    if i < 1 || i > graph.J || i != floor(i)
        println("remove_node: node i should be an integer between 1 and $(graph.J).")
        return
    end

    # create new nodes structure
    nodes = [Dict() for _ in 1:graph.J-1]
    x = zeros(graph.J-1)
    y = zeros(graph.J-1)

    for k in 1:graph.J-1
        cursor = k + (k >= i)

        nodes[k]["neighbors"] = setdiff(graph.nodes[cursor]["neighbors"], i) # remove this node from others' neighbors.
        nodes[k]["neighbors"] = [n-1 for n in nodes[k]["neighbors"] if n > i] # reindex nodes k > i to k-1

        x[k] = graph.x[cursor]
        y[k] = graph.y[cursor]
    end

    # rebuild adjacency matrix, delta_i and delta_tau
    adjacency = [graph.adjacency[1:i-1, 1:i-1] graph.adjacency[1:i-1, i+1:end];
                 graph.adjacency[i+1:end, 1:i-1] graph.adjacency[i+1:end, i+1:end]]

    delta_i = [graph.delta_i[1:i-1, 1:i-1] graph.delta_i[1:i-1, i+1:end];
               graph.delta_i[i+1:end, 1:i-1] graph.delta_i[i+1:end, i+1:end]]

    delta_tau = [graph.delta_i[1:i-1, 1:i-1] graph.delta_i[1:i-1, i+1:end];
                 graph.delta_i[i+1:end, 1:i-1] graph.delta_i[i+1:end, i+1:end]]

    ndeg = sum(tril(adjacency))

    # return new graph structure
    graph = Dict("J" => graph.J-1, "x" => x, "y" => y, "nodes" => nodes, "adjacency" => adjacency,
                 "delta_i" => delta_i, "delta_tau" => delta_tau, "ndeg" => ndeg)

    # now, update the param structure
    param["J"] = graph["J"]
    param["Lj"] = vcat(param["Lj"][1:i-1], param["Lj"][i+1:end])
    param["Hj"] = vcat(param["Hj"][1:i-1], param["Hj"][i+1:end])
    param["hj"] = vcat(param["hj"][1:i-1], param["hj"][i+1:end])
    param["omegaj"] = vcat(param["omegaj"][1:i-1], param["omegaj"][i+1:end])
    param["Zjn"] = vcat(param["Zjn"][1:i-1, :], param["Zjn"][i+1:end, :])

    return param, graph
end


# Please note that this translation assumes that `param` and `graph` are dictionaries and that `graph.nodes` is an array of dictionaries. If the actual data structures are different, you may need to adjust the code accordingly.