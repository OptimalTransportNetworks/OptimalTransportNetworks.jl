
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