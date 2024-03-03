
function dict_to_namedtuple(dict)
    if !(dict isa NamedTuple)
        return NamedTuple{Tuple(keys(dict))}(values(dict))
    else
        return dict
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
    id = 1
    for i in 1:graph.J
        for j in graph.nodes[i]
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
        error("length(regions) must match number of edges (graph.J)")
    end
    res = zeros(ng)
    for i in 1:n
        @inbounds res[g[i]] += x[i]
    end
    return res
end