
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
function create_auxdata(param, graph, I)
    # Make named tuples
    # param = dict_to_namedtuple(param)
    # graph = dict_to_namedtuple(graph)

    # Initialize kappa
    kappa = max.(I.^param[:gamma] ./ graph.delta_tau, param[:kappa_min])
    kappa[.!graph.adjacency] .= 0
    kappa_ex = kappa_extract(graph, kappa)  # extract the ndeg free values of matrix kappa (due to symmetry)

    # Create matrix A
    # Note: matrix A is of dimension J*ndeg and takes value 1 when node J is
    # the starting point of edge ndeg, and -1 when it is the ending point. It 
    # is used to make hessian and jacobian calculations using matrix algebra
    A = zeros(graph.J, graph.ndeg)
    id = 1
    for j in 1:graph.J
        for k in 1:length(graph.nodes[j])
            if graph.nodes[j][k] > j
                A[j, id] = 1
                A[graph.nodes[j][k], id] = -1
                id += 1
            end
        end
    end

    # Store in auxdata
    auxdata = (
        param = param,
        graph = graph,
        kappa = kappa,
        kappa_ex = kappa_ex,
        # Iex = kappa_extract(graph, I),
        # delta_tau_ex = kappa_extract(graph, graph.delta_tau),
        A = A,
        Apos = max.(A, 0),
        Aneg = max.(-A, 0)
    )

    return auxdata
end
