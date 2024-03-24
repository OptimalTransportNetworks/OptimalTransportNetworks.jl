
# using LinearAlgebra

"""
    create_graph(param, w = 11, h = 11; type = "map", kwargs...) -> Dict, NamedTuple

Initialize the underlying graph, population and productivity parameters.

# Arguments
- `param::Dict`: Structure that contains the model parameters
- `w::Int64=11`: Number of nodes along the width of the underlying graph (integer)  
- `h::Int64=11`: Number of nodes along the height of the underlying graph (integer, odd if triangle)

# Keyword Arguments
- `type::String="map"`: Either "map", "square", "triangle", or "custom" 
- `omega::Vector{Float64}`: Vector of Pareto weights for each node or region in partial mobility case (default ones(J or nregions))
- `Zjn::Matrix{Float64}`: J x N matrix of producties per node (j = 1:J) and good (n = 1:N) (default ones(J, N))
- `adjacency::BitMatrix`: J x J Adjacency matrix (only used for custom network)
- `x::Vector{Float64}`: x coordinate (longitude) of each node (only used for custom network)
- `y::Vector{Float64}`: y coordinate (latitude) of each node (only used for custom network)
- `Lj::Vector{Float64}`: Vector of populations in each node (j = 1:J) (only for no mobility)
- `Hj::Vector{Float64}`: Vector of immobile good in each node (j = 1:J) (e.g. housing, default ones(J))
- `Lr::Vector{Float64}`: Vector of populations in each region (r = 1:nregions) (only for partial mobility)
- `region::Vector{Int64}`: Vector indicating region of each location (only for partial mobility)

# Examples
```julia
param, graph = create_graph(init_parameters())
```
"""
function create_graph(param, w = 11, h = 11; type = "map", kwargs...)

    options = retrieve_options_create_graph(param, w, h, type; kwargs...)

    if type == "map"
        graph = create_map(w, h)
    elseif type == "triangle"
        graph = create_triangle(w, h)
    elseif type == "square"
        graph = create_square(w, h)
    elseif type == "custom"
        graph = create_custom(options[:adjacency], options[:x], options[:y])
    end

    param[:J] = graph.J
    param[:Zjn] = haskey(options, :Zjn) ? options[:Zjn] : ones(graph.J, param[:N])
    param[:Hj] = haskey(options, :Hj) ? options[:Hj] : ones(graph.J)

    if param[:mobility] == false
        param[:Lj] = haskey(options, :Lj) ? options[:Lj] : ones(graph.J) / graph.J
        param[:hj] = param[:Hj] ./ param[:Lj]
        param[:hj][param[:Lj] .== 0] .= 1
        param[:omegaj] = options[:omega]
    elseif param[:mobility] == 0.5
        graph = namedtuple_to_dict(graph)
        graph[:region] = options[:region]
        graph = dict_to_namedtuple(graph)
        param[:omegar] = options[:omega]
        param[:Lr] = options[:Lr]
        param[:nregions] = options[:nregions]
    end

    # Running checks on population / productivity / housing parameters
    check_graph_param(param)

    return param, graph
end

function isadjacency(M)
    res = true

    # check is matrix is square
    sz = size(M)
    if sz[1] != sz[2]
        @warn "adjacency matrix is not square.\n"
        res = false
    end

    # check is matrix is symmetric 
    if any(M - M' .!= 0)
        @warn "adjacency matrix is not symmetric.\n"
        res = false
    end

    # check if matrix has only 0's and 1's
    if any(M .!= 0 .& M .!= 1)
        @warn "adjacency matrix should have only 0s and 1s.\n"
        res = false
    end

    return res
end

function retrieve_options_create_graph(param, w, h, type; kwargs...)

    options = Dict()
    options[:type] = type

    for (k, v) in kwargs
        options[k] = v
    end

    if type == "custom"
        if !haskey(options, :adjacency)
            error("Custom network requires an adjacency matrix to be provided.")
        end
        if !isadjacency(options[:adjacency])
            error("adjacency matrix must be square and symmetric, and only contain 0 and 1")
        end
        if !haskey(options, :x) || !haskey(options, :y)
            error("X and Y coordinates of locations must be provided.")
        end
        J = length(options[:x])
        if J != length(options[:y])
            error("The provided X and Y do not have the same size.")
        end
        if size(options[:adjacency], 1) != J
            error("The adjacency matrix and X should have the same number of locations.")
        end
    elseif type == "triangle"
        J = Int(w * ceil(h / 2) + (w - 1) * (ceil(h / 2) - 1))
    else
        J = w * h
    end

    if haskey(options, :Zjn) && size(options[:Zjn]) != (J, param[:N])
        error("Zjn matrix should be of size J x N, where J = $J is number of nodes and N = $(param[:N]) number of goods.")
    end
    if haskey(options, :Lj) && length(options[:Lj]) != J
        error("Length of 'Lj' vector should match the number of locations/nodes in the network = $J.")
    end
    if haskey(options, :Hj) && length(options[:Hj]) != J
        error("Length of 'Hj' vector should match the number of locations/nodes in the network = $J.")
    end

    if param[:mobility] == 0.5 # || haskey(param, :nregions) || haskey(param, :region) || haskey(param, :omegar) || haskey(param, :Lr)
        if !haskey(options, :Lr)
            error("For partial mobility case need to provide a parameter 'Lr' containing a vector with the total populations of each region")        
        end
        if !haskey(options, :region)
            error("For partial mobility case need to provide a parameter 'region' containing an integer vector that maps each node of the graph to a region. The vector should have values in the range 1:length(Lr) and be of length $J.")        
        end
        if length(options[:region]) != J
            error("Length of 'region' vector should match the number of locations/nodes in the network = $J.")
        end
        options[:nregions] = length(options[:Lr])
        if !isinteger(options[:region][1]) || typeof(options[:region][1]) == Float64
            error("'region' needs to be an integer vector.")
        end
        if minimum(options[:region]) != 1 || maximum(options[:region]) > options[:nregions]
            error("'region' values need to be in the range 1:nregions = $(options[:nregions]).")
        end
        if length(unique(options[:region])) != options[:nregions]
            error("'region' does not match 'Lr'.")
        end
        if !haskey(options, :omega)
            options[:omega] = ones(options[:nregions])
        elseif length(options[:omega]) != options[:nregions]
            error("Pareto weights should be a vector of size $(options[:nregions]).")
        end
    elseif param[:mobility] == 0 
        if !haskey(options, :omega) 
            options[:omega] = ones(J)
        elseif length(options[:omega]) != J
            error("Pareto weights should be a vector of size $J.")
        end
    end

    return options
end

"""
    create_map(w, h)

Creates a square graph structure with width `w` and height `h` 
(nodes have 8 neighbors in total, along horizontal and vertical 
dimensions and diagonals)

# Arguments
- `w`: Width of graph (i.e. the number of nodes along horizontal dimension), must be an integer
- `h`: Height of graph (i.e. the number of nodes along vertical dimension), must be an integer
"""
function create_map(w, h)
    J = w * h
    nodes = [Vector{Int64}() for _ in 1:J]

    delta = zeros(J, J)
    x = zeros(Int64, J)
    y = zeros(Int64, J)
    for i in 1:J
        neighbors = Vector{Int64}()

        y[i] = floor((i - 1) / w) + 1
        x[i] = i - w * (y[i] - 1)

        if x[i] < w
            push!(neighbors, x[i] + 1 + w * (y[i] - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + 1 + w * (y[i] - 1)] = 1
        end
        if x[i] > 1
            push!(neighbors, x[i] - 1 + w * (y[i] - 1)) 
            delta[x[i] + w * (y[i] - 1), x[i] - 1 + w * (y[i] - 1)] = 1
        end
        if y[i] < h
            push!(neighbors, x[i] + w * (y[i] + 1 - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + w * (y[i] + 1 - 1)] = 1 
        end
        if y[i] > 1
            push!(neighbors, x[i] + w * (y[i] - 1 - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + w * (y[i] - 1 - 1)] = 1
        end
        if x[i] < w && y[i] < h
            push!(neighbors, x[i] + 1 + w * (y[i] + 1 - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + 1 + w * (y[i] + 1 - 1)] = sqrt(2)
        end
        if x[i] < w && y[i] > 1 
            push!(neighbors, x[i] + 1 + w * (y[i] - 1 - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + 1 + w * (y[i] - 1 - 1)] = sqrt(2)
        end
        if x[i] > 1 && y[i] < h
            push!(neighbors, x[i] - 1 + w * (y[i] + 1 - 1))
            delta[x[i] + w * (y[i] - 1), x[i] - 1 + w * (y[i] + 1 - 1)] = sqrt(2)
        end
        if x[i] > 1 && y[i] > 1
            push!(neighbors, x[i] - 1 + w * (y[i] - 1 - 1)) 
            delta[x[i] + w * (y[i] - 1), x[i] - 1 + w * (y[i] - 1 - 1)] = sqrt(2)
        end

        nodes[i] = neighbors
    end

    adjacency = falses(J, J)
    for i in 1:J
        for j in nodes[i]
            adjacency[i, j] = 1
        end
    end

    ndeg = sum(tril(adjacency))

    return (J=J, x=x, y=y, nodes=nodes, adjacency=adjacency, delta_i=delta, delta_tau=delta, ndeg=ndeg)
end


"""
    create_triangle(w, h)

Creates a triangular graph structure with width `w` and height `h` 
(each node is the center of a hexagon and each node has 6 neighbors, 
horizontal and along the two diagonals)

# Arguments
- `w`: Width of graph (i.e. the max number of nodes along horizontal dimension), 
must be an integer
- `h`: Height of graph (i.e. the max number of nodes along vertical dimension),  
must be an odd integer
"""
function create_triangle(w, h)
    if h % 2 == 0
        error("create_triangle(w,h): argument h must be an odd number.")
    end

    rows_outer = Int(ceil(h / 2))
    rows_inner = Int(ceil(h / 2) - 1)
    J = Int(w * rows_outer + (w - 1) * rows_inner)

    nodes = [Vector{Int64}() for _ in 1:J]

    delta = falses(J, J)
    x = zeros(J) # Needs to be Float64
    y = zeros(Int64, J)
    for j in 1:rows_outer
        for i in 1:w
            neighbors = Vector{Int64}()

            id = i + w * (j - 1) + (w - 1) * (j - 1)
            x[id] = i
            y[id] = 1 + (j - 1) * 2

            if i < w
                push!(neighbors, id + 1)
                delta[id, id + 1] = 1

                if j < rows_outer
                    push!(neighbors, id + w)
                    delta[id, id + w] = 1
                end
                if j > 1
                    push!(neighbors, id - (w - 1))
                    delta[id, id - (w - 1)] = 1
                end
            end
            if i > 1
                push!(neighbors, id - 1)
                delta[id, id - 1] = 1
                if j < rows_outer
                    push!(neighbors, id + (w - 1))
                    delta[id, id + (w - 1)] = 1
                end
                if j > 1
                    push!(neighbors, id - w)
                    delta[id, id - w] = 1
                end
            end

            nodes[id] = neighbors
        end
    end

    for j in 1:rows_inner
        for i in 1:w - 1
            neighbors = Vector{Int64}()

            id = i + w * j + (w - 1) * (j - 1)
            x[id] = i + 0.5
            y[id] = 2 + (j - 1) * 2

            if i < w - 1
                push!(neighbors, id + 1)
                delta[id, id + 1] = 1
            end
            if i > 1
                push!(neighbors, id - 1)
                delta[id, id - 1] = 1
            end
            push!(neighbors, id + w)
            delta[id, id + w] = 1
            push!(neighbors, id - (w - 1))
            delta[id, id - (w - 1)] = 1
            push!(neighbors, id + (w - 1))
            delta[id, id + (w - 1)] = 1
            push!(neighbors, id - w)
            delta[id, id - w] = 1

            nodes[id] = neighbors
        end
    end

    adjacency = falses(J, J)
    for i in 1:J
        for j in nodes[i]
            adjacency[i, j] = 1
        end
    end

    ndeg = sum(tril(adjacency))

    return (J=J, x=x, y=y, nodes=nodes, adjacency=adjacency, delta_i=delta, delta_tau=delta, ndeg=ndeg)
end



"""
    create_square(w, h)

Creates a square graph structure
with width w and height h (nodes have 4 neighbors in total, along
horizontal and vertical dimensions, NOT diagonals)

# Arguments
- `w`: width of graph (ie. the number of nodes along horizontal
dimension), must be an integer 
- `h`: height of graph (ie. the number of nodes along vertical
dimension), must be an integer
"""
function create_square(w, h)
    J = w * h
    nodes = [Vector{Int64}() for _ in 1:J]

    delta = falses(J, J)
    x = zeros(Int64, J)
    y = zeros(Int64, J)
    for i in 1:J
        neighbors = Vector{Int64}()

        y[i] = floor((i - 1) / w) + 1
        x[i] = i - w * (y[i] - 1)

        if x[i] < w
            push!(neighbors, x[i] + 1 + w * (y[i] - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + 1 + w * (y[i] - 1)] = 1
        end
        if x[i] > 1
            push!(neighbors, x[i] - 1 + w * (y[i] - 1))
            delta[x[i] + w * (y[i] - 1), x[i] - 1 + w * (y[i] - 1)] = 1
        end
        if y[i] < h
            push!(neighbors, x[i] + w * (y[i] + 1 - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + w * (y[i] + 1 - 1)] = 1
        end
        if y[i] > 1
            push!(neighbors, x[i] + w * (y[i] - 1 - 1))
            delta[x[i] + w * (y[i] - 1), x[i] + w * (y[i] - 1 - 1)] = 1
        end

        nodes[i] = neighbors
    end

    adjacency = falses(J, J)
    for i in 1:J
        for j in nodes[i]
            adjacency[i, j] = 1
        end
    end

    ndeg = sum(tril(adjacency))

    return (J=J, x=x, y=y, nodes=nodes, adjacency=adjacency, delta_i=delta, delta_tau=delta, ndeg=ndeg)
end


"""
    create_custom(adjacency, x, y)

Creates a custom graph structure with given adjacency matrix, x and y vectors 
of coordinates.

# Arguments
- `adjacency`: Adjacency matrix 
- `x`: Vector of x coordinates of locations
- `y`: Vector of y coordinates of locations
"""
function create_custom(adjacency, x, y)
    J = length(x)
    nodes = [Vector{Int64}() for _ in 1:J]

    for i in 1:J
        neighbors = findall(adjacency[i, :] .== 1)
        nodes[i] = neighbors
    end

    ndeg = sum(tril(adjacency)) # TODO: check if this is correct

    delta = zeros(J, J)
    xx = repeat(x, 1, J)
    yy = repeat(y, 1, J)
    delta = sqrt.((xx .- xx').^2 + (yy .- yy').^2)
    delta[adjacency .== 0] .= 0.0

    return (J=J, x=x, y=y, nodes=nodes, adjacency=adjacency, delta_i=delta, delta_tau=delta, ndeg=ndeg)
end


