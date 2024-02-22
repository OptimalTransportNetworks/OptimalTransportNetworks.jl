
# using LinearAlgebra

"""
    create_graph(param, w, h; kwargs...)

Initialize the underlying graph, population and productivity parameters.

# Arguments
- `param`: Structure that contains the model parameters
- `w`: Number of nodes along the width of the underlying graph (integer)  
- `h`: Number of nodes along the height of the underlying graph (integer, odd if triangle)

# Keyword Arguments
- `type`: Either "map", "square", "triangle", or "custom" (default "map")
- `pareto_weights`: Vector of Pareto weights for each node (default ones(w*h)) 
- `adjacency`: Adjacency matrix (only used for custom network)
- `nregions`: Number of regions (only for partial mobility)
- `region`: Vector indicating region of each location (only for partial mobility)
"""
function create_graph(param, w, h; kwargs...)
    options = retrieve_options_create_graph(param, w, h; kwargs...)

    if options[:type] == "map"
        graph = create_map(w, h)
    elseif options[:type] == "triangle"
        graph = create_triangle(w, h)
    elseif options[:type] == "square"
        graph = create_square(w, h)
    elseif options[:type] == "custom"
        if !isadjacency(options[:adjacency])
            error("adjacency matrix must be square and symmetric, and only contain 0 and 1")
        end
        graph = create_custom(options[:adjacency], options[:x], options[:y])
    end

    param[:J] = graph[:J]

    if param[:mobility] == false
        param[:Lj] = ones(param[:J]) / param[:J]
        param[:Zjn] = ones(param[:J], param[:N])
        param[:Hj] = ones(param[:J])
        param[:hj] = param[:Hj] ./ param[:Lj]
        param[:hj][param[:Lj] .== 0] = 1
        param[:omegaj] = options[:omega]
    elseif param[:mobility] == true
        param[:Zjn] = ones(param[:J], param[:N])
        param[:Hj] = ones(param[:J])
    elseif param[:mobility] == 0.5
        param[:Zjn] = ones(param[:J], param[:N])
        param[:Hj] = ones(param[:J])
        graph[:region] = options[:region]
        param[:nregions] = options[:nregions]
        param[:omegar] = options[:omega]
        param[:Lr] = ones(options[:nregions]) / options[:nregions]
    end

    return param, graph
end

function isadjacency(M)
    res = true

    # check is matrix is square
    sz = size(M)
    if sz[1] != sz[2]
        print("$(@__FILE__): adjacency matrix is not square.\n")
        res = false
    end

    # check is matrix is symmetric 
    if any(M - M' .!= 0)
        print("$(@__FILE__): adjacency matrix is not symmetric.\n")
        res = false
    end

    # check if matrix has only 0's and 1's
    if any(M .!= 0 .& M .!= 1)
        print("$(@__FILE__): adjacency matrix should have only 0s and 1s.\n")
        res = false
    end

    return res
end

function retrieve_options_create_graph(param, w, h; kwargs...)
    options = Dict(
        :type => "map",
        :omega => ones(w * h),
        :adjacency => [],
        :x => [],
        :y => [],
        :nregions => 1,
        :region => []
    )

    for (k, v) in kwargs
        options[k] = v
    end

    options[:J] = w * h

    if options[:type] == "triangle"
        options[:J] = w * ceil(h / 2) + (w - 1) * (ceil(h / 2) - 1)
    end

    if options[:type] == "custom"
        if isempty(options[:adjacency])
            error("Custom network requires an adjacency matrix to be provided.")
        end

        if isempty(options[:x]) || isempty(options[:y])
            error("X and Y coordinates of locations must be provided.")
        end

        if length(options[:x]) != length(options[:y])
            error("The provided X and Y do not have the same size.")
        end

        if size(options[:adjacency], 1) != length(options[:x])
            error("The adjacency matrix and X should have the same number of locations.")
        end

        options[:J] = length(options[:x])
    end

    if param[:mobility] == 0.5 && isempty(options[:region])
        options[:region] = ones(options[:J])
    end

    if param[:mobility] == 0.5 && isempty(options[:nregions]) && !isempty(options[:region])
        options[:nregions] = length(unique(options[:region]))
    end

    if param[:mobility] == 0.5 && !isempty(options[:nregions]) && !isempty(options[:region])
        if length(unique(options[:region])) > options[:nregions]
            error("NRegions does not match the provided Region vector.")
        end
    end

    if length(options[:omega]) != options[:J] && param[:mobility] == 0
        println("Pareto weights should be a vector of size $(options[:J]). Using default instead.")
        options[:omega] = ones(options[:J])
    end

    if length(options[:omega]) != options[:nregions] && param[:mobility] == 0.5
        println("Pareto weights should be a vector of size $(options[:nregions]). Using default instead.")
        options[:omega] = ones(options[:nregions])
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
    nodes = [Dict(:neighbors => Vector{Int64}()) for _ in 1:J]

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

        nodes[i][:neighbors] = neighbors
    end

    adjacency = falses(J, J)
    for i in 1:J
        for j in nodes[i][:neighbors]
            adjacency[i, j] = 1
        end
    end

    ndeg = sum(tril(adjacency))

    return Dict(:J => J, :x => x, :y => y, :nodes => nodes, :adjacency => adjacency, :delta_i => delta, :delta_tau => delta, :ndeg => ndeg)
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

    nodes = [Dict(:neighbors => Vector{Int64}()) for _ in 1:J]

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

            nodes[id][:neighbors] = neighbors
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

            nodes[id][:neighbors] = neighbors
        end
    end

    adjacency = falses(J, J)
    for i in 1:J
        for j in nodes[i][:neighbors]
            adjacency[i, j] = 1
        end
    end

    ndeg = sum(tril(adjacency))

    return Dict(:J => J, :x => x, :y => y, :nodes => nodes, :adjacency => adjacency, :delta_i => delta, :delta_tau => delta, :ndeg => ndeg)
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
    nodes = [Dict(:neighbors => []) for _ in 1:J]

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

        nodes[i][:neighbors] = neighbors
    end

    adjacency = falses(J, J)
    for i in 1:J
        for j in nodes[i][:neighbors]
            adjacency[i, j] = 1
        end
    end

    ndeg = sum(tril(adjacency))

    return Dict(:J => J, :x => x, :y => y, :nodes => nodes, :adjacency => adjacency, :delta_i => delta, :delta_tau => delta, :ndeg => ndeg)
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
    nodes = [Dict(:neighbors => Vector{Int64}()) for _ in 1:J]

    for i in 1:J
        neighbors = findall(adjacency[i, :] .== 1)
        nodes[i][:neighbors] = neighbors
    end

    ndeg = sum(tril(adjacency)) # TODO: check if this is correct

    delta = zeros(J, J)
    xx = repeat(x, 1, J)
    yy = repeat(y, 1, J)
    delta = sqrt.((xx .- xx').^2 + (yy .- yy').^2)
    delta[adjacency .== 0] .= 0.0

    return Dict(:J => J, :x => x, :y => y, :nodes => nodes, :adjacency => adjacency, :delta_i => delta, :delta_tau => delta, :ndeg => ndeg)
end

