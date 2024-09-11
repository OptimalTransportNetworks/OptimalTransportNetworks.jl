
"""
    apply_geography(graph, geography; kwargs...) -> updated_graph

Update the network building costs of a graph based on geographical features and remove edges impeded by geographical barriers. 
Aversion to altitude changes rescales building infrastructure costs `delta_i` by (see also user manual to MATLAB toolbox):\n

`1 + alpha_up * max(0, Z2-Z1)^beta_up + alpha_down * max(0, Z1-Z2)^beta_down`\n

and similarly for graph traversal costs `delta_tau`.

# Arguments
- `graph`: Dict or NamedTuple that contains the network graph to which the geographical features will be applied.
- `geography`: Dict or NamedTuple representing the geographical features, with the following fields:\n
   - `z::Vector{Float64}`: (Optional) J x 1 vector containing the z-coordinate (elevation) for each node, or `nothing` if no elevation data.\n
   - `z_is_friction::Bool`: (Optional) logical value indicate that `z` represents friction rather than elevation. In that case, the measure of building cost is the average friction of the two nodes mean(Z1,Z2) rather than the difference Z2-Z1.\n
   - `obstacles::Matrix{Int64}`: (Optional) Nobs x 2 matrix specifying (i, j) pairs of nodes that are connected by obstacles, where Nobs is the number of obstacles, or `nothing` if no obstacles.

# Keyword Arguments
- `across_obstacle_delta_i::Float64=Inf`: Rescaling parameter for building cost that crosses an obstacle.
- `along_obstacle_delta_i::Float64=Inf`: Rescaling parameter for building cost that goes along an obstacle.
- `across_obstacle_delta_tau::Float64=Inf`: Rescaling parameter for transport cost that crosses an obstacle.
- `along_obstacle_delta_tau::Float64=Inf`: Rescaling parameter for transport cost that goes along an obstacle.
- `alpha_up_i::Float64=0`: Building cost scale parameter for roads that go up in elevation.
- `beta_up_i::Float64=1`: Building cost elasticity parameter for roads that go up in elevation.
- `alpha_up_tau::Float64=0`: Transport cost scale parameter for roads that go up in elevation.
- `beta_up_tau::Float64=1`: Transport cost elasticity parameter for roads that go up in elevation.
- `alpha_down_i::Float64=0`: Building cost scale parameter for roads that go down in elevation.
- `beta_down_i::Float64=1`: Building cost elasticity parameter for roads that go down in elevation.
- `alpha_down_tau::Float64=0`: Transport cost scale parameter for roads that go down in elevation.
- `beta_down_tau::Float64=1`: Transport cost elasticity parameter for roads that go down in elevation.

# Examples
```julia
param, graph = create_graph(init_parameters())
geography = (z = 10*(rand(graph[:J]) .> 0.95), obstacles = [1 15; 70 72])
updated_graph = apply_geography(graph, geography)

plot_graph(updated_graph, geography = geography, obstacles = true)
```
"""
function apply_geography(graph, geography; kwargs...)

    graph = dict_to_namedtuple(graph) # convert graph to namedtuple for cleaner access

    options = Dict(
        :across_obstacle_delta_i => Inf,
        :along_obstacle_delta_i => Inf, # 1
        :across_obstacle_delta_tau => Inf,
        :along_obstacle_delta_tau => Inf, # 1
        :alpha_up_i => 0,
        :beta_up_i => 1,
        :alpha_up_tau => 0,
        :beta_up_tau => 1,
        :alpha_down_i => 0,
        :beta_down_i => 1,
        :alpha_down_tau => 0,
        :beta_down_tau => 1
    )

    for (k, v) in kwargs
        options[k] = v
    end

    op = dict_to_namedtuple(options)
    
    z = haskey(geography, :z) ? geography[:z] : nothing
    obstacles = haskey(geography, :obstacles) ? geography[:obstacles] : nothing
    z_is_friction = haskey(geography, :z_is_friction) && geography[:z_is_friction] === true 

    # New graph object components
    delta_i_new = deepcopy(graph.delta_i)
    delta_tau_new = deepcopy(graph.delta_tau)

    if obstacles !== nothing
        adjacency_new = deepcopy(graph.adjacency)
        nodes_new = deepcopy(graph.nodes)
    end

    # Embed geographical barriers into network building costs
    if z !== nothing
        for i in 1:graph.J
            for neighbor in graph.nodes[i]
                distance = sqrt((graph.x[i] - graph.x[neighbor])^2 + (graph.y[i] - graph.y[neighbor])^2)
                Delta_z = z_is_friction ? (z[i] + z[neighbor]) / 2 : z[neighbor] - z[i]
                delta_i_new[i, neighbor] = distance * (1 + op.alpha_up_i * max(0, Delta_z)^op.beta_up_i + op.alpha_down_i * max(0, -Delta_z)^op.beta_down_i)
                delta_tau_new[i, neighbor] = distance * (1 + op.alpha_up_tau * max(0, Delta_z)^op.beta_up_tau + op.alpha_down_tau * max(0, -Delta_z)^op.beta_down_tau)
            end
        end
    end

    # Remove edges where geographical barriers are (rivers)
    if obstacles !== nothing

        # Store initial delta matrics (avoid double counting)
        delta_i_new_tmp = deepcopy(delta_i_new)
        delta_tau_new_tmp = deepcopy(delta_tau_new)

        # Initialize flags for across and along obstacles
        sz = size(obstacles)[1]
        across_obstacle = falses(graph.J, graph.J)
        along_obstacle = falses(graph.J, graph.J)
        remove_edge = isinf(op.across_obstacle_delta_i) || isinf(op.across_obstacle_delta_tau) # remove the edge

        for i in 1:graph.J
            neighbors = graph.nodes[i]
            for j in neighbors
                # check if edge (i,j) intersects any of the obstacles
                k = 1
                has_been_destroyed = false
                while !has_been_destroyed && k <= sz
                    io, jo = obstacles[k, :]
                    if i != io || j != jo # if the link is not the obstacle itself (only across obstacle links)
                        Z2_obj = [graph.x[jo], graph.y[jo]]
                        Z1_obj = [graph.x[io], graph.y[io]]
                        Z2 = [graph.x[j], graph.y[j]]
                        Z1 = [graph.x[i], graph.y[i]]
                        # This computes the directional vector of the obstacle line segment.
                        vec = Z2_obj - Z1_obj
                        # This calculates a vector that is normal (perpendicular) to the obstacle line segment vector. It effectively rotates the original vector by 90 degrees anticlockwise.
                        normvec = [-vec[2], vec[1]]
                        # This computes the dot product of the normal vector with the difference between the endpoints of the current line segment and the first point of the obstacle line segment. The product of the two dot products is used to determine if points Z1 and Z2 are on opposite sides of the obstacle line segmen
                        val1 = (normvec' * (Z1 - Z1_obj)) * (normvec' * (Z2 - Z1_obj))
                        vec = Z2 - Z1
                        normvec = [-vec[2], vec[1]]
                        val2 = (normvec' * (Z1_obj - Z1)) * (normvec' * (Z2_obj - Z1))
                        # val1 <= 0 && val2 <= 0: This is true if both pairs of points (Z1 with Z2, and Z1_obj with Z2_obj) are on opposite sides of each other's line segments, suggesting that the line segments intersect.
                        # normvec' * (Z2_obj - Z1_obj): Checking that the dot product of the normal vector and the obstacle vector is not close to zero (which would imply collinearity).
                        if (val1 <= 0 && val2 <= 0) && !isapprox(normvec' * (Z2_obj - Z1_obj), 0) # if the two edges intersect and are not colinears (=along obstacle)
                            if remove_edge
                                rmi = findfirst(==(i), nodes_new[j])
                                if rmi !== nothing
                                    deleteat!(nodes_new[j], rmi)
                                    # if isempty(nodes_new[j])
                                    #     deleteat!(nodes_new, j)
                                    # end
                                end
                                rmj = findfirst(==(j), nodes_new[i])
                                if rmj !== nothing
                                    deleteat!(nodes_new[i], rmj)
                                    # if isempty(nodes_new[i])
                                    #     deleteat!(nodes_new, i)
                                    # end
                                end
                                
                                adjacency_new[i, j] = false
                                adjacency_new[j, i] = false
                                has_been_destroyed = true
                            else
                                # Or make it costly to cross
                                across_obstacle[i, j] = true
                                delta_i_new[i, j] = delta_i_new_tmp[i, j] * op.across_obstacle_delta_i
                                delta_tau_new[i, j] = delta_tau_new_tmp[i, j] * op.across_obstacle_delta_tau
                                across_obstacle[j, i] = true
                                delta_i_new[j, i] = delta_i_new_tmp[j, i] * op.across_obstacle_delta_i
                                delta_tau_new[j, i] = delta_tau_new_tmp[j, i] * op.across_obstacle_delta_tau
                            end
                        end
                    end
                    k += 1
                end
            end
        end

        # This is for edges along obstable: allowing different delta (e.g., water transport on river)
        if isinf(op.along_obstacle_delta_i) || isinf(op.along_obstacle_delta_tau)
            # if infinite, remove edge
            for (io, jo) in zip(obstacles[:, 1], obstacles[:, 2])
                rmio = findfirst(==(io), nodes_new[jo])
                if rmio !== nothing
                    deleteat!(nodes_new[jo], rmio)
                    # if isempty(nodes_new[jo])
                    #     deleteat!(nodes_new, jo)
                    # end
                end
                rmjo = findfirst(==(jo), nodes_new[io])
                if rmjo !== nothing
                    deleteat!(nodes_new[io], rmjo)
                    # if isempty(nodes_new[io])
                    #     deleteat!(nodes_new, io)
                    # end
                end
                adjacency_new[io, jo] = false
                adjacency_new[jo, io] = false
            end
        else 
            for (io, jo) in zip(obstacles[:, 1], obstacles[:, 2])
                delta_i_new[io, jo] *= op.along_obstacle_delta_i
                delta_i_new[jo, io] *= op.along_obstacle_delta_i
                delta_tau_new[io, jo] *= op.along_obstacle_delta_tau
                delta_tau_new[jo, io] *= op.along_obstacle_delta_tau
                along_obstacle[io, jo] = true
                along_obstacle[jo, io] = true
                # This should be redundant if the above block works as intended:
                across_obstacle[io, jo] = false
                across_obstacle[jo, io] = false
            end
        end
    end

    # Creating new object
    graph_new = namedtuple_to_dict(graph)
    graph_new[:delta_i] = delta_i_new
    graph_new[:delta_tau] = delta_tau_new

    if obstacles !== nothing
        graph_new[:nodes] = nodes_new
        graph_new[:adjacency] = adjacency_new
        # make sure that the degrees of freedom of the updated graph match the # of links
        graph_new[:ndeg] = sum(tril(adjacency_new))
        graph_new[:across_obstacle] = across_obstacle
        graph_new[:along_obstacle] = along_obstacle
    end

    return graph_new
end

