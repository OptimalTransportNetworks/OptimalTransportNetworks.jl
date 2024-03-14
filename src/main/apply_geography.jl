
"""
    apply_geography(graph, geography; kwargs...)

Applies a geography structure to a graph by updating its network building costs and removing edges where geographical barriers (obstacles) are.

The geography structure should contain the fields:
- z: a Jx1 vector containing the z coordinate of each node
- obstacles: a Nobs*2 matrix of (i,j) pairs of nodes that are linked by the obstacles (Nobs is an arbitrary number of obstacles)

The function apply_geography takes the geography structure as an input and uses it to calibrate the building cost delta_i. Aversion to altitude changes rescales building costs by (see documentation):
    1+alpha_up*max(0,Z2-Z1)^beta_up+alpha_down*max(0,Z1-Z2)^beta_down.

Optional arguments:
- 'across_obstacle_delta_i': rescaling parameter for building cost that cross an obstacle (default Inf)
- 'along_obstacle_delta_i': rescaling parameter for building cost that goes along an obstacle (default 1)
- 'across_obstacle_delta_tau': rescaling parameter for transport cost that cross an obstacle (default Inf)
- 'along_obstacle_delta_tau': rescaling parameter for transport cost that goes along nn obstacle (default 1)
- 'alpha_up_i': building cost parameter (scale) for roads that go up in elevation (default 0)
- 'beta_up_i': building cost parameter (elasticity) for roads that go up in elevation (default 1)
- 'alpha_up_tau': transport cost parameter (scale) for roads that go up in elevation (default 0)
- 'beta_up_tau': transport cost parameter (elasticity) for roads that go up in elevation (default 1)
- 'alpha_down_i': building cost parameter (scale) for roads that go down in elevation (default 0)
- 'beta_down_i': building cost parameter (elasticity) for roads that go down in elevation (default 1)
- 'alpha_down_tau': transport cost parameter (scale) for roads that go down in elevation (default 0)
- 'beta_down_tau': transport cost parameter (elasticity) for roads that go down in elevation (default 1)

"""
function apply_geography(graph, geography; kwargs...)

    options = Dict(
        :across_obstacle_delta_i => Inf,
        :along_obstacle_delta_i => 1,
        :across_obstacle_delta_tau => Inf,
        :along_obstacle_delta_tau => 1,
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
    geography = dict_to_namedtuple(geography)
    z, obstacles = geography.z, geography.obstacles

    # New graph object components
    delta_i_new = copy(graph.delta_i)
    delta_tau_new = copy(graph.delta_tau)
    adjacency_new = copy(graph.adjacency)
    nodes_new = deepcopy(graph.nodes)

    # Embed geographical barriers into network building costs
    for i in 1:graph.J
        for neighbor in graph.nodes[i]
            distance = sqrt((graph.x[i] - graph.x[neighbor])^2 + (graph.y[i] - graph.y[neighbor])^2)
            Delta_z = z[neighbor] - z[i]
            delta_i_new[i, neighbor] = distance * (1 + op.alpha_up_i * max(0, Delta_z)^op.beta_up_i + op.alpha_down_i * max(0, -Delta_z)^op.beta_down_i)
            delta_tau_new[i, neighbor] = distance * (1 + op.alpha_up_tau * max(0, Delta_z)^op.beta_up_tau + op.alpha_down_tau * max(0, -Delta_z)^op.beta_down_tau)
        end
    end

    # Remove edges where geographical barriers are (rivers)
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
                            end
                            rmj = findfirst(==(j), nodes_new[i])
                            if rmj !== nothing
                                deleteat!(nodes_new[i], rmj)
                            end
                            adjacency_new[i, j] = 0
                            adjacency_new[j, i] = 0
                            has_been_destroyed = true
                        else
                            # Or make it costly to cross: link could cross multiple obstacles, but we only allow one obstacle crossing to take effect
                            if !across_obstacle[i, j]
                                across_obstacle[i, j] = true
                                delta_i_new[i, j] *= op.across_obstacle_delta_i
                                delta_tau_new[i, j] *= op.across_obstacle_delta_tau
                            end
                            if !across_obstacle[j, i]
                                across_obstacle[j, i] = true
                                delta_i_new[j, i] *= op.across_obstacle_delta_i
                                delta_tau_new[j, i] *= op.across_obstacle_delta_tau
                            end
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
            deleteat!(nodes_new[io], findfirst(==(jo), nodes_new[io]))
            deleteat!(nodes_new[jo], findfirst(==(io), nodes_new[jo]))
            adjacency_new[io, jo] = 0
            adjacency_new[jo, io] = 0
        end
    else 
        for (io, jo) in zip(obstacles[:, 1], obstacles[:, 2])
            delta_i_new[io, jo] *= op.along_obstacle_delta_i
            delta_i_new[jo, io] *= op.along_obstacle_delta_i
            delta_tau_new[io, jo] *= op.along_obstacle_delta_tau
            delta_tau_new[jo, io] *= op.along_obstacle_delta_tau
            along_obstacle[io, jo] = true
            along_obstacle[jo, io] = true
            # This should be redundant if the above block works as intended
            # across_obstacle[io, jo] = false
            # across_obstacle[jo, io] = false
        end
    end

    # Creating new object
    graph_new = namedtuple_to_dict(graph)
    graph_new[:nodes] = nodes_new
    graph_new[:delta_i] = delta_i_new
    graph_new[:delta_tau] = delta_tau_new
    graph_new[:adjacency] = adjacency_new
    graph_new[:across_obstacle] = across_obstacle
    graph_new[:along_obstacle] = along_obstacle
    # make sure that the degrees of freedom of the updated graph match the # of links
    graph_new[:ndeg] = sum(tril(adjacency_new))

    return dict_to_namedtuple(graph_new)
end

