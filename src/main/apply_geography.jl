
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

Notes:
- The function is partially mutating the graph structure, but also adding new fields to the graph structure.
"""
function apply_geography!(graph, geography; kwargs...)

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

    options = dict_to_namedtuple(options)
    geography = dict_to_namedtuple(geography)

    # Embed geographical barriers into network building costs
    for i in 1:graph.J
        for neighbor in graph.nodes[i]
            distance = sqrt((graph.x[i] - graph.x[neighbor])^2 + (graph.y[i] - graph.y[neighbor])^2)
            Delta_z = geography.z[neighbor] - geography.z[i]
            graph.delta_i[i, neighbor] = distance * (1 + options.alpha_up_i * max(0, Delta_z)^options.beta_up_i + options.alpha_down_i * max(0, -Delta_z)^options.beta_down_i)
            graph.delta_tau[i, neighbor] = distance * (1 + options.alpha_up_tau * max(0, Delta_z)^options.beta_up_tau + options.alpha_down_tau * max(0, -Delta_z)^options.beta_down_tau)
        end
    end


    # Remove edges where geographical barriers are (rivers)
    sz = size(geography.obstacles)[1]
    across_obstacle = zeros(graph.J, graph.J)
    along_obstacle = zeros(graph.J, graph.J)

    # Store initial delta matrics (avoid double counting)
    delta_i = copy(graph.delta_i)
    delta_tau = copy(graph.delta_tau)

    for i in 1:graph.J
        neighbors = graph.nodes[i]
        for j in neighbors
            # check if edge (i,j) intersects any of the obstacles
            k = 1
            has_been_destroyed = false
            while !has_been_destroyed && k <= sz
                if geography.obstacles[k, :] != [i, j] # if the link is not the obstacle itself
                    Z2_obj = [graph.x[geography.obstacles[k, 2]], graph.y[geography.obstacles[k, 2]]]
                    Z1_obj = [graph.x[geography.obstacles[k, 1]], graph.y[geography.obstacles[k, 1]]]
                    Z2 = [graph.x[j], graph.y[j]]
                    Z1 = [graph.x[i], graph.y[i]]

                    vec = Z2_obj - Z1_obj
                    normvec = [-vec[2], vec[1]]
                    val1 = (normvec' * (Z1 - Z1_obj)) * (normvec' * (Z2 - Z1_obj))

                    vec = Z2 - Z1
                    normvec = [-vec[2], vec[1]]
                    val2 = (normvec' * (Z1_obj - Z1)) * (normvec' * (Z2_obj - Z1))

                    if (val1 <= 0 && val2 <= 0) && normvec' * (Z2_obj - Z1_obj) != 0 # if the two edges intersect and are not colinears
                        if isinf(options.across_obstacle_delta_i) || isinf(options.across_obstacle_delta_tau) # remove the edge
                            deleteat!(graph.nodes[j], findfirst(==(i), graph.nodes[j]))
                            graph.adjacency[i, j] = 0
                            graph.adjacency[j, i] = 0
                            deleteat!(graph.nodes[i], findfirst(==(j), graph.nodes[i]))
                            has_been_destroyed = true
                        else
                            # or make it costly to cross
                            graph.delta_i[i, j] = options.across_obstacle_delta_i * delta_i[i, j]
                            graph.delta_i[j, i] = options.across_obstacle_delta_i * delta_i[j, i]
                            graph.delta_tau[i, j] = options.across_obstacle_delta_tau * delta_tau[i, j]
                            graph.delta_tau[j, i] = options.across_obstacle_delta_tau * delta_tau[j, i]
                            across_obstacle[i, j] = true
                            across_obstacle[j, i] = true
                        end
                    end
                end
                k += 1
            end
        end
    end

    # allow for different delta while traveling along obstacles (e.g., water transport)
    for i in 1:sz
        if !isinf(options.along_obstacle_delta_i) && !isinf(options.along_obstacle_delta_tau)
            graph.delta_i[geography.obstacles[i, 1], geography.obstacles[i, 2]] = options.along_obstacle_delta_i * graph.delta_i[geography.obstacles[i, 1], geography.obstacles[i, 2]]
            graph.delta_i[geography.obstacles[i, 2], geography.obstacles[i, 1]] = options.along_obstacle_delta_i * graph.delta_i[geography.obstacles[i, 2], geography.obstacles[i, 1]]
            graph.delta_tau[geography.obstacles[i, 1], geography.obstacles[i, 2]] = options.along_obstacle_delta_tau * graph.delta_tau[geography.obstacles[i, 1], geography.obstacles[i, 2]]
            graph.delta_i[geography.obstacles[i, 2], geography.obstacles[i, 1]] = options.along_obstacle_delta_tau * graph.delta_tau[geography.obstacles[i, 2], geography.obstacles[i, 1]]
            along_obstacle[geography.obstacles[i, 1], geography.obstacles[i, 2]] = true
            along_obstacle[geography.obstacles[i, 2], geography.obstacles[i, 1]] = true
            across_obstacle[geography.obstacles[i, 1], geography.obstacles[i, 2]] = false
            across_obstacle[geography.obstacles[i, 2], geography.obstacles[i, 1]] = false
        else # if infinite, remove edge
            deleteat!(graph.nodes[geography.obstacles[i, 1]], findfirst(==(geography.obstacles[i, 2]), graph.nodes[geography.obstacles[i, 1]]))
            deleteat!(graph.nodes[geography.obstacles[i, 2]], findfirst(==(geography.obstacles[i, 1]), graph.nodes[geography.obstacles[i, 2]]))
            graph.adjacency[geography.obstacles[i, 1], geography.obstacles[i, 2]] = 0
            graph.adjacency[geography.obstacles[i, 2], geography.obstacles[i, 1]] = 0
        end
    end

    graph_new = namedtuple_to_dict(graph)
    graph_new[:across_obstacle] = across_obstacle
    graph_new[:along_obstacle] = along_obstacle

    # make sure that the degrees of freedom of the updated graph match the # of links
    graph_new[:ndeg] = sum(tril(graph.adjacency))

    return dict_to_namedtuple(graph_new)
end

