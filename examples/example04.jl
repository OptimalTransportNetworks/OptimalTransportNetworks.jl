
using OptimalTransportNetworks
import Random

# ==============
# INITIALIZATION
# ==============

# Init and Solve network

param = init_parameters(K = 100, labor_mobility = false,
                        N = 1, gamma = 1, beta = 1, duality = true)
w = 13; h = 13
graph = create_graph(param, w, h, type = "map")

# ----------------
# Draw populations

graph[:Zjn] = ones(graph[:J], 1) .* 1e-6  # matrix of productivity (not 0 to avoid numerical glitches)
graph[:Lj] = ones(graph[:J]) .* 1e-6  # matrix of population

Ni = find_node(graph, ceil(w/2), ceil(h/2))  # center
graph[:Zjn][Ni] = 1  # more productive node
graph[:Lj][Ni] = 1  # more productive node

Random.seed!(5)
ncities = 20  # draw a number of random cities in space
for i in 1:ncities-1
    newdraw = false
    while newdraw == false
        j = round(Int, 1 + rand() * (graph[:J] - 1))
        if graph[:Lj][j] <= 1e-6
            newdraw = true
            graph[:Lj][j] = 1
        end
    end
end

graph[:hj] = graph[:Hj] ./ graph[:Lj]
graph[:hj][graph[:Lj] .== 1e-6] .= 1  # catch errors in places with infinite housing per capita


# --------------
# Draw geography

mountain_size = 0.75  # radius of mountain -- i.e. stdev of gaussian distribution
mountain_height = 1
mount_x = 10  # peak of mountain in (mount_x, mount_y)
mount_y = 10
z = mountain_height * exp.(-((graph[:x] .- mount_x).^2 + (graph[:y] .- mount_y).^2) / (2 * mountain_size^2))  # create a gaussian mountain 

# now introduce 'obstacles', i.e. river or trees, that is a list of edges
# which will have specific cost to cross or travel along....
obstacles = [6 + (1-1)*w 6 + (2-1)*w
             6 + (2-1)*w 6 + (3-1)*w
             6 + (3-1)*w 7 + (4-1)*w
             7 + (4-1)*w 8 + (5-1)*w
             8 + (5-1)*w 9 + (5-1)*w
             11 + (5-1)*w 12 + (5-1)*w
             12 + (5-1)*w 13 + (5-1)*w]  # Nobj x 2 matrix of (i,j) pairs of locations where a geographic barrier should be drawn

# create a geography structure, which is an input to the function
# 'apply_geography' and 'plot_graph' 
geography = (z = z, obstacles = obstacles)

# now apply geography to existing graph and recover the new graph
# we set the delta_i to infinite (no crossing possible) and some aversion
# to changes in elevation in building costs (symmetric up/down)
graph = apply_geography(graph, geography, alpha_up_i = 10, alpha_down_i = 10)

# Plot endowments          
plot_graph(graph, aspect_ratio = 3/4, 
            geography = geography, obstacles = true,
            mesh = true, mesh_transparency = 0.2, 
            node_sizes = graph[:Lj], node_shades = graph[:Zjn], 
            node_color = :seismic,
            edge_min_thickness = 1, edge_max_thickness = 4)

# =======================
# COMPUTE OPTIMAL NETWORK
# =======================

@time results = optimal_network(param, graph)

# ============
# PLOT RESULTS
# ============

sizes = 2 .* results[:cj] .* (graph[:Lj] .> 1e-6) / maximum(results[:cj])
shades = results[:cj] .* (graph[:Lj] .> 1e-6) / maximum(results[:cj])

plot_graph(graph, results[:Ijk], aspect_ratio = 3/4,
           geography = geography, obstacles = true,
           mesh = true, mesh_transparency = 0.2, 
           node_sizes = sizes, node_shades = shades, 
           edge_min_thickness = 1, edge_max_thickness = 4)

