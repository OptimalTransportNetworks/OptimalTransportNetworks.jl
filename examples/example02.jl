
using OptimalTransportNetworks
import Random

# ==============
# INITIALIZATION
# ==============

param = init_parameters(labor_mobility =false, K = 100, gamma = 1, beta = 1, N = 1, 
                        annealing = false)

# ------------
# Init network

w, h = 13, 13
param, graph = create_graph(param, w, h, type = "triangle") # create a triangular network of 21x21

ncities = 20 # number of random cities
param[:Lj] .= 0 # set population to minimum everywhere
param[:Zjn] .= 0.01 # set low productivity everywhere

Ni = find_node(graph, ceil(Int, w/2), ceil(Int, h/2)) # Find the central node
param[:Zjn][Ni] = 1 # central node is more productive
param[:Lj][Ni] = 1 # cities are equally populated

# Draw the rest of the cities randomly
Random.seed!(5) # reinit random number generator
for i in 1:ncities
    newdraw = false
    while !newdraw
        j = round(Int, 1 + rand() * (graph.J - 1))
        if param[:Lj][j] != 1 / (ncities + 1) # make sure node j is unpopulated           
            newdraw = true
            param[:Lj][j] = 1
        end
    end
end


# ==========
# RESOLUTION
# ==========

@time begin
    # first, compute the optimal network in the convex case (beta>=gamma)
    res = Vector{Any}(undef, 3)
    res[1] = optimal_network(param, graph)

    # second, in the nonconvex case (gamma>beta)
    param = init_parameters(param = param, gamma = 2) # change only gamma, keep other parameters
    res[2] = optimal_network(param, graph) # optimize by iterating on FOCs
    res[3] = annealing(param, graph, res[2][:Ijk]) # improve with annealing, starting from previous result
end

# Plot them 

using Plots

labels = ["Convex", "Nonconvex (FOC)", "Nonconvex (annealing)"]

for i in 1:3
    results = res[i]
    sizes = 4 * results[:Cj] / maximum(results[:Cj])
    shades = results[:Cj] / maximum(results[:Cj])
    plot_graph(param, graph, results[:Ijk], node_sizes=sizes, node_shades=shades,
              edge_max_thickness=4)
    annotate!(0.5, 0.05, text(labels[i], :center))
end


