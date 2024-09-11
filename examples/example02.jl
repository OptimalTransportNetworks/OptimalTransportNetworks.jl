
using OptimalTransportNetworks
import Random

# ==============
# INITIALIZATION
# ==============

param = init_parameters(labor_mobility = false, K = 100, gamma = 1, beta = 1, N = 1, 
                        annealing = false)

# ------------
# Init network

w, h = 13, 13
graph = create_graph(param, w, h, type = "triangle") # create a triangular network of 21x21

ncities = 20 # number of random cities
graph[:Lj] .= 0 # set population to minimum everywhere
graph[:Zjn] .= 0.01 # set low productivity everywhere

Ni = find_node(graph, ceil(Int, w/2), ceil(Int, h/2)) # Find the central node
graph[:Zjn][Ni] = 1 # central node is more productive
graph[:Lj][Ni] = 1 # cities are equally populated

# Draw the rest of the cities randomly
Random.seed!(5) # reinit random number generator
for i in 1:ncities
    newdraw = false
    while !newdraw
        j = round(Int, 1 + rand() * (graph[:J] - 1))
        if graph[:Lj][j] != 1 / (ncities + 1) # make sure node j is unpopulated           
            newdraw = true
            graph[:Lj][j] = 1
        end
    end
end

# For Ipopt: population cannot be zero!
graph[:Lj][graph[:Lj] .== 0] .= 1e-6

# ==========
# RESOLUTION
# ==========

@time begin
    # first, compute the optimal network in the convex case (beta>=gamma)
    res = Vector{Any}(undef, 3)
    res[1] = optimal_network(param, graph)
    # second, in the nonconvex case (gamma>beta)
    param[:gamma] = 2 # change only gamma, keep other parameters
    res[2] = optimal_network(param, graph) # optimize by iterating on FOCs
    res[3] = annealing(graph, res[2][:Ijk]) # improve with annealing, starting from previous result
end

# ==========
# PLOT
# ==========

using Plots

labels = ["Convex", "Nonconvex (FOC)", "Nonconvex (annealing)"]
plots = [] # Initialize an empty array to hold the subplots

for i in 1:3
    results = res[i]
    sizes = 4 * results[:Cj] / maximum(results[:Cj])
    shades = results[:Cj] / maximum(results[:Cj])
    p = plot_graph(graph, results[:Ijk], edge_max_thickness=4,
                   node_sizes=sizes, node_shades=shades)
    title!(p, labels[i])
    push!(plots, p) # Add the subplot to the array of plots
end

# Combine the plots into a single figure with a layout of 1 row and 3 columns
final_plot = plot(plots..., layout = (1, 3), size = (1200, 400))

# Display the final plot
display(final_plot)
