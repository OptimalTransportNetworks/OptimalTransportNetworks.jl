
using OptimalTransportNetworks
import Random

# ==============
# INITIALIZATION
# ==============

# Set parameters: try with labor_mobility: true/false, convex: beta>=gamma,
# nonconvex: gamma>beta, cross_good_congestion: true/false

param = init_parameters(labor_mobility = true, K = 100, gamma = 2, beta = 1, N = 6, 
                        tol = 1e-4, cross_good_congestion = false)

# ------------
# Init network

param, graph = create_graph(param, 7, 7, type = "triangle") # create a triangular network of 21x21

param[:Zjn][:, 1:param[:N]-1] .= 0 # default locations cannot produce goods 1-10
param[:Zjn][:, param[:N]] .= 1 # but they can all produce good 11 (agricultural)

# Draw the cities randomly
Random.seed!(5) # reinit random number generator
for i in 1:param[:N]-1
    newdraw = false
    while newdraw == false
        j = round(Int, 1 + rand() * (graph.J - 1))
        if any(param[:Zjn][j, 1:param[:N]-1] .> 0) == false # make sure node j does not produce any differentiated good
            newdraw = true
            param[:Zjn][j, 1:param[:N]] .= 0
            param[:Zjn][j, i] = 1
        end
    end
end


# ==========
# RESOLUTION
# ==========

@time res = optimal_network(param, graph)

# ==========
# PLOT
# ==========
using Plots

rows = ceil(Int, (param[:N] + 1) / 4)
cols = min(4, param[:N] + 1)

plots = [] # Initialize an empty array to hold the subplots

# First subplot for the network Ijk
results = res
p1 = plot_graph(graph, results[:Ijk], nodes = false) 
title!(p1, "(a) Network I")
push!(plots, p1)

# Subsequent subplots for the flows of each good
for i in 1:param[:N]
    results = res
    p = plot_graph(graph, results[:Qjkn][:, :, i], arrows = true, nodes = false)
    title!(p, string('(', Char(96 + i + 1), ')', " Flows good ", i))
    push!(plots, p)
end

# Combine all the plots into a single figure with the specified layout
final_plot = plot(plots..., layout = (rows, cols), size = (1200, 400))

# Display the final plot
display(final_plot)

