using OptimalTransportNetworks
using Random

# 4.1.2 RANDOM CITIES

# Init and Solve network

width = 9 
height = 9
nb_cities = 20

param = init_parameters(K = 100, annealing = false)
param, g = create_graph(param, width, height, type = "triangle") # case with random cities, one good

# set fundamentals

Random.seed!(5)
minpop = minimum(param[:Lj])
param[:Zjn] = minpop .* ones(g[:J])  # matrix of productivity

Ni = find_node(g, ceil(width/2), ceil(height/2)) # find center
param[:Zjn][Ni] = 1 # more productive node
param[:Lj][Ni] = 1  # more productive node

for i in 1:nb_cities-1
    newdraw = false
    while newdraw == false
        j = round(Int, 1 + rand() * (g[:J] - 1))
        if param[:Lj][j] <= minpop
            newdraw = true
            param[:Lj][j] = 1
        end
    end
end


# Convex case
results = Vector{Any}(undef, 3)
results[1] = optimal_network(param, g)

# Nonconvex - no annealing
param[:gamma] = 2
results[2] = optimal_network(param, g)

# Nonconvex - annealing
results[3] = annealing(param, g, results[2][:Ijk], perturbation_method = "rebranching")

welfare_increase = (results[3][:welfare] / results[2][:welfare]) ^ (1 / (param[:alpha] * (1 - param[:rho]))) # compute welfare increase in consumption equivalent


# Plot results

plots = Vector{Any}(undef, 6) # Initialize an empty array to hold the subplots
shades = sizes = results[i][:Cj] / maximum(results[i][:Cj])

plots[1] = plot_graph(g, results[1][:Ijk], node_sizes = sizes, node_shades = shades, node_sizes_scale = 40)
title!(plots[1], "Convex Network (I_{jk})")
plots[2] = plot_graph(g, results[1][:Qjkn][:, :, 1], edge_color = :brown, arrows = true, node_sizes = sizes, node_shades = shades, node_sizes_scale = 40)
title!(plots[2], "Convex Shipping (Q_{jk})")

plots[3] = plot_graph(g, results[2][:Ijk], node_sizes = sizes, node_shades = shades, node_sizes_scale = 40)
title!(plots[3], "Nonconvex Network (I_{jk})")
plots[4] = plot_graph(g, results[2][:Qjkn][:, :, 1], edge_color = :brown, arrows = true, node_sizes = sizes, node_shades = shades, node_sizes_scale = 40)
title!(plots[4], "Nonconvex Shipping (Q_{jk})")

plots[5] = plots[3]
plots[6] = plot_graph(g, results[3][:Ijk], node_sizes = sizes, node_shades = shades, node_sizes_scale = 40)
title!(plots[6], "Nonconvex Network (I_{jk})")
annotate!(plots[6], [(0.5, 1.04, text("With Annealing. Welfare increase: $(round((welfare_increase-1)*100, digits = 2))%", :black, :center, 10))])

# Combine the plots into a single figure with a layout of 1 row and 3 columns
final_plot = plot(plots..., layout = (3, 2), size = (800, 1200))

# Display the final plot
display(final_plot)


