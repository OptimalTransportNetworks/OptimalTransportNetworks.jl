
using OptimalTransportNetworks
using Random
using Plots

# ------------------------------------
# 4.2 Random Cities with Multiple Goods
# -------------------------------------

ngoods = 10
width = 9  
height = 9

# Init and Solve network

param = init_parameters(K = 10000, labor_mobility = true, annealing = false)
param, g = create_graph(param, width, height, type = "triangle")

# set fundamentals

param[:N] = 1 + ngoods
param[:Zjn] = zeros(g[:J], param[:N]) # matrix of productivity

param[:Zjn][:,1] .= 1 # the entire countryside produces the homogenenous good 1

if ngoods > 0
    Ni = find_node(g, 5, 5) # center
    param[:Zjn][Ni, 2] = 1 # central node
    param[:Zjn][Ni, 1] = 0 # central node
    
    Random.seed!(5)
    for i in 2:ngoods
        newdraw = false
        while newdraw == false
            j = round(Int, 1 + rand() * (g[:J] - 1))
            if param[:Zjn][j, 1] > 0
                newdraw = true
                param[:Zjn][j, i+1] = 1
                param[:Zjn][j, 1] = 0
            end
        end
    end
end

# Convex case
results = Array{Any}(undef, 2)
results[1] = optimal_network(param, g)

# Nonconvex
param[:gamma] = 2
results[2] = optimal_network(param, g, I0 = results[1][:Ijk])

# Plot results

cols = 3 # number of columns
rows = Int(ceil((1 + param[:N]) / cols))

for j in 1:2
    # Initialize an empty array to hold the subplots
    plots = Vector{Any}(undef, (1 + param[:N])) 
    # Plot network
    plots[1] = plot_graph(g, results[j][:Ijk], node_shades = results[j][:Lj], node_sizes = results[j][:Lj], node_sizes_scale = 40)
    title!(plots[1], "(a) Transport Network")
    # Plot goods flows
    for i in 1:param[:N]
        plots[i+1] = plot_graph(g, results[j][:Qjkn][:, :, i], edge_color = :brown, arrows = true, arrow_style = "thin", 
                                node_sizes = results[j][:Yjn][:, i], node_sizes_scale = 40,
                                node_shades = param[:Zjn][:, i])
        title!(plots[i+1], string('(', Char(96 + i + 1), ')', " Flows Good ", i))
    end
    # Combine plots
    final_plot = plot(plots..., layout = (cols, rows), size = (rows*400, cols*400))
    display(final_plot)
end
