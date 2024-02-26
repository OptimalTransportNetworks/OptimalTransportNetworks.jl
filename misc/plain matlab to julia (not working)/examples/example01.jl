
using OptimalTransportNetworks

# ==============
# INITIALIZATION
# ==============

# Set parameters: try with LaborMobility: on/off, convex: beta>=gamma,
# nonconvex: gamma>beta, CrossGoodCongestion: on/off, or ADiGator: on/off

param = init_parameters(LaborMobility="on", K=10, gamma=1, beta=1,
                        N=1, TolKappa=1e-5, ADiGator=true, CrossGoodCongestion=false, nu=1)


# TolKappa is the tolerance in distance b/w iterations for road capacity
# kappa=I^gamma/delta_i, default is 1e-7 but we relax it a bit here

# ------------
# Init network

param, graph = create_graph(param, 11, 11, type = "map") # create a map network of 11x11 nodes located in [0,10]x[0,10]
# note: by default, productivity and population are equalized everywhere

# Customize graph
param[:Zjn] = fill(0.1, param[:J]) # set most places to low productivity
Ni = find_node(graph, 6, 6) # Find index of the central node at (6,6)
param[:Zjn][Ni] = 1 # central node more productive

# ==========
# RESOLUTION
# ==========

@time results = optimal_network(param, graph)

# Plot them 

sizes = 1.5 * (results["Cj"] .- minimum(results["Cj"])) ./ (maximum(results["Cj"]) - minimum(results["Cj"])) # size of each node
shades = (results["Cj"] .- minimum(results["Cj"])) ./ (maximum(results["Cj"]) - minimum(results["Cj"])) # shading for each node in [0,1] between NodeBgColor and NodeFgColor

plot_graph(param, graph, results["Ijk"], "Sizes" => sizes, "Shades" => shades,
    "NodeFgColor" => [1 .9 .4], "NodeBgColor" => [.8 .1 .0], "NodeOuterColor" => [0 .5 .6])

# the size/shade of the nodes reflects the total consumption at each node


# Please note that this translation assumes that the `OptimalTransportNetworks` package in Julia has similar functions and data structures as the MATLAB code. If this is not the case, the translation may need to be adjusted accordingly.