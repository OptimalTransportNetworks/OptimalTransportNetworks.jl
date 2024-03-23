
using OptimalTransportNetworks

# ==============
# INITIALIZATION
# ==============

# Set parameters: try with labor mobility: true/false, convex: beta>=gamma,
# nonconvex: gamma>beta, cross good congestion: true/false

param = init_parameters(labor_mobility = true, K = 10, gamma = 1, beta = 1, verbose = true,
                        N = 1, kappa_tol = 1e-5, cross_good_congestion = false, nu = 1)

# kappa_tol is the tolerance in distance b/w iterations for road capacity
# kappa=I^gamma/delta_i, default is 1e-7 but we relax it a bit here

# ------------
# Init network

param, graph = create_graph(param, 11, 11, type = "map") # create a map network of 11x11 nodes located in [0,10]x[0,10]
# note: by default, productivity and population are equalized everywhere

# Customize graph
param[:Zjn] = fill(0.1, param[:J], param[:N]) # set most places to low productivity
Ni = find_node(graph, 6, 6) # Find index of the central node at (6,6)
param[:Zjn][Ni, :] .= 1 # central node more productive

# ==========
# RESOLUTION
# ==========

@time results = optimal_network(param, graph) # solve the network


# Plot the network with the optimal transportation plan

plot_graph(graph, results[:Ijk], node_sizes = results[:Cj])

# The size/shade of the nodes reflects the total consumption at each node
