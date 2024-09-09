using OptimalTransportNetworks
using Random
using Plots

# Initialize parameters
param = init_parameters(K = 100, labor_mobility = false)
width, height = 13, 13

# Create graph
param, g0 = create_graph(param, width, height, type = "map")

# Set fundamentals
param[:Zjn] = ones(param[:J], 1) .* 1e-3  # matrix of productivity (not 0 to avoid numerical glitches)
param[:Lj] = ones(param[:J]) .* 1e-3  # matrix of population

Ni = find_node(g0, ceil(width/2), ceil(height/2)) # center
param[:Zjn][Ni] = 1 # more productive node
param[:Lj][Ni] = 1 # more productive node

Random.seed!(5)
nb_cities = 20 # draw a number of random cities in space
for i in 1:nb_cities-1
    newdraw = false
    while newdraw == false
        j = round(Int, 1 + rand() * (g0[:J] - 1))
        if param[:Lj][j] <= 1e-3
            newdraw = true
            param[:Lj][j] = 1
        end
    end
end

param[:hj] = param[:Hj] ./ param[:Lj]
param[:hj][param[:Lj] .== 1e-3] .= 1

# Draw geography
z = zeros(g0[:J]) # altitude of each node
geographies = Dict()
geographies[:blank] = (z = z, obstacles = nothing)

g = apply_geography(g0, geographies[:blank])

# Blank geography
results = Dict(:blank => optimal_network(param, g))

# Mountain
mountain_size = 0.75
mountain_height = 1
mount_x = 10
mount_y = 10
geographies[:mountain] = (z = mountain_height * exp.(-((g0[:x] .- mount_x).^2 .+ (g0[:y] .- mount_y).^2) / (2 * mountain_size^2)),
                          obstacles = nothing)
g = apply_geography(g0, geographies[:mountain], alpha_up_i = 10, alpha_down_i = 10)
results[:mountain] = optimal_network(param, g)

# Adding river and access by land
geographies[:river] = (z = geographies[:mountain].z, 
                       obstacles = [6 + (1-1)*width 6 + (2-1)*width;
                          6 + (2-1)*width 6 + (3-1)*width;
                          6 + (3-1)*width 7 + (4-1)*width;
                          7 + (4-1)*width 8 + (5-1)*width;
                          8 + (5-1)*width 9 + (5-1)*width;
                          11 + (5-1)*width 12 + (5-1)*width;
                          12 + (5-1)*width 13 + (5-1)*width])

g = apply_geography(g0, geographies[:river], 
                    along_abstacle_delta_i = 1, 
                    along_abstacle_delta_tau = 1, 
                    alpha_up_i = 10, alpha_down_i = 10)
results[:river] = optimal_network(param, g)

# Reinit and put another river and bridges
geographies[:bridges] = (z = mountain_height * exp.(-((g0[:x] .- mount_x).^2 .+ (g0[:y] .- mount_y).^2) / (2 * mountain_size^2)),
                         obstacles = [6 + (1-1)*width 6 + (2-1)*width;
                             6 + (2-1)*width 6 + (3-1)*width;
                             6 + (3-1)*width 7 + (4-1)*width;
                             7 + (4-1)*width 8 + (5-1)*width;
                             8 + (5-1)*width 9 + (5-1)*width;
                             9 + (5-1)*width 10 + (5-1)*width;
                             10 + (5-1)*width 11 + (5-1)*width;
                             11 + (5-1)*width 12 + (5-1)*width;
                             12 + (5-1)*width 13 + (5-1)*width])

g = apply_geography(g0, geographies[:bridges], alpha_up_i = 10, alpha_down_i = 10, 
                    across_abstacle_delta_i = 2, 
                    along_abstacle_delta_tau = 1)
results[:bridges] = optimal_network(param, g)

# Allowing for water transport
geographies[:water_transport] = (z = mountain_height * exp.(-((g0[:x] .- mount_x).^2 .+ (g0[:y] .- mount_y).^2) / (2 * mountain_size^2)),
                                 obstacles = [6 + (1-1)*width 6 + (2-1)*width;
                                              6 + (2-1)*width 6 + (3-1)*width;
                                              6 + (3-1)*width 7 + (4-1)*width;
                                              7 + (4-1)*width 8 + (5-1)*width;
                                              8 + (5-1)*width 9 + (5-1)*width;
                                              9 + (5-1)*width 10 + (5-1)*width;
                                              10 + (5-1)*width 11 + (5-1)*width;
                                              11 + (5-1)*width 12 + (5-1)*width;
                                              12 + (5-1)*width 13 + (5-1)*width])

g = apply_geography(g0, geographies[:water_transport], alpha_up_i = 10, alpha_down_i = 10, 
                    across_abstacle_delta_i = 2, 
                    along_obstacle_delta_i = 0.5,
                    along_abstacle_delta_tau = 1)
results[:water_transport] = optimal_network(param, g)

# Increasing returns to transport
param[:gamma] = 2
geographies[:increasing_returns] = geographies[:water_transport]
results[:increasing_returns] = optimal_network(param, g)

# Plot results
simulation = ["blank", "mountain", "river", "bridges", "water_transport", "increasing_returns"]
obstacles = ["off", "off", "on", "on", "on", "on"]
plots = Vector{Any}(undef, length(simulation)) 

i = 0
for s in simulation
    i += 1
    plots[i] = plot_graph(g, results[Symbol(s)][:Ijk], 
                          geography = geographies[Symbol(s)], obstacles = obstacles[i] == "on",
                          mesh = true, mesh_transparency = 0.2,
                          node_sizes = results[Symbol(s)][:cj] .* (param[:Lj] .> 1e-3), 
                          node_shades = param[:Zjn], 
                          edge_min_thickness = 1.5)
    title!(plots[i], "Geography $(s)")
end

# Combine plots
final_plot = plot(plots..., layout = (2, 3), size = (3*400, 2*400))
display(final_plot)
