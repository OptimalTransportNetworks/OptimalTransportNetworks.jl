
using Random
using LinearAlgebra
using Plots

# Initialize parameters
param = init_parameters(TolKappa=1e-5, K=100, LaborMobility=false)
width, height = 13, 13

# Create graph
param, g = create_graph(param, width, height, Type="map")

# Set fundamentals
Random.seed!(5)
param.N = 1
param.Zjn = fill(param.minpop, g.J) # matrix of productivity
param.Hj = ones(g.J) # matrix of housing
param.Lj = zeros(g.J) # matrix of population

Ni = find_node(g, 5, 5) # center
param.Zjn[Ni] = 1 # more productive node
param.Lj[Ni] = 1 # more productive node

nb_cities = 20 # draw a number of random cities in space
for i in 1:nb_cities-1
    newdraw = false
    while newdraw == false
        j = round(Int, 1 + rand() * (g.J - 1))
        if param.Lj[j] <= param.minpop
            newdraw = true
            param.Lj[j] = 1
        end
    end
end

param.hj = param.Hj ./ param.Lj
param.hj[param.Lj .== 0] = 1

# Draw geography
z = zeros(g.J) # altitude of each node
obstacles = []

geography = Dict("z" => z, "obstacles" => obstacles)
g = apply_geography(g, geography)

param0 = param # store initial params
g0 = g # store initial graph

# Blank geography
results = Dict(1 => optimal_network(param, g))

# Mountain
mountain_size = 0.75
mountain_height = 1
mount_x = 10
mount_y = 10
geography[2] = geography[1]
geography[2]["z"] = mountain_height * exp.(-((g.x .- mount_x).^2 .+ (g.y .- mount_y).^2) / (2 * mountain_size^2))

g = apply_geography(g, geography[2], AlphaUp_i=10, AlphaDown_i=10)
results[2] = optimal_network(param, g)

# Adding river and access by land
g = g0
geography[3] = geography[2]
geography[3]["obstacles"] = [6 + (1-1)*width 6 + (2-1)*width;
                             6 + (2-1)*width 6 + (3-1)*width;
                             6 + (3-1)*width 7 + (4-1)*width;
                             7 + (4-1)*width 8 + (5-1)*width;
                             8 + (5-1)*width 9 + (5-1)*width;
                             11 + (5-1)*width 12 + (5-1)*width;
                             12 + (5-1)*width 13 + (5-1)*width]

g = apply_geography(g, geography[3], AcrossObstacleDelta_i=Inf, AlphaUp_i=10, AlphaDown_i=10)
results[3] = optimal_network(param, g)

# Reinit and put another river and bridges
g = g0
geography[4] = geography[1]
geography[4]["z"] = mountain_height * exp.(-((g.x .- mount_x).^2 .+ (g.y .- mount_y).^2) / (2 * mountain_size^2))

geography[4]["obstacles"] = [6 + (1-1)*width 6 + (2-1)*width;
                             6 + (2-1)*width 6 + (3-1)*width;
                             6 + (3-1)*width 7 + (4-1)*width;
                             7 + (4-1)*width 8 + (5-1)*width;
                             8 + (5-1)*width 9 + (5-1)*width;
                             9 + (5-1)*width 10 + (5-1)*width;
                             10 + (5-1)*width 11 + (5-1)*width;
                             11 + (5-1)*width 12 + (5-1)*width;
                             12 + (5-1)*width 13 + (5-1)*width]

g = apply_geography(g, geography[4], AlphaUp_i=10, AlphaDown_i=10, AcrossObstacleDelta_i=2, AlongObstacleDelta_i=Inf)
results[4] = optimal_network(param, g)

# Allowing for water transport
g = g0
geography[5] = geography[1]
geography[5]["z"] = mountain_height * exp.(-((g.x .- mount_x).^2 .+ (g.y .- mount_y).^2) / (2 * mountain_size^2))
geography[5]["obstacles"] = [6 + (1-1)*width 6 + (2-1)*width;
                             6 + (2-1)*width 6 + (3-1)*width;
                             6 + (3-1)*width 7 + (4-1)*width;
                             7 + (4-1)*width 8 + (5-1)*width;
                             8 + (5-1)*width 9 + (5-1)*width;
                             9 + (5-1)*width 10 + (5-1)*width;
                             10 + (5-1)*width 11 + (5-1)*width;
                             11 + (5-1)*width 12 + (5-1)*width;
                             12 + (5-1)*width 13 + (5-1)*width]

g = apply_geography(g, geography[5], AlphaUp_i=10, AlphaDown_i=10, AcrossObstacleDelta_i=2, AlongObstacleDelta_i=0.5)
results[5] = optimal_network(param, g)

# Increasing returns to transport
param.gamma = 2
geography[6] = geography[5]
results[6] = optimal_network(param, g)

# Plot results
s = ["geography_blank", "geography_mountain", "geography_river", "geography_bridges", "geography_water_transport", "geography_increasing_returns"]
obstacles = ["off", "off", "on", "on", "on", "on"]

for j in 1:length(s)
    fig = plot_graph(param, g, results[j].Ijk, Geography=true, GeographyStruct=geography[j], Sizes=3*results[j].cj .* (param.Lj .> param.minpop) / maximum(results[j].cj), Mesh=true, MeshTransparency=0.2, Obstacles=obstacles[j], Shades=param.Zjn, MinEdgeThickness=1.5)
    savefig(fig, "$(s[j]).png")
end

# Please note that this translation assumes that the functions `init_parameters`, `create_graph`, `find_node`, `apply_geography`, `optimal_network`, and `plot_graph` have been previously defined in Julia with the same functionality as in the original Matlab code.