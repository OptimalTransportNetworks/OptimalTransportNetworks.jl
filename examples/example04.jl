
using Random

# ==============
# INITIALIZATION
# ==============

# Init and Solve network

param = init_parameters(TolKappa=1e-5, K=100, LaborMobility="off", N=1, gamma=1, beta=1)
w = 13; h = 13
param, graph = create_graph(param, w, h, Type="map")

# ----------------
# Draw populations

Random.seed!(5)
param.Zjn = ones(param.J, 1) .* 1e-3  # matrix of productivity (not 0 to avoid numerical glitches)
param.Lj = zeros(param.J, 1)  # matrix of population

Ni = find_node(graph, ceil(w/2), ceil(h/2))  # center
param.Zjn[Ni] = 1  # more productive node
param.Lj[Ni] = 1  # more productive node

ncities = 20  # draw a number of random cities in space
for i in 1:ncities-1
    newdraw = false
    while newdraw == false
        j = round(1 + rand() * (param.J - 1))
        if param.Lj[j] <= param.minpop
            newdraw = true
            param.Lj[j] = 1
        end
    end
end

param.hj = param.Hj ./ param.Lj
param.hj[param.Lj .== 0] = 1  # catch errors in places with infinite housing per capita

# --------------
# Draw geography

mountain_size = .75  # radius of mountain -- i.e. stdev of gaussian distribution
mountain_height = 1
mount_x = 10  # peak of mountain in (mount_x, mount_y)
mount_y = 10
z = mountain_height * exp.(-((graph.x .- mount_x).^2 + (graph.y .- mount_y).^2) / (2 * mountain_size^2))  # create a gaussian mountain 

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
geography = Dict("z" => z, "obstacles" => obstacles)

# now apply geography to existing graph and recover the new graph
# we set the delta_i to infinite (no crossing possible) and some aversion
# to changes in elevation in building costs (symmetric up/down)
graph = apply_geography(graph, geography, AcrossObstacleDelta_i=Inf, AlphaUp_i=10, AlphaDown_i=10)

# =======================
# COMPUTE OPTIMAL NETWORK
# =======================

results = optimal_network(param, graph)


# ============
# PLOT RESULTS
# ============

using Plots

fig = plot(size=(7.5, 5))

sizes = 2 .* results.cj .* (param.Lj .> param.minpop) / maximum(results.cj)
shades = results.cj .* (param.Lj .> param.minpop) / maximum(results.cj)
plot_graph(param, graph, results.Ijk, Geography="on", GeographyStruct=geography,
           Mesh="on", MeshTransparency=.2, Obstacles="on",
           NodeFgColor=[1 .9 .4], NodeBgColor=[.8 .1 .0], NodeOuterColor=[0 .5 .6],
           Sizes=sizes, Shades=shades, MinEdgeThickness=1, MaxEdgeThickness=4)


# Please note that this translation assumes that the functions `init_parameters`, `create_graph`, `find_node`, `apply_geography`, `optimal_network`, and `plot_graph` have been appropriately defined in Julia. Also, the plotting library Plots.jl is used for the plotting part.