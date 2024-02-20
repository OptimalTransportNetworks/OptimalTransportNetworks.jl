
using Random

# ==============
# INITIALIZATION
# ==============

# Set parameters: try with LaborMobility: on/off, convex: beta>=gamma,
# nonconvex: gamma>betam, CrossGoodCongestion: on/off, or ADiGator: on/off

param = init_parameters("LaborMobility" => "on", "K" => 100, "gamma" => 2, "beta" => 1, "N" => 6, "TolKappa" => 1e-4,
                        "CrossGoodCongestion" => "off", "ADiGator" => "off")

# ------------
# Init network

param, graph = create_graph(param, 7, 7, "Type" => "triangle") # create a triangular network of 21x21

param.Zjn[:, 1:param.N-1] .= 0 # default locations cannot produce goods 1-10
param.Zjn[:, param.N] .= 1 # but they can all produce good 11 (agricultural)

# Draw the cities randomly
# Random.seed!(5) # reinit random number generator
for i in 1:param.N-1
    newdraw = false
    while newdraw == false
        j = round(Int, 1 + rand() * (graph.J - 1))
        if any(param.Zjn[j, 1:param.N-1] .> 0) == false # make sure node j does not produce any differentiated good
            newdraw = true
            param.Zjn[j, 1:param.N] .= 0
            param.Zjn[j, i] = 1
        end
    end
end


# ==========
# RESOLUTION
# ==========

@time res = optimal_network(param, graph)

# Plot them 

using Plots

rows = ceil((param.N + 1) / 4)
cols = min(4, param.N + 1)

plot(layout = (rows, cols))

plot!(subplot = 1)
results = res
plot_graph(param, graph, results.Ijk)
title!("(a) Network I")

for i in 1:param.N
    plot!(subplot = i + 1)
    results = res
    plot_graph(param, graph, results.Qjkn[:, :, i], arrows = true, arrowscale = 1.5)
    title!(sprintf("(%c) Flows good %i", 97 + i, i))
end


# Please note that the translation assumes that the functions `init_parameters`, `create_graph`, `optimal_network`, and `plot_graph` have been defined in Julia with the same functionality as in the original Matlab code. The translation also assumes that the `param` and `graph` objects have similar properties and that array indexing works in a similar way. If this is not the case, the code may need to be adjusted.