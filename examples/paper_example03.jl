
using Random

# ------------------------------------
# 4.2 Random Cities with Multiple Goods
# -------------------------------------

ngoods=10
width=9 
height=9

# Init and Solve network

param = init_parameters(TolKappa=1e-5, K=10000, LaborMobility="on", Annealing="off")
param, g = create_graph(param, width, height, Type="triangle")

# set fundamentals

param.N = 1+ngoods
param.Zjn = zeros(g.J, param.N) # matrix of productivity

param.Zjn[:,1] .= 1 # the entire countryside produces the homogenenous good 1

if ngoods > 0
    Ni = find_node(g, 5, 5) # center
    param.Zjn[Ni, 2] = 1 # central node
    param.Zjn[Ni, 1] = 0 # central node
    
    Random.seed!(5)
    for i in 2:ngoods
        newdraw = false
        while newdraw == false
            j = round(Int, 1 + rand() * (g.J - 1))
            if param.Zjn[j, 1] > 0
                newdraw = true
                param.Zjn[j, i+1] = 1
                param.Zjn[j, 1] = 0
            end
        end
    end
end

# Convex case
results = Array{Any}(undef, 2)
results[1] = optimal_network(param, g)

# Nonconvex
param.gamma = 2
results[2] = optimal_network(param, g, results[1].Ijk)


# Plot results

cols = 3 # number of columns
rows = ceil((1 + param.N) / cols)

s = ["random_cities_multigoods_convex", "random_cities_multigoods_nonconvex"]

for j in 1:2
    fig = figure("Units" => "inches", "Position" => [0, 0, 7.5, 11], "Name" => s[j])

    # Plot network
    subplot(rows, cols, 1)
    plot_graph(param, g, results[j].Ijk, Shades = (results[j].Lj .- minimum(results[j].Lj)) ./ (maximum(results[j].Lj) - minimum(results[j].Lj)), Sizes = 1 .+ 16 .* (results[j].Lj ./ mean(results[j].Lj) .- 1), NodeFgColor = [.6, .8, 1], Transparency = "off")
    title("(a) Transport Network (I_{jk})", fontweight = "normal", fontname = "Times", fontsize = 9)

    for i in 1:param.N
        subplot(rows, cols, i+1)
        sizes = 3 .* results[j].Yjn[:, i] ./ sum(results[j].Yjn[:, i])
        shades = param.Zjn[:, i] ./ maximum(param.Zjn[:, i])
        plot_graph(param, g, results[j].Qjkn[:, :, i], Arrows = "on", ArrowScale = 1, ArrowStyle = "thin", Nodes = "on", Sizes = sizes, Shades = shades, NodeFgColor = [.6, .8, 1], Transparency = "off")
        title("($Char(97+i)) Shipping (Q^{i}_{jk})", fontweight = "normal", fontname = "Times", fontsize = 9)
    end

    # Save
    # savefig(fig, s[j] * ".eps")
    # savefig(fig, s[j] * ".jpg")
end


# Please note that this translation assumes that the functions `init_parameters`, `create_graph`, `find_node`, `optimal_network`, and `plot_graph` have been previously defined in Julia with the same functionality as in the original Matlab code. Also, the `figure` and `subplot` functions are assumed to be from a plotting package like `PyPlot`. The saving of the figures is commented out because the directory to save the figures is not specified.