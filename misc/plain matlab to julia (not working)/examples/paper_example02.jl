
using Random

# 4.1.2 RANDOM CITIES
#
# Init and Solve network

width=9; height=9;
nb_cities=20;

param = init_parameters(TolKappa=1e-5, K=100, Annealing="off")
param, g = create_graph(param, width, height, Type="triangle") # case with random cities, one good

# set fundamentals

Random.seed!(5)
param.N = 1
param.Zjn = param.minpop * ones(g.J) # matrix of productivity
param.Hj = ones(g.J) # matrix of housing
param.Lj = zeros(g.J) # matrix of population

Ni = find_node(g, ceil(width/2), ceil(height/2)) # find center
param.Zjn[Ni] = 1 # more productive node
param.Lj[Ni] = 1 # more productive node

for i in 1:nb_cities-1
    newdraw = false
    while newdraw == false
        j = nearest(1 + rand() * (g.J - 1))
        if param.Lj[j] <= param.minpop
            newdraw = true
            param.Lj[j] = 1
        end
    end
end

param.hj = param.Hj ./ param.Lj
param.hj[param.Lj .== 0] = 1

# Convex case
results = Vector{Any}(undef, 3)
results[1] = optimal_network(param, g)

# Nonconvex - no annealing
param.gamma = 2
results[2] = optimal_network(param, g)

# Nonconvex - annealing
results[3] = annealing(param, g, results[2].Ijk, PerturbationMethod="rebranching")

welfare_increase = (results[3].welfare / results[2].welfare) ^ (1 / (param.alpha * (1 - param.rho))) # compute welfare increase in consumption equivalent


# plot
s = ["random_cities_convex", "random_cities_nonconvex", "random_cities_annealing"]
titles = [["Transport Network (I_{jk})", "Shipping (Q_{jk})"],
          ["Transport Network (I_{jk})", "Shipping (Q_{jk})"],
          ["Before annealing", "After annealing"]]

plots = [["results[i].Ijk", "results[i].Qjkn"],
         ["results[i].Ijk", "results[i].Qjkn"],
         ["results[2].Ijk", "results[3].Ijk"]]

texts = [["", ""],
         ["", ""],
         ["Welfare = 1", sprintf("Welfare = %1.3f (cons. eq.)", welfare_increase)]]

arrows = [["off", "on"],
          ["off", "on"],
          ["off", "off"]]

for i in 1:3
    fig = figure("Units" => "inches", "Position" => [0, 0, 7.5, 3], "Name" => s[i])

    subplot(1, 2, 1)
    plot_graph(param, g, eval(Meta.parse(plots[i][1])), Sizes=1.2 * param.Lj, Arrows=arrows[i][1])
    title(sprintf("(%c) %s", 97 + 2 * (i - 1), titles[i][1]), "FontWeight" => "normal", "FontName" => "Times", "FontSize" => 9)
    text(0.5, -0.05, texts[i][1], "HorizontalAlignment" => "center", "Fontsize" => 8)

    subplot(1, 2, 2)
    plot_graph(param, g, eval(Meta.parse(plots[i][2])), Sizes=1.2 * param.Lj, Arrows=arrows[i][2], ArrowScale=1)
    title(sprintf("(%c) %s", 97 + 2 * (i - 1) + 1, titles[i][2]), "FontWeight" => "normal", "FontName" => "Times", "FontSize" => 9)
    text(0.5, -0.05, texts[i][2], "HorizontalAlignment" => "center", "Fontsize" => 8)
end

# Please note that this translation assumes that the functions `init_parameters`, `create_graph`, `find_node`, `nearest`, `optimal_network`, `annealing`, `figure`, `subplot`, `plot_graph`, `title`, and `text` have been defined elsewhere in your Julia code. Also, the `sprintf` function from MATLAB has been replaced with the equivalent `@sprintf` macro in Julia.