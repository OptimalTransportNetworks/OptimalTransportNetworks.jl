# OptimalTransportNetworks.jl -- Optimal Transport Ntworks in Spatial Equilibrium - in Julia

# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4

# -----------------------------------------------------------------------------------
# REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
# Fajgelbaum and Edouard Schaal.

# Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
# pfajgelbaum@ucla.edu, eschaal@crei.cat
# -----------------------------------------------------------------------------------

"""
### Optimal Transport Networks in Spatial Equilibrium

Pure Julia implementation of the model and algorithms presented in:

Fajgelbaum, P. D., & Schaal, E. (2020). Optimal transport networks in spatial equilibrium. Econometrica, 88(4), 1411-1452.

The library is based on the JuMP modeling framework for mathematical optimization in Julia, and roughly follows the 
MATLAB OptimalTransportNetworkToolbox (v1.0.4b) provided by the authors. Compared to MATLAB, the Julia library presents
a simplified interface and only exports key functions, while retaining full flexibility. 

# Exported Functions

**Problem Setup**

`init_paramaters()`   - Create parameters dictionary\n
`create_graph()`      - Create network graph dictionary\n
`apply_geography()`   - (Optional) apply geographical features to alter the graph edge weights (network building and traversing costs)\n

**Compute Optimal Network + Refine Solution in Non-Convex Cases**

`optimal_network()`   - Compute optimal network given parameters and graph\n
`annealing()`         - Refine solution using simulated annealing in non-convex cases (automatically called in optimal_network() if param[:annealing] == true))\n

**Plot Graph (Incl. Network Solution)**

`plot_graph()`        - Plot network graph and optimal infrastructure levels\n

**Helper Functions to Manipulate Graphs**

`find_node()`         - Find index of node that is closest to a given pair of coordinates\n
`add_node()`          - Add new node to graph with given coordinates and connected neighbors\n
`remove_node()`       - Remove node from graph\n

# Examples
```julia
# Convex case
param = init_parameters()
param, graph = create_graph(param)
param[:Zjn][61] = 10.0
result = optimal_network(param, graph)
plot_graph(graph, result[:Ijk])


# Nonconvex case, disabling automatic annealing
param = init_parameters(annealing = false, gamma = 2)
param, graph = create_graph(param)
param[:Zjn][61] = 10.0
result = optimal_network(param, graph)

# Run annealing
results_annealing = annealing(param, graph, result[:Ijk])

# Comparison
plot_graph(graph, result[:Ijk])
plot_graph(graph, result_annealing[:Ijk])
```
"""
module OptimalTransportNetworks

using LinearAlgebra, JuMP, Plots
using SparseArrays: sparse
using Statistics: mean
using Dierckx: Spline2D, evaluate
import Ipopt, Plots, Random #, MathOptSymbolicAD
# import MathOptInterface as MOI

# Function to include all .jl files in a directory
function include_directory(directory)
    # Use the path of the current file to construct the directory path
    dir_path = joinpath(dirname(@__FILE__), directory)
    # Check if directory exists
    if isdir(dir_path)
        for file in readdir(dir_path)
            if endswith(file, ".jl")
                # Include the file using its full path
                include(joinpath(dir_path, file))
            end
        end
    else
        error("Directory does not exist: $dir_path")
    end
end

## Include all .jl files in a specific directory
include_directory("main")
include_directory("models")

# Defining exports
export init_parameters, create_graph, plot_graph
export optimal_network, annealing
export find_node, add_node, remove_node, apply_geography

end

# Load using: 
#  using .OptimalTransportNetworks