# OptimalTransportNetworks.jl -- Optimal Transport Ntworks in Spatial Equilibrium - in Julia

module OptimalTransportNetworks

using LinearAlgebra, Plots # Ipopt, JuMP,
using SparseArrays: sparse
import Ipopt, ForwardDiff # Try to replace with Enzyme.jl

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

# Include all .jl files in a specific directory
include_directory("objective")
include_directory("constraints")
include_directory("solve_allocation")
include_directory("main")

# Defining exports
export init_parameters, create_graph, plot_graph, create_auxdata
export optimal_network, annealing, solve_allocation, call_adigator
export add_node, find_node, remove_node, apply_geography

end

# Load using: 
#  using .OptimalTransportNetworks