# OptimalTransportNetworks.jl -- Optimal Transport Ntworks in Spatial Equilibrium - in Julia

module OptimalTransportNetworks

using LinearAlgebra, Statistics, Ipopt, JuMP

# Function to include all .jl files in a directory
function include_directory(directory)
    for file in readdir(directory)
        if endswith(file, ".jl")
            include(joinpath(directory, file))
        end
    end
end

# Include all .jl files in a specific directory
include_directory("src/objective")
include_directory("src/constraints")
include_directory("src/solve_allocation")
include_directory("src/main")

# Defining exports
export init_parameters, create_graph, plot_graph, create_auxdata, 
export optimal_network, annealing, solve_allocation, call_adigator
export add_node, find_node, remove_node, apply_geography

end