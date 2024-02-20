# OptimalTransportNetworks.jl -- Optimal Transport Ntworks in Spatial Equilibrium - in Julia

module OptimalTransportNetworks

using LinearAlgebra

# Function to include all .jl files in a directory
function include_directory(directory)
    for file in readdir(directory)
        if endswith(file, ".jl")
            include(joinpath(directory, file))
        end
    end
end

# Include all .jl files in a specific directory
include_directory("objective/")
include_directory("constraints/")
include_directory("solve_allocation/")
include_directory("main/")

# Defining exports
export init_parameters, optimal_network, apply_geography, create_graph, plot_graph, create_auxdata, call_adigator
export find_node, remove_node, add_node

end