
"""
call_adigator(param, graph, I, verbose)

Call package ADiGator to produce the gradient, jacobian and hessian
to be used by IPOPT

Arguments:
- param: structure created by create_parameters() that contains all the model's parameters
- graph: structure created by create_graph() that contains all the graph's parameters
- I: structure created by create_I() that contains all the I's parameters
- verbose: switch whether or not the function should display output

Output:
- funcs: funcs structure used as input in IPOPT (returns [] if error)
"""
function call_adigator(param, graph, I, verbose=false)

    funcs = nothing

    # Check parameters
    if !param[:adigator]
        @warn "Please enable ADiGator with init_parameters() before."
        return funcs
    end

    # Call ADiGator
    auxdata = create_auxdata(param, graph, I)

    if param[:mobility] && param[:cong] && !param[:custom]
        setup = Dict(
            :numvar => 1 + graph.J * param[:N] + 2 * graph.ndeg * param[:N] + graph.J + graph.J + graph.J * param[:N],
            :objective => "objective_mobility_cgc",
            :auxdata => auxdata,
            :order => 2,
            :constraint => "constraints_mobility_cgc"
        )
    elseif !param[:mobility] && param[:cong] && !param[:custom]
        setup = Dict(
            :numvar => graph.J * param[:N] + 2 * graph.ndeg * param[:N] + graph.J + graph.J * param[:N],
            :objective => "objective_cgc",
            :auxdata => auxdata,
            :order => 2,
            :constraint => "constraints_cgc"
        )
    elseif param[:mobility] && !param[:cong] && !param[:custom]
        setup = Dict(
            :numvar => 1 + graph.J * param[:N] + graph.ndeg * param[:N] + graph.J + graph.J * param[:N],
            :objective => "objective_mobility",
            :auxdata => auxdata,
            :order => 2,
            :constraint => "constraints_mobility"
        )
    elseif (!param[:mobility] && !param[:cong] && !param[:custom]) && (param[:beta] <= 1 && param[:a] < 1)
        setup = Dict(
            :numvar => graph.J * param[:N],
            :objective => "objective_duality",
            :auxdata => auxdata,
            :order => 2
        )
    elseif (!param[:mobility] && !param[:cong] && !param[:custom]) && (param[:beta] > 1 || param[:a] == 1)
        setup = Dict(
            :numvar => graph.J * param[:N] + graph.ndeg * param[:N] + graph.J * param[:N],
            :objective => "objective",
            :auxdata => auxdata,
            :order => 2,
            :constraint => "constraints"
        )
    elseif param[:custom]
        setup = Dict(
            :numvar => size(x0, 1),
            :objective => "objective_custom",
            :auxdata => auxdata,
            :order => 2,
            :constraint => "constraints_custom"
        )
    end

    if verbose
        println("\n-------------------")
        println("CALLING ADIGATOR...\n")
    end

    setup[:options] = adigatorOptions("ECHO" => verbose, "OVERWRITE" => 1)

    funcs = adigatorGenFiles4Ipopt(setup)

    return funcs
end


# Please note that this translation assumes that the `adigatorOptions` and `adigatorGenFiles4Ipopt` functions have been appropriately translated to Julia and are available in the current scope. Also, the `create_auxdata` function should be translated and available in the current scope.