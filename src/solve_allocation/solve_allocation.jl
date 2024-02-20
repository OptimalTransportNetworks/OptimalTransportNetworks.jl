
"""
    solve_allocation(param, graph, I, verbose=false, x0=[], funcs=nothing)

This function is a wrapper that calls the relevant functions to solve for
the economic allocation given a certain matrix of infrastructure
investments in all cases (convex/nonconvex,mobile/immobile labor,cross-good
congestion or none, etc.)

Arguments:
- param: structure that contains the model's parameters
- graph: structure that contains the underlying graph (created by
create_graph function)
- I: JxJ symmetric matrix of infrastructure investments
- verbose: (optional) {true | false} tells IPOPT to display results or not
- x0: (optional) initial seed for the solver
- funcs: (optional) funcs structure returned by ADiGator to avoid calling
it avoid. Can be obtained with the function call_adigator().

Output:
- results: structure of results (Cj,Qjkn,etc.)
- flag: flag returned by IPOPT
- x: returns the 'x' variable returned by IPOPT (useful for warm start)
"""
function solve_allocation(param, graph, I, verbose=false, x0=[], funcs=nothing)
    param = dict_to_namedtuple(param)
    graph = dict_to_namedtuple(graph)
    auxdata = create_auxdata(param, graph, I)

    if funcs == nothing && param.adigator
        funcs = call_adigator(param, graph, I, false)
    end

    if param.adigator && param.mobility != 0.5
        if param.mobility && param.cong && !param.custom
            results, flag, x = solve_allocation_mobility_cgc_ADiGator(x0, auxdata, funcs, verbose)
        elseif !param.mobility && param.cong && !param.custom
            results, flag, x = solve_allocation_cgc_ADiGator(x0, auxdata, funcs, verbose)
        elseif param.mobility && !param.cong && !param.custom
            results, flag, x = solve_allocation_mobility_ADiGator(x0, auxdata, funcs, verbose)
        elseif !param.mobility && !param.cong && !param.custom && param.beta <= 1 && param.a < 1 && param.duality
            results, flag, x = solve_allocation_by_duality_ADiGator(x0, auxdata, funcs, verbose)
        elseif !param.mobility && !param.cong && !param.custom && (param.beta > 1 || param.a == 1)
            results, flag, x = solve_allocation_ADiGator(x0, auxdata, funcs, verbose)
        elseif param.custom
            results, flag, x = solve_allocation_custom_ADiGator(x0, auxdata, funcs, verbose)
        end
    else
        if !param.cong
            if param.mobility == 0
                if param.beta <= 1 && param.duality
                    results, flag, x = solve_allocation_by_duality(x0, auxdata, verbose)
                else
                    results, flag, x = solve_allocation_primal(x0, auxdata, verbose)
                end
            elseif param.mobility == 1
                results, flag, x = solve_allocation_mobility(x0, auxdata, verbose)
            elseif param.mobility == 0.5
                results, flag, x = solve_allocation_partial_mobility(x0, auxdata, verbose)
            end
        else
            if param.mobility == 0
                results, flag, x = solve_allocation_cgc(x0, auxdata, verbose)
            elseif param.mobility == 1
                results, flag, x = solve_allocation_mobility_cgc(x0, auxdata, verbose)
            elseif param.mobility == 0.5
                results, flag, x = solve_allocation_partial_mobility_cgc(x0, auxdata, verbose)
            end
        end
    end

    return results, flag, x
end


# Please note that this translation assumes that the functions `create_auxdata`, `call_adigator`, `solve_allocation_mobility_cgc_ADiGator`, `solve_allocation_cgc_ADiGator`, `solve_allocation_mobility_ADiGator`, `solve_allocation_by_duality_ADiGator`, `solve_allocation_ADiGator`, `solve_allocation_custom_ADiGator`, `solve_allocation_by_duality`, `solve_allocation_primal`, `solve_allocation_mobility`, `solve_allocation_partial_mobility`, `solve_allocation_cgc`, `solve_allocation_mobility_cgc`, and `solve_allocation_partial_mobility_cgc` have been appropriately translated to Julia and are available in the scope where `solve_allocation` is defined.