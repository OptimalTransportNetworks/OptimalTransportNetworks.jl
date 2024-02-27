
"""
 ==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

optimal_network.m: solve for the optimal network by solving the inner problem (dual if no mobility and
no cross good congestion, primal otherwise) and the outer problem by iterating over the FOCs

Arguments:
- param: structure that contains the model's parameters
- graph: structure that contains the underlying graph (created by create_graph function)
- I0: (optional) provides the initial guess for the iterations (matrix JxJ)
- Il: (optional) exogenous lower bound on infrastructure levels (matrix JxJ)
- Iu: (optional) exogenous upper bound on infrastructure levels (matrix JxJ)
- verbose: (optional) tell IPOPT to display results
- x0: (optional) provide initial condition for IPOPT

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
"""


# using LinearAlgebra
# I0=nothing; Il=nothing; Iu=nothing; verbose=false; x0=nothing

function optimal_network(param, graph, I0=nothing, Il=nothing, Iu=nothing, verbose=false, x0=nothing)

    J = graph.J
    TOL_I_BOUNDS = 1e-7
    error_status = false

    # ---------------------------------------------
    # CHECK PARAMETERS FOR ERRORS OR MISSING VALUES

    if param[:a] > 1
        error("Model with increasing returns to scale are not convex.")
    end

    if param[:nu] < 1 && param[:cong]
        error("nu has to be larger or equal to one for the problem to be guaranteed convex.")
    end

    if I0 === nothing
        I0 = zeros(J, J)
        for i in 1:J
            for j in graph.nodes[i]
                I0[i, j] = 1
            end
        end
        I0 = param[:K] * I0 / sum(graph.delta_i .* I0)
    end

    Il = Il === nothing ? zeros(graph.J, graph.J) : Il
    Iu = Iu === nothing ? Inf * ones(graph.J, graph.J) : Iu
    x0 = x0 === nothing ? [] : x0

    if param[:mobility] || param[:beta] > 1 || param[:cong]
        Il = max.(1e-6 * graph.adjacency, Il)
    end

    # --------------
    # INITIALIZATION

    optimizer = get(param, :optimizer, Ipopt.Optimizer)
    auxdata = create_auxdata(param, graph, I0)

    # if param[:adigator] && param[:mobility] != 0.5
    if param[:custom]
        model = param[:custom_model](optimizer, auxdata)
    elseif param[:mobility] == 1 && param[:cong]
        model = model_mobility_cgc(optimizer, auxdata)
    elseif param[:mobility] == 0 && param[:cong]
        model = model_fixed_cgc(optimizer, auxdata)
    elseif param[:mobility] == 1 && !param[:cong]
        model = model_mobility(optimizer, auxdata)
        recover_allocation = recover_allocation_mobility
    elseif (param[:mobility] == 0 && !param[:cong]) && (param[:beta] <= 1 && param[:a] < 1 && param[:duality])
        # TODO: Dual solution !!
        solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_by_duality_ADiGator(x0, auxdata, funcs, verbose)
    else 
        model = model_fixed(optimizer, auxdata)
    end
    # else
    #     if !param[:cong]
    #         if param[:mobility] == 0
    #             if param[:beta] <= 1 && param[:duality]
    #                 solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_by_duality(x0, auxdata, verbose)
    #             else
    #                 solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_primal(x0, auxdata, verbose)
    #             end
    #         elseif param[:mobility] == 1
    #             solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_mobility(x0, auxdata, verbose)
    #         elseif param[:mobility] == 0.5
    #             solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_partial_mobility(x0, auxdata, verbose)
    #         end
    #     else
    #         if param[:mobility] == 0
    #             solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_cgc(x0, auxdata, verbose)
    #         elseif param[:mobility] == 1
    #             solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_mobility_cgc(x0, auxdata, verbose)
    #         elseif param[:mobility] == 0.5
    #             solve_allocation_handle = (x0, auxdata, funcs, verbose) -> solve_allocation_partial_mobility_cgc(x0, auxdata, verbose)
    #         end
    #     end
    # end
    if haskey(param, :optimizer_attr)
        for (key, value) in param[:optimizer_attr]
            set_optimizer_attribute(model, String(key), value)
        end
    else
        # set_optimizer_attribute(model, "tol", 1e-5)  
        set_optimizer_attribute(model, "max_iter", 10000) 
    end
    # set_attribute(model,
    #     MOI.AutomaticDifferentiationBackend(),
    #     MathOptSymbolicAD.DefaultBackend(),
    # )
    # if haskey(param, :model_attr) ...

    has_converged = false
    counter = 0
    weight_old = 0.5
    I1 = zeros(graph.J, graph.J)

    while (!has_converged && counter < param[:MAX_ITER_KAPPA]) || counter <= 20

        # auxdata = create_auxdata(param, graph, I0)

        # if save_before_it_crashes
        #     debug_file_str = "debug.mat"
        #     save(debug_file_str, "param", "graph", "kappa", "x0", "I0", "I1", "counter")
        # end

        t0 = time()
        optimize!(model)
        # results, flag, x1 = solve_allocation_handle(x0, auxdata, funcs, verbose)
        t1 = time()
        if !is_solved_and_feasible(model, allow_almost = true)
            error("Solver returned with error code $(termination_status(model)).")
        end

        results = recover_allocation(model, auxdata)
        # Set next run starting values to previous run solution.
        # vars = all_variables(model)
        # vars_solution = value.(vars)
        # set_start_value.(vars, vars_solution)
        # # Previous: x0 = x1
        
        if !param[:cong]
            PQ = repeat(results[:Pjn], 1, graph.J, 1) .* results[:Qjkn] .^ (1 + param[:beta])
            PQ = dropdims(sum(PQ + permutedims(PQ, [2, 1, 3]), dims=3), dims = 3)
            I1 = (graph.delta_tau ./ graph.delta_i .* PQ) .^ (1 / (1 + param[:gamma]))
            I1[graph.adjacency .== 0] .= 0
            I1[PQ .== 0] .= 0
            I1[graph.delta_i .== 0] .= 0
        else
            PCj = repeat(results[:PCj], 1, graph.J)
            matm = permutedims(repeat(param[:m], 1, graph.J, graph.J), 2)
            cost = sum(matm .* results[:Qjkn] .^ param[:nu], dims=3) .^ ((param[:beta] + 1) / param[:nu])
            PQ = PCj .* cost
            PQ = PQ + PQ'
            I1 = (graph.delta_tau ./ graph.delta_i .* PQ) .^ (1 / (param[:gamma] + 1))
            I1[graph.adjacency.==false] .= 0
            I1[PQ.==0] .= 0
            I1[graph.delta_i.==0] .= 0
        end

        I1 = param[:K] * I1 / sum(graph.delta_i .* I1)
        distance_lb = max(maximum(Il - I1), 0)
        distance_ub = max(maximum(I1 - Iu), 0)
        counter_rescale = 0

        while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
            I1 = max.(min.(I1, Iu), Il)
            I1 = param[:K] * I1 / sum(graph.delta_i .* I1)
            distance_lb = max(maximum(Il - I1), 0)
            distance_ub = max(maximum(I1 - Iu), 0)
            counter_rescale += 1
        end

        if counter_rescale == 100 && distance_lb + distance_ub > param[:tol_kappa] && param[:verbose]
            println("Warning! Could not impose bounds on network properly.")
        end

        distance = maximum(abs.(I1 - I0)) / (param[:K] / mean(graph.delta_i[graph.adjacency.==1]))
        has_converged = distance < param[:tol_kappa]
        counter += 1

        if param[:verbose]
            println("Iteration No. $counter distance=$distance duration=$(t1 - t0) secs. Welfare=$(results[:welfare])")
        end

        if !has_converged || counter <= 20
            I0 = weight_old * I0 + (1 - weight_old) * I1
        end

    end

    if counter <= param[:MAX_ITER_KAPPA] && !has_converged && param[:verbose]
        println("Reached MAX iterations with convergence at $distance.")
        error_status = true
    end

    results[:Ijk] = I0

    if param[:verbose] && !error_status
        println("\nCOMPUTATION RESULTED WITH SUCCESS.\n----------------------------------\n")
    end

    if param[:gamma] > param[:beta] && param[:annealing]
        results = annealing(param, graph, I0, "Funcs" => funcs)
    end

    return results
end

# Please note that this is a direct translation and the code might not work correctly due to differences in how MATLAB and Julia handle arrays and other data structures. You might need to adjust the code to fit Julia's conventions and syntax.