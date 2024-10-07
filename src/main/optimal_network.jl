# ==============================================================
# OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
# by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================ version 1.0.4

"""
    optimal_network(param, graph; I0=nothing, Il=nothing, Iu=nothing, 
                    verbose=false, return_model=0, solve_allocation = false) -> Dict

Solve for the optimal network by solving the inner problem and the outer problem by iterating over the FOCs.

# Arguments
- `param`: Dict or NamedTuple that contains the model's parameters
- `graph`: Dict or NamedTuple that contains the underlying graph (created by `create_graph()` function)
- `I0::Matrix{Float64}=nothing`: (Optional) J x J matrix providing the initial guess for the iterations 
- `Il::Matrix{Float64}=nothing`: (Optional) J x J matrix providing exogenous lower bound on infrastructure levels
- `Iu::Matrix{Float64}=nothing`: (Optional) J x J matrix providing exogenous upper bound on infrastructure levels
- `verbose::Bool=false`: (Optional) tell IPOPT to display results
- `return_model::Int=0`: (Optional) return the JuMP model and corresponding `recover_allocation()` function: 1 just returns these before solving the model, while 2 solves the model + optimal network and returns the two alongside the results. 
- `solve_allocation::Bool=false`: (Optional) just solve the model with existing infrastructure I0 and return the results. 

# Examples
```julia
param = init_parameters(K = 10)
graph = create_graph(param)
graph[:Zjn][61] = 10.0
results = optimal_network(param, graph)
plot_graph(graph, results[:Ijk])
```
"""
function optimal_network(param, graph; I0=nothing, Il=nothing, Iu=nothing, verbose=false, return_model=0, solve_allocation = false)

    # I0=nothing; Il=nothing; Iu=nothing; verbose=false; return_model = false; return_model = 0;
    # Check graph.Zjn and make matrix if vector:
    if length(size(graph[:Zjn])) == 1
        graph[:Zjn] = reshape(graph[:Zjn], graph[:J], 1)
    end   
    graph = dict_to_namedtuple(graph)
    param = dict_to_namedtuple(param)
    edges = represent_edges(graph)
 
    J = graph.J
    error_status = false

    # ---------------------------------------------
    # CHECK PARAMETERS FOR ERRORS OR MISSING VALUES

    if param.a > 1
        error("Model with increasing returns to scale are not convex.")
    end

    if param.nu < 1 && param.cong
        error("nu has to be larger or equal to one for the problem to be guaranteed convex.")
    end

    Il = Il === nothing ? zeros(J, J) : max.(Il, 0.0)
    Iu = Iu === nothing ? fill(Inf, J, J) : max.(Iu, 0.0)
    I0 = I0 === nothing ? Float64.(graph.adjacency) : max.(I0, 0.0)
    I0 *= param.K / sum(graph.delta_i .* I0) # 
    # I0 = rescale_network!(param, graph, I0, Il, Iu)

    if param.mobility != 0 || param.beta > 1 || param.cong
        Il = max.(1e-6 * graph.adjacency, Il)
    end

    # --------------
    # INITIALIZATION

    auxdata = create_auxdata(param, graph, edges, I0)
    model, recover_allocation = get_model(auxdata)

    if return_model == 1
        return model, recover_allocation
    end

    if !verbose && model !== nothing
        set_silent(model)
    end
    
    # --------------------
    # NETWORK OPTIMIZATION

    has_converged = false
    counter = 0
    weight_old = 0.5
    I1 = zeros(J, J)
    # K in average infrastructure units
    K_infra = (param.K / mean(graph.delta_i[graph.adjacency .== 1]))
    distance = 0.0
    if model === nothing
        start_values = []
    else
        all_vars = all_variables_except_kappa_ex(model)
        start_values = start_value.(all_vars)
    end
    used_warm_start = false

    while (!has_converged && counter < param.max_iter) || counter <= param.min_iter


        skip_update = false

        # Dual Solution: Most efficient
        if model === nothing
            t0 = time()
            results, status, start_values = recover_allocation(start_values, auxdata, verbose)
            t1 = time()

            if solve_allocation
                if status != 0
                    warning("Solver returned with error code $(status)).")
                end
                return results
            end

            if !any(status .== [0, 1])
                if used_warm_start # if error happens with warm start, then try again starting cold
                    start_values = []
                    used_warm_start = false
                    skip_update = true
                else
                    error("Solver returned with error code $(status).")
                end
            else
                if param.warm_start
                    used_warm_start = true
                else
                    start_values = []
                end
            end

        else # Primal Solution (Using JuMP Model)
            t0 = time()
            optimize!(model)
            results = recover_allocation(model, auxdata)
            t1 = time()

            if solve_allocation
                if !is_solved_and_feasible(model, allow_almost = true)
                    warning("Solver returned with error code $(termination_status(model)).")
                end
                return results
            end

            if !is_solved_and_feasible(model, allow_almost = true)
                if used_warm_start # if error happens with warm start, then try again starting cold
                    set_start_value.(all_vars, start_values)
                    used_warm_start = false
                    skip_update = true
                else
                    error("Solver returned with error code $(termination_status(model)).")
                end
            elseif param.warm_start # Set next run starting values to previous run solution.
                vars_solution = value.(all_vars)
                set_start_value.(all_vars, vars_solution)
                used_warm_start = true
            end
        end

        # Computing prices times optimal flows term which yields optimal infrastructure
        if !param.cong
            PQ = permutedims(repeat(results[:Pjn], 1, 1, J), [1, 3, 2]) .* results[:Qjkn] .^ (1 + param.beta)
            PQ = dropdims(sum(PQ + permutedims(PQ, [2, 1, 3]), dims=3), dims = 3)
        else
            PQ = repeat(results[:PCj], 1, J)
            matm = permutedims(repeat(param.m, 1, J, J), [3, 2, 1])
            cost = dropdims(sum(matm .* results[:Qjkn] .^ param.nu, dims=3), dims = 3) .^ ((param.beta + 1) / param.nu)
            PQ .*= cost
            PQ += PQ'
        end
        
        # This computes optimal infrastructure
        I1 = (graph.delta_tau ./ graph.delta_i .* PQ) .^ (1 / (1 + param.gamma))
        I1[graph.adjacency .== 0] .= 0
        I1[PQ .== 0] .= 0
        I1[graph.delta_i .== 0] .= 0
        I1 *= param.K / sum(graph.delta_i .* I1)
        I1 = rescale_network!(param, graph, I1, Il, Iu)

        distance = maximum(abs.(I1 - I0)) / K_infra
        has_converged = distance < param.tol
        counter += 1

        if param.verbose
            println("Iteration No. $counter distance=$distance duration=$(t1 - t0) secs. Welfare=$(results[:welfare])")
        end

        if (!has_converged || counter <= param.min_iter) && !skip_update
            I0 *= weight_old 
            I0 += (1 - weight_old) * I1
            # This creates kappa and updates the model
            auxdata = create_auxdata(param, graph, edges, I0)
            if model !== nothing
                set_parameter_value.(model[:kappa_ex], auxdata.kappa_ex)
            end
        end
    end

    if counter <= param.max_iter && !has_converged && param.verbose
        println("Reached MAX iterations with convergence at $distance.")
        error_status = true
    end

    results[:Ijk] = I0

    if param.verbose && !error_status
        println("\nCOMPUTATION RESULTED WITH SUCCESS.\n----------------------------------\n")
    end

    # --------------------
    # SIMULATED ANNEALING

    if param.gamma > param.beta && param.annealing
        if model !== nothing
            set_start_value.(all_vars, start_values) # MATLAB code does this in the annealing function
        end
        results = annealing(param, graph, I0, final_model = model, 
                            recover_allocation = recover_allocation, 
                            allocation = results, verbose = verbose)
    end

    if return_model == 2
        if model !== nothing
            if !verbose
                unset_silent(model)
            end
            # Better do this for use in annealing
            set_start_value.(all_vars, start_values) 
        end
        return results, model, recover_allocation
    end

    return results
end

