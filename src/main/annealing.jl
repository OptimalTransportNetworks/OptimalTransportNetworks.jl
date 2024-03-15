# include("optimal_network.jl")
"""
    annealing(param, graph, I0; kwargs...)

Runs the simulated annealing method starting from network `I0`. Only sensible if `param[:gamma] > param[:beta]`.

# Arguments
- `param`: Dict that contains the model's parameters
- `graph`: Named tuple that contains the underlying graph 
- `I0`: (optional) provides the initial guess for the iterations

# Keyword Arguments
- `perturbation_method::String="random rebranching"`: Method to be used to perturbate the network 
    (random is purely random, works horribly; shake applies a gaussian blur
     along a random direction, works alright; rebranching (default) is the algorithm
     described in Appendix A.4 in the paper, works nicely )
- `preserve_central_symmetry::Bool=false`: Only applies to shake method
- `preserve_vertical_symmetry::Bool=false`: Only applies to shake method
- `preserve_horizontal_symmetry::Bool=false`: Only applies to shake method
- `smoothing_radius::Float64=0.25`: Parameters of the Gaussian blur
- `mu_perturbation::Float64=log(0.3)`: Parameters of the Gaussian blur
- `sigma_perturbation::Float64=0.05`: Parameters of the Gaussian blur
- `display::Bool`: Display the graph in each iteration as we go
- `t_start::Float64=100`: Initial temperature
- `t_end::Float64=1`: Final temperature
- `t_step::Float64=0.9`: Speed of cooling
- `num_deepening::Int64=4`: Number of FOC iterations between candidate draws
- `num_random_perturbations::Int64=1`: Number of links to be randomly affected ('random' and 'random rebranching' only)
- `Iu::Matrix{Float64}=Inf * ones(J, J)`: J x J matrix of upper bounds on network infrastructure Ijk
- `Il::Matrix{Float64}=zeros(J, J)`: J x J matrix of lower bounds on network infrastructure Ijk
- `model::Function`: For custom models => a function that taks an optimizer and an 'auxdata' structure as created by create_auxdata() as input and returns a fully parameterized JuMP model
- `final_model::JuMPModel`: Alternatively: a readily parameterized JuMP model to be used (from `optimal_network()`)
- `recover_allocation::Function`: The `recover_allocation()` function corresponding to either `model` or `final_model`

# Examples
```julia
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
function annealing(param, graph, I0; kwargs...)

    # Retrieve economy's parameters
    J = graph.J
    TOL_I_BOUNDS = 1e-7 
    best_results = nothing

    # Retrieve optional parameters
    options = retrieve_options_annealing(graph; kwargs...)
    Iu = options.Iu
    Il = options.Il

    if param[:mobility] || param[:beta] > 1 # use with the primal version only
        Il = max.(1e-6 * graph.adjacency, Il) # the primal approach in the mobility case requires a non-zero lower bound on kl, otherwise derivatives explode
    end

    # Parameters for simulated annealing algorithm
    T = options.t_start
    T_min = options.t_end # 1e-3
    T_step = options.t_step
    num_deepening = options.num_deepening # number of iterations for local network deepening

    # -------------------
    # PERTURBATION METHOD

    if options.perturbation_method == "random"
        perturbate = random_perturbation
    elseif options.perturbation_method == "shake"  
        perturbate = shake_network
    elseif options.perturbation_method == "rebranching"
        perturbate = rebranch_network
    elseif options.perturbation_method == "random rebranching"
        perturbate = random_rebranch_network
    # elseif options.perturbation_method == "hybrid alder"
    #     perturbate = hybrid
    else 
        error("unknown perturbation method %s\n", options.perturbation_method)
    end

    # =======================================
    # SET FUNCTION HANDLE TO SOLVE ALLOCATION

    if haskey(kwargs, :final_model)
        model = kwargs[:final_model]
        if !haskey(kwargs, :recover_allocation)
            error("Must provide recover_allocation function when providing final_model.")
        end
        recover_allocation = kwargs[:recover_allocation]
        # TODO: set IO (kappa_ex)?
    else
        auxdata = create_auxdata(param, graph, I0)
        optimizer = get(param, :optimizer, Ipopt.Optimizer)

        if haskey(param, :model)
            model = param[:model](optimizer, auxdata)
            if !haskey(param, :recover_allocation)
                error("The custom model does not have the recover_allocation function.")
            end
            recover_allocation = param[:recover_allocation] 
        elseif param[:mobility] == 1 && param[:cong]
            model = model_mobility_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility_cgc
        elseif param[:mobility] == 0.5 && param[:cong]
            model = model_partial_mobility_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility_cgc
        elseif param[:mobility] == 0 && param[:cong]
            model = model_fixed_cgc(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed_cgc
        elseif param[:mobility] == 1 && !param[:cong]
            model = model_mobility(optimizer, auxdata)
            recover_allocation = recover_allocation_mobility
        elseif param[:mobility] == 0.5 && !param[:cong]
            model = model_partial_mobility(optimizer, auxdata)
            recover_allocation = recover_allocation_partial_mobility    
        elseif param[:mobility] == 0 && !param[:cong]
            model = model_fixed(optimizer, auxdata)
            recover_allocation = recover_allocation_fixed
        else
            error("Usupported model configuration with labor_mobility = $(param[:mobility]) and cross_good_congestion = $(param[:cong])")
        end

        # --------------
        # CUSTOMIZATIONS

        if !param[:verbose]
            set_silent(model)
        end

        if haskey(param, :optimizer_attr)
            for (key, value) in param[:optimizer_attr]
                set_optimizer_attribute(model, String(key), value)
            end
        end

        if haskey(param, :model_attr) 
            for value in values(param[:model_attr])
                if !(value isa Tuple)  
                    error("model_attr must be a dict of tuples.")
                end
                set_optimizer_attribute(model, value[1], value[2])
            end
        end
        # E.g.:
        # set_attribute(model,
        #    MOI.AutomaticDifferentiationBackend(),
        #    MathOptSymbolicAD.DefaultBackend(),
        # )
    end

    # =========
    # ANNEALING
    # =========

    if param[:verbose]
        print("\n-------------------------------\n")
        print("STARTING SIMULATED ANNEALING...\n\n")
    end

    score = NaN # To avoid scoping errors
    best_score = -Inf
    best_I = I0

    old_score = best_score
    weight_old = 0.5
    counter = 0
    I1 = I0
    results = nothing

    # rng(0) # set the seed of random number generator for replicability
    acceptance_str = ["rejected", "accepted"]

    while T > T_min
        accepted = false

        if param[:verbose]
            println("Perturbating network (Temperature=$(T))...")
        end

        if counter > 0
            I1 = perturbate(param, graph, I0, results, options)
        end

        if options.display
            plot_graph(graph, I1)
        end

        if param[:verbose]
            println("Iterating on FOCs...")
        end

        k = 0
        while k <= num_deepening - 1

            auxdata = create_auxdata(param, graph, I1)
            set_parameter_value.(model.obj_dict[:kappa_ex], auxdata.kappa_ex)

            optimize!(model)

            if !is_solved_and_feasible(model, allow_almost = true) && param[:verbose] # optimization failed
                println("optimization failed! k=$(k), return flag=$(termination_status(model))")
                k = num_deepening - 1
                score = -Inf
            end

            results = recover_allocation(model, auxdata)
            score = results[:welfare]

            if param[:warm_start]
                vars = all_variables(model)
                vars_solution = value.(vars)
                set_start_value.(vars, vars_solution)
            end

            if score > best_score
                best_results = results
                best_I = I1
                best_score = score
            end

            # Deepen network
            if k < num_deepening - 1
                if !param[:cong]
                    PQ = permutedims(repeat(results[:Pjn], 1, 1, graph.J), [1, 3, 2]) .* results[:Qjkn] .^ (1 + param[:beta])
                    PQ = dropdims(sum(PQ + permutedims(PQ, [2, 1, 3]), dims=3), dims = 3)
                else
                    PQ = repeat(results[:PCj], 1, graph.J)
                    matm = permutedims(repeat(param[:m], 1, graph.J, graph.J), [3, 2, 1])
                    cost = dropdims(sum(matm .* results[:Qjkn] .^ param[:nu], dims=3), dims = 3) .^ ((param[:beta] + 1) / param[:nu])
                    PQ .*= cost
                    PQ += PQ'
                end
                
                I1 = (graph.delta_tau ./ graph.delta_i .* PQ) .^ (1 / (1 + param[:gamma]))
                I1[graph.adjacency .== 0] .= 0
                # I1[PQ .== 0] .= 0
                # I1[graph.delta_i .== 0] .= 0
                I1 *= param[:K] / sum(graph.delta_i .* I1)
                
                distance_lb = max(maximum(Il - I1), 0)
                distance_ub = max(maximum(I1 - Iu), 0)
                counter_rescale = 0
                
                while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
                    I1 = max.(min.(I1, Iu), Il)
                    I1 *= param[:K] / sum(graph.delta_i .* I1)
                    distance_lb = max(maximum(Il - I1), 0)
                    distance_ub = max(maximum(I1 - Iu), 0)
                    counter_rescale += 1
                end
                
                if counter_rescale == 100 && distance_lb + distance_ub > param[:kappa_tol] && param[:verbose]
                    println("Warning! Could not impose bounds on network properly.")
                end
            end
            k += 1
        end

        # Probabilistically accept perturbation 
        acceptance_prob = exp(1e4 * (score - old_score) / (abs(old_score) * T)) # set such that a 1% worse allocation is accepted with proba exp(-.1)=0.36 at beginning
        if rand() <= acceptance_prob || counter == 0
            I0 = I1
            old_score = score
            accepted = true
        end

        # --------------
        # DISPLAY STATUS

        if options.display
            plot_graph(graph, I1, node_sizes = results[:Lj])
        end

        if param[:verbose]
            println("Iteration No.$(counter), score=$(score), $(acceptance_str[1 + convert(Int32, accepted)]), (best=$(best_score))")
        end

        counter += 1
        T *= T_step
    end

    # Last deepening before returning found optimum
    has_converged = false
    I0 = best_I
    while !has_converged && counter < 100
        # Update auxdata
        auxdata = create_auxdata(param, graph, I0)
        set_parameter_value.(model.obj_dict[:kappa_ex], auxdata.kappa_ex)

        # Solve allocation
        optimize!(model)
        if !is_solved_and_feasible(model, allow_almost = true)
            error("Solver returned with error code $(termination_status(model)).")
        end
        results = recover_allocation(model, auxdata)
        score = results[:welfare]

        if param[:warm_start]
            vars = all_variables(model)
            vars_solution = value.(vars)
            set_start_value.(vars, vars_solution)
        end

        if options.display
            plot_graph(graph, I0, node_sizes = results[:Lj])
        end

        if score > best_score
            best_results = results
            best_I = I0
            best_score = score
        end

        # DEEPEN NETWORK
        if !param[:cong]
            PQ = permutedims(repeat(results[:Pjn], 1, 1, graph.J), [1, 3, 2]) .* results[:Qjkn] .^ (1 + param[:beta])
            PQ = dropdims(sum(PQ + permutedims(PQ, [2, 1, 3]), dims=3), dims = 3)
        else
            PQ = repeat(results[:PCj], 1, graph.J)
            matm = permutedims(repeat(param[:m], 1, graph.J, graph.J), [3, 2, 1])
            cost = dropdims(sum(matm .* results[:Qjkn] .^ param[:nu], dims=3), dims = 3) .^ ((param[:beta] + 1) / param[:nu])
            PQ .*= cost
            PQ += PQ'
        end
        
        I1 = (graph.delta_tau ./ graph.delta_i .* PQ) .^ (1 / (1 + param[:gamma]))
        I1[graph.adjacency .== 0] .= 0
        # I1[PQ .== 0] .= 0
        # I1[graph.delta_i .== 0] .= 0
        I1 *= param[:K] / sum(graph.delta_i .* I1)
        
        distance_lb = max(maximum(Il - I1), 0)
        distance_ub = max(maximum(I1 - Iu), 0)
        counter_rescale = 0
        
        while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
            I1 = max.(min.(I1, Iu), Il)
            I1 *= param[:K] / sum(graph.delta_i .* I1)
            distance_lb = max(maximum(Il - I1), 0)
            distance_ub = max(maximum(I1 - Iu), 0)
            counter_rescale += 1
        end
        
        if counter_rescale == 100 && distance_lb + distance_ub > param[:kappa_tol] && param[:verbose]
            println("Warning! Could not impose bounds on network properly.")
        end

        # UPDATE AND DISPLAY RESULTS
        distance = sum(abs.(I1 .- I0)) / (J^2)
        I0 *= weight_old 
        I0 += (1 - weight_old) * I1
        has_converged = distance < param[:kappa_tol]
        counter += 1

        if param[:verbose]
            println("Iteration No.$counter - final iterations - distance=$distance - welfare=$(score)")
        end
    end

    if param[:verbose]
        println("Welfare = ", score)
    end

    best_results[:Ijk] = best_I
    return best_results
end



function retrieve_options_annealing(graph; kwargs...)

    # Set up default options with lowercase names
    options = Dict{Symbol, Any}(
        :perturbation_method => "random rebranching",
        :preserve_central_symmetry => false,
        :preserve_vertical_symmetry => false,
        :preserve_horizontal_symmetry => false,
        :smoothing_radius => 0.25,
        :mu_perturbation => log(0.3),
        :sigma_perturbation => 0.05,
        :display => false,
        :t_start => 100,
        :t_end => 1,
        :t_step => 0.9,
        :num_deepening => 4,
        :num_random_perturbations => 1,
        :Iu => Inf * ones(graph.J, graph.J),
        :Il => zeros(graph.J, graph.J)
    )

    # Update options with user-provided values
    for (k, v) in kwargs
        sym_key = Symbol(lowercase(string(k)))  # Convert to lowercase symbol
        if haskey(options, sym_key)
            options[sym_key] = v
        else 
            if !(sym_key in [:model, :final_model, :recover_allocation])
                error("Unknown parameter: $sym_key")
            end
        end
    end

    # Ensure numerical options are correct type
    options[:smoothing_radius] = Float64(options[:smoothing_radius])
    options[:mu_perturbation] = Float64(options[:mu_perturbation])
    options[:sigma_perturbation] = Float64(options[:sigma_perturbation])
    options[:t_start] = Float64(options[:t_start])
    options[:t_end] = Float64(options[:t_end])
    options[:t_step] = Float64(options[:t_step])
    options[:num_deepening] = round(Int, options[:num_deepening])
    options[:num_random_perturbations] = round(Int, options[:num_random_perturbations])

    # Validate sizes of matrix options
    if size(options[:Iu]) != (graph.J, graph.J)
        error("Iu must be of size (graph.J, graph.J)")
    end
    if size(options[:Il]) != (graph.J, graph.J)
        error("Il must be of size (graph.J, graph.J)")
    end

    return dict_to_namedtuple(options)
end

# using Random

# This function adds #num_random_perturbations random links to the network and
# applies a Gaussian smoothing to prevent falling too quickly in a local optimum
function random_perturbation(param, graph, I0, results, options)
    size_perturbation = 0.1 * param[:K] / graph.J
    I1 = copy(I0)

    # Draw random perturbations
    link_list = Random.randperm(graph.J)[1:options.num_random_perturbations]

    for i in link_list
        j = rand(1:length(graph.nodes[i]))
        neighbor = graph.nodes[i][j]
        I1[i, neighbor] = size_perturbation * exp(randn()) / exp(0.5)
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 += I1'  # make sure kappa is symmetric
    I1 ./= 2
    I1 *= param[:K] / sum(graph.delta_i .* I1)  # rescale

    # Smooth network (optional)
    I1 = smooth_network(param, graph, I1)

    return I1
end



# This function produces a smoothed version of the network by applying a
# Gaussian smoother (i.e., a Gaussian kernel estimator). The main benefit is
# that it makes the network less sparse and the variance of investments gets reduced.
function smooth_network(param, graph, I0)
    J = graph.J
    smoothing_radius = 0.3  # This could also be an option in `param` if variable

    I1 = zeros(J, J)
    feasible_edge = zeros(Bool, J, J)
    vec_x = zeros(J)
    vec_y = zeros(J)

    # Set up node coordinates and feasible edges
    for i = 1:J
        vec_x[i] = graph.x[i]
        vec_y[i] = graph.y[i]
        for neighbor in graph.nodes[i]
            feasible_edge[i, neighbor] = true
        end
    end

    edge_x = 0.5 * (vec_x * ones(J)' .+ ones(J) * vec_x')
    edge_y = 0.5 * (vec_y * ones(J)' .+ ones(J) * vec_y')

    # Proceed to Gaussian kernel smoothing
    # TODO: Speed up??
    for i = 1:J
        for neighbor in graph.nodes[i]
            x0 = edge_x[i, neighbor]
            y0 = edge_y[i, neighbor]

            weight = exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0).^2 .+ (edge_y .- y0).^2))
            weight[.!feasible_edge] .= 0
            I1[i, neighbor] = sum(I0 .* weight) / sum(weight)
        end
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 += I1'  # make sure kappa is symmetric
    I1 ./= 2
    I1 *= param[:K] / sum(graph.delta_i .* I1)  # rescale

    return I1
end



# A simple way to perturb the network that simulates shaking the network in some 
# random direction and applying a Gaussian smoothing (see smooth_network()).
function shake_network(param, graph, I0, results, options)

    J = graph.J

    # ===================
    # RETRIEVE PARAMETERS
    # ===================

    smoothing_radius = options.smoothing_radius
    mu_perturbation = options.mu_perturbation
    sigma_perturbation = options.sigma_perturbation

    # ===============
    # PERTURB NETWORK
    # ===============

    theta = cispi(rand() * 2) # cispi(x) is exp(i*pi*x) in Julia
    rho = exp(mu_perturbation + sigma_perturbation * randn()) / exp(sigma_perturbation^2)
    direction = rho * theta  
    direction_x = real(direction)
    direction_y = imag(direction)

    I1 = zeros(J, J)
    vec_x = graph.x
    vec_y = graph.y
    feasible_edge = graph.adjacency

    edge_x = 0.5 * (vec_x * ones(J)' .+ ones(J) * vec_x')
    edge_y = 0.5 * (vec_y * ones(J)' .+ ones(J) * vec_y')

    # Proceed to Gaussian kernel smoothing
    for i = 1:J
        for neighbor in graph.nodes[i]
            x0 = edge_x[i, neighbor]
            y0 = edge_y[i, neighbor]

            weight = exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0 .- direction_x).^2 .+ (edge_y .- y0 .- direction_y).^2))
            if options.preserve_central_symmetry
                weight .+= exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0 .+ direction_x).^2 .+ (edge_y .- y0 .+ direction_y).^2))
            end
            if options.preserve_horizontal_symmetry
                weight .+= exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0 .- direction_x).^2 .+ (edge_y .- y0 .+ direction_y).^2))
            end
            if options.preserve_vertical_symmetry
                weight .+= exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0 .+ direction_x).^2 .+ (edge_y .- y0 .- direction_y).^2))
            end
            weight[.!feasible_edge] .= 0
            I1[i, neighbor] = sum(I0 .* weight) / sum(weight)
        end
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 += I1'  # make sure kappa is symmetric
    I1 ./= 2
    I1 *= param[:K] / sum(graph.delta_i .* I1)  # rescale

    return I1
end


"""
    rebranch_network(param, graph, I0, results, options)

This function implements the rebranching algorithm described in the paper.
Links are reshuffled everywhere so that each node is better connected to its best neighbor
(those with the lowest price index for traded goods, i.e., more central places in the trading network).
"""
function rebranch_network(param, graph, I0, results, options)
    J = graph.J

    # ===============
    # PERTURB NETWORK
    # ===============

    I1 = copy(I0)

    # Rebranch each location to its lowest price parent
    for i = 1:J
        neighbors = graph.nodes[i]
        parents = neighbors[results[:PCj][neighbors] .< results[:PCj][i]]
        
        if length(parents) >= 2
            lowest_price_parent = findmin(results[:PCj][parents])[2]
            lowest_price_parent = parents[lowest_price_parent]
            best_connected_parent = findmax(I0[i, parents])[2]
            best_connected_parent = parents[best_connected_parent]
            
            # swap roads
            I1[i, lowest_price_parent] = I0[i, best_connected_parent]
            I1[i, best_connected_parent] = I0[i, lowest_price_parent]
            I1[lowest_price_parent, i] = I0[best_connected_parent, i]
            I1[best_connected_parent, i] = I0[lowest_price_parent, i]
        end
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 += I1'  # make sure kappa is symmetric
    I1 ./= 2
    I1 *= param[:K] / sum(graph.delta_i .* I1)  # rescale

    return I1
end


"""
    random_rebranch_network(param, graph, I0, results, options)

This function does the same as `rebranch_network` except that only a few nodes
(#num_random_perturbations) are selected for rebranching at random.
"""
function random_rebranch_network(param, graph, I0, results, options)
    J = graph.J

    # ===============
    # PERTURB NETWORK
    # ===============

    I1 = copy(I0)

    # Random selection of nodes to rebranch
    link_list = Random.randperm(J)[1:options.num_random_perturbations]

    # Rebranch each selected location to its lowest price parent
    for i in link_list
        neighbors = graph.nodes[i]
        parents = neighbors[results[:PCj][neighbors] .< results[:PCj][i]]
        
        if length(parents) >= 2
            lowest_price_parent_index = findmin(results[:PCj][parents])[2]
            lowest_price_parent = parents[lowest_price_parent_index]
            best_connected_parent_index = findmax(I0[i, parents])[2]
            best_connected_parent = parents[best_connected_parent_index]
            
            # Swap roads
            I1[i, lowest_price_parent] = I0[i, best_connected_parent]
            I1[i, best_connected_parent] = I0[i, lowest_price_parent]
            I1[lowest_price_parent, i] = I0[best_connected_parent, i]
            I1[best_connected_parent, i] = I0[lowest_price_parent, i]
        end
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 += I1'  # make sure kappa is symmetric
    I1 ./= 2
    I1 *= param[:K] / sum(graph.delta_i .* I1)  # rescale

    return I1
end


"""
    hybrid(param, graph, I0, results, options)

This function attempts to adapt the spirit of Alder (2018)'s algorithm to
delete/add links to the network in a way that blends with our model.
"""
function hybrid(param, graph, I0, results, options)
    # ========================
    # COMPUTE GRADIENT WRT Ijk
    # ========================
    J = graph.J

    if !param[:cong] # no cross-good congestion
        PQ = permutedims(repeat(results[:Pjn], 1, 1, graph.J), [1, 3, 2]) .* results[:Qjkn] .^ (1 + param[:beta])
        PQ = dropdims(sum(PQ + permutedims(PQ, [2, 1, 3]), dims=3), dims = 3)
    else # cross-good congestion
        PQ = repeat(results[:PCj], 1, graph.J)
        matm = permutedims(repeat(param[:m], 1, graph.J, graph.J), [3, 2, 1])
        cost = dropdims(sum(matm .* results[:Qjkn] .^ param[:nu], dims=3), dims = 3) .^ ((param[:beta] + 1) / param[:nu])
        PQ .*= cost
        PQ += PQ'
    end
    
    grad = param[:gamma] * graph.delta_tau ./ graph.delta_i .* PQ .* I0.^-(1 + param[:gamma])
    grad[graph.adjacency .== 0] .= 0
    # I1[PQ .== 0] .= 0
    # I1[graph.delta_i .== 0] .= 0

    # ============
    # REMOVE LINKS: remove 5% worst links
    # ============
    I1 = copy(I0)
    nremove = ceil(Int, 0.05 * graph.ndeg) # remove 5% of the links
    remove_list = sortperm(vec(grad[tril(graph.adjacency)]), alg=QuickSort)[1:nremove]

    id = 1
    for j in 1:J
        for k in graph.nodes[j]
            if id in remove_list  # if link is in the list to remove
                I1[j, k] = 1e-6
                I1[k, j] = 1e-6
            end
            id += 1
        end
    end

    # TODO: Finish revision this function from here onwards! 
    # Problem: need model as input to solve again!!
    # ====================
    # COMPUTE NEW GRADIENT
    # ====================
    results = solve_allocation(param, graph, I1) # Assuming this function is defined elsewhere

    if !param[:cong] # no cross-good congestion
        Pjkn = repeat(permute(results[:Pjn], [1, 3, 2]), outer=[1, J, 1])
        PQ = Pjkn .* results[:Qjkn].^(1 + param[:beta])
        grad = param[:gamma] * graph.delta_tau .* sum(PQ + permute(PQ, [2, 1, 3]), dims=3) .* I0.^-(1 + param[:gamma]) ./ graph.delta_i
        grad[.!graph.adjacency] .= 0
    else # cross-good congestion
        PCj = repeat(results[:PCj], outer=[1, J])
        matm = permutedims(repeat(param[:m], outer=[1, J, J]), [2, 1, 3])
        cost = sum(matm .* results[:Qjkn].^param[:nu], dims=3).^((param[:beta] + 1) / param[:nu])
        PQ = PCj .* cost
        grad = param[:gamma] * graph.delta_tau .* (PQ + PQ') .* I0.^-(1 + param[:gamma]) ./ graph.delta_i
        grad[.!graph.adjacency] .= 0
    end

    # ========
    # ADD LINK: add the most beneficial link
    # ========

    I2 = copy(I1)
    add_list = sortperm(vec(grad[tril(graph.adjacency)]), rev=true, alg=QuickSort)
    add_link = add_list[1]

    id = 1
    for j in 1:J
        for k in graph.nodes[j]
            if id == add_link # if link is the one to add
                I2[j, k] = param[:K] / (2 * graph.ndeg)
                I2[k, j] = param[:K] / (2 * graph.ndeg)
            end
            id += 1
        end
    end

    # =======
    # RESCALE
    # =======

    # Make sure graph satisfies symmetry and capacity constraint
    I2 = (I2 + I2') / 2 # make sure kappa is symmetric
    total_delta_i = sum(graph.delta_i .* I2)
    I2 *= param[:K] / total_delta_i # rescale

    return I2
end



