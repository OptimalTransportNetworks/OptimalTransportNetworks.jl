"""
    annealing(param, graph, I0; kwargs...)

Runs the simulated annealing method starting from network `I0`.

# Arguments
- `param`: Dict that contains the model's parameters
- `graph`: Dict that contains the underlying graph (created by
  `create_map`, `create_rectangle` or `create_triangle` functions)
- `I0`: (optional) provides the initial guess for the iterations
- `kwargs...`: various optional arguments, see below

# Optional Arguments
- `:PerturbationMethod`: method to be used to perturbate the network
  (random is purely random, works horribly; shake applies a gaussian blur
  along a random direction, works alright; rebranching (default) is the algorithm
  described in Appendix A.4 in the paper, works nicely )
- `:PreserveCentralSymmetry`: only applies to shake method
- `:PreserveVerticalSymmetry`: only applies to shake method
- `:PreserveHorizontalSymmetry`: only applies to shake method
- `:SmoothingRadius`: parameters of the Gaussian blur
- `:MuPerturbation`: parameters of the Gaussian blur
- `:SigmaPerturbation`: parameters of the Gaussian blur
- `:Display`: display the graph as we go
- `:TStart`: initial temperature
- `:TEnd`: final temperature
- `:TStep`: speed of cooling
- `:NbDeepening`: number of FOC iterations between candidate draws
- `:NbRandomPerturbations`: number of links to be randomly affected
  ('random' and 'random rebranching' only)
- `:Funcs`: funcs structure computed by ADiGator in order to skip rederivation
- `:Iu`: JxJ matrix of upper bounds on network infrastructure Ijk
- `:Il`: JxJ matrix of lower bounds on network infrastructure Ijk

# Reference
"Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

# License
This code is distributed under BSD-3 License.
"""

function annealing(param, graph, I0; kwargs...)
    # Retrieve economy's parameters
    J = graph.J
    gamma = param.gamma
    verbose = false
    TOL_I_BOUNDS = 1e-7

    # Retrieve optional parameters
    options = retrieve_options_annealing(param, graph; kwargs...)
    Iu = options.Iu
    Il = options.Il

    if param.mobility || param.beta > 1 # use with the primal version only
        Il = max.(1e-6 * graph.adjacency, Il) # the primal approach in the mobility case requires a non-zero lower bound on kl, otherwise derivatives explode
    end

    # Parameters for simulated annealing algorithm
    T = options.t_start
    T_min = options.t_end # 1e-3
    T_step = options.t_step
    nb_network_deepening = options.nb_deepening # number of iterations for local network deepening

    # -------------------
    # PERTURBATION METHOD

    if options.perturbation_method == "random"
        perturbate = random_perturbation(param, graph, I0, res, options)
    elseif options.perturbation_method == "shake"  
        perturbate = shake_network(param, graph, I0, res, options)
    elseif options.perturbation_method == "rebranching"
        perturbate = rebranch_network(param, graph, I0, res, options)  
    elseif options.perturbation_method == "random rebranching"
        perturbate = random_rebranch_network(param, graph, I0, res, options)
    elseif options.perturbation_method == "hybrid alder"
        perturbate = hybrid(param, graph, I0, res, options)
    else 
        error("unknown perturbation method %s\n", options.perturbation_method)
    end

    # ----------
    # INIT STUFF

    # initial point in the IPOPT optimization
    x0 = [] # automatic initialization except for custom case
    # CUSTOMIZATION 1: provide here the initial point of optimization for custom case
    if param.custom # Enter here the initial point of optimization for custom case
        x0 = [1e-6 * ones(graph.J * param.N, 1); zeros(graph.ndeg * param.N, 1); sum(param.Lj) / (graph.J * param.N) * ones(graph.J * param.N, 1)]
        # Based on the primal case with immobile and no cross-good congestion, to be customized
    end

    # =======================================
    # SET FUNCTION HANDLE TO SOLVE ALLOCATION

    if param.adigator && param.mobility!=0.5 # IF USING ADIGATOR
        if param.mobility && param.cong && !param.custom # implement primal with mobility and congestion
            solve_allocation_handle = solve_allocation_mobility_cgc_ADiGator
        elseif !param.mobility && param.cong && !param.custom # implement primal with congestion
            solve_allocation_handle = solve_allocation_cgc_ADiGator
        elseif param.mobility && !param.cong && !param.custom # implement primal with mobility
            solve_allocation_handle = solve_allocation_mobility_ADiGator
        elseif (!param.mobility && !param.cong && !param.custom) && (param.beta<=1 && param.a < 1 && param.duality) # implement dual
            solve_allocation_handle = solve_allocation_by_duality_ADiGator
        elseif (!param.mobility && !param.cong && !param.custom) && (param.beta>1 || param.a == 1) # implement primal
            solve_allocation_handle = solve_allocation_ADiGator
        elseif param.custom # run custom optimization
            solve_allocation_handle = solve_allocation_custom_ADiGator
        end
    else # IF NOT USING ADIGATOR
        if !param.cong
            if param.mobility==0 
                if param.beta<=1 && param.duality # dual is only twice differentiable if beta<=1
                    solve_allocation_handle = solve_allocation_by_duality
                else # otherwise solve the primal
                    solve_allocation_handle = solve_allocation_primal
                end
            elseif param.mobility==1 # always solve the primal with labor mobility
                solve_allocation_handle = solve_allocation_mobility
            elseif param.mobility==0.5
                solve_allocation_handle = solve_allocation_partial_mobility
            end
        else
            if param.mobility==0
                solve_allocation_handle = solve_allocation_cgc
            elseif param.mobility==1
                solve_allocation_handle = solve_allocation_mobility_cgc
            elseif param.mobility==0.5
                solve_allocation_handle = solve_allocation_partial_mobility_cgc
            end
        end
    end


    # ===================================================
    # GENERATE AUTODIFFERENTIATED FUNCTIONS WITH ADIGATOR
    # ===================================================

    funcs = options.funcs
    if param.adigator && isempty(funcs)
        funcs = call_adigator(param, graph, I0, verbose)
    end

    # =========
    # ANNEALING
    # =========

    if param.verbose
        print("\n-------------------------------\n")
        print("STARTING SIMULATED ANNEALING...\n\n")
    end

    best_score = -Inf
    best_I = I0

    old_score = best_score
    weight_old = 0.5
    counter = 0
    I1 = I0

    # rng(0) # set the seed of random number generator for replicability
    acceptance_str = ["rejected", "accepted"]

    while T > T_min
        accepted = false

        if param.verbose
            println("Perturbating network (Temperature=$(T))...")
        end

        if counter > 0
            I1 = perturbate(param, graph, I0, results)
        end

        if options.display
            plot_graph(param, graph, I1)
        end

        if param.verbose
            println("Iterating on FOCs...");
        end

        k = 0
        x = x0 # use default initial condition for allocation
        while k <= nb_network_deepening - 1
            # Create auxdata structure for IPOPT/ADiGator
            auxdata = create_auxdata(param, graph, I1)
            # Solve allocation
            results, flag, x = solve_allocation_handle(x, auxdata, funcs, verbose)
            score = results.welfare

            if (!any(flag.status .== [0, 1]) || isnan(score)) && param.verbose # optimization failed
                println("optimization failed! k=$(k), return flag=$(flag.status)")
                k = nb_network_deepening - 1
                score = -Inf
            end

            if score > best_score
                best_results = results
                best_I = I1
                best_score = score
            end

            # Deepen network
            if k < nb_network_deepening - 1
                if !param.cong # no cross-good congestion
                    Pjkn = repeat(permute(results.Pjn, [1 3 2]), [1 graph.J 1])
                    PQ = Pjkn .* results.Qjkn.^(1 + param.beta)
                    I1 = (graph.delta_tau ./ graph.delta_i .* sum(PQ + permutedims(PQ, [2 1 3]), 3)).^(1 / (1 + param.gamma))
                    I1[graph.adjacency .== false] .= 0
                else # cross-good congestion
                    PCj = repeat(results.PCj, [1 graph.J])
                    matm = shiftdim(repeat(param.m, [1 graph.J graph.J]), 1)
                    cost = sum(matm .* results.Qjkn.^param.nu, 3).^((param.beta + 1) / param.nu)
                    PQ = PCj .* cost
                    I1 = (graph.delta_tau ./ graph.delta_i .* (PQ + PQ')).^(1 / (param.gamma + 1))
                    I1[graph.adjacency .== false] .= 0
                end

                # CUSTOMIZATION 2: updating network
                if param.custom
                    # enter here how to update the infrastructure network I1 (if needed) in the custom case
                end

                # Take care of scaling and bounds
                I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1])) # rescale so that sum(delta*kappa^1/gamma)=K

                distance_lb = max(maximum(Il .- I1), 0)
                distance_ub = max(maximum(I1 .- Iu), 0)
                counter_rescale = 0
                while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
                    I1 = max.(min.(I1, Iu), Il) # impose the upper and lower bounds
                    I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1])) # rescale again
                    distance_lb = max(maximum(Il .- I1), 0)
                    distance_ub = max(maximum(I1 .- Iu), 0)
                    counter_rescale += 1
                end

                if counter_rescale == 100 && distance_lb + distance_ub > param.tol_kappa && param.verbose
                    println("Warning! Could not impose bounds on network properly.")
                end

            end
            k = k + 1
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
            plot_graph(param, graph, I1; Sizes = results.Lj)
        end

        if param.verbose
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
        # Solve allocation
        results, flag, x = solve_allocation_handle(x0, auxdata, funcs, verbose)
        score = results.welfare

        if options.display
            plot_graph(param, graph, I0; Sizes = results.Lj)
        end

        if score > best_score
            best_results = results
            best_I = I0
            best_score = score
        end

        # DEEPEN NETWORK
        if !param.cong # no cross-good congestion
            Pjkn = repeat(permute(results.Pjn, [1 3 2]), [1 graph.J 1])
            PQ = Pjkn .* results.Qjkn.^(1 + param.beta)
            I1 = (graph.delta_tau ./ graph.delta_i .* sum(PQ + permutedims(PQ, [2 1 3]), 3)).^(1 / (1 + param.gamma))
            I1[graph.adjacency .== false] .= 0
        else # cross-good congestion
            PCj = repeat(results.PCj, [1 graph.J])
            matm = shiftdim(repeat(param.m, [1 graph.J graph.J]), 1)
            cost = sum(matm .* results.Qjkn.^param.nu, 3).^((param.beta + 1) / param.nu)
            PQ = PCj .* cost
            I1 = (graph.delta_tau ./ graph.delta_i .* (PQ + PQ')).^(1 / (param.gamma + 1))
            I1[graph.adjacency .== false] .= 0
        end

        # CUSTOMIZATION 2': updating network
        if param.custom
            # enter here how to update the infrastructure network I1 (if needed) in the custom case
        end

        # Take care of scaling and bounds
        I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1])) # rescale so that sum(delta*kappa^1/gamma)=K

        distance_lb = max.(maximum(Il .- I1), 0)
        distance_ub = max.(maximum(I1 .- Iu), 0)
        counter_rescale = 0
        while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
            I1 = max.(min.(I1, Iu), Il) # impose the upper and lower bounds
            I1 = param.K * I1 ./ sum(reshape.(graph.delta_i .* I1, [graph.J^2 1])) # rescale again
            distance_lb = max.(max.(Il .- I1), 0) 
            distance_ub = max.(max.(I1 .- Iu), 0)
            counter_rescale += 1
        end

        if counter_rescale == 100 && distance_lb + distance_ub > param.tol_kappa && param.verbose
            println("Warning! Could not impose bounds on network properly.")
        end

        # UPDATE AND DISPLAY RESULTS
        distance = sum(abs.(I1 .- I0)) / (J^2)
        I0 = weight_old * I0 + (1 - weight_old) * I1
        has_converged = distance < param.tol_kappa
        counter += 1

        if param.verbose
            println("Iteration No.$counter - final iterations - distance=$distance - welfare=$(score)")
        end

    end

    if param.verbose
        println("Welfare = ", score)
    end

    best_results.Ijk = best_I
    return best_results
end




function retrieve_options_annealing(param, graph; kwargs...)
    # Set up default options with lowercase names
    options = Dict{Symbol, Any}(
        :perturbation_method => "random rebranching",
        :preserve_central_symmetry => "off",
        :preserve_vertical_symmetry => "off",
        :preserve_horizontal_symmetry => "off",
        :smoothing_radius => 0.25,
        :mu_perturbation => log(0.3),
        :sigma_perturbation => 0.05,
        :display => "off",
        :t_start => 100,
        :t_end => 1,
        :t_step => 0.9,
        :nb_deepening => 4,
        :nb_random_perturbations => 1,
        :funcs => [],
        :iu => Inf * ones(graph.J, graph.J),
        :il => zeros(graph.J, graph.J)
    )

    # A helper function to convert "on"/"off" to true/false
    convert_switch = (val) -> val == "on"

    # Update options with user-provided values
    for (k, v) in kwargs
        sym_key = Symbol(lowercase(string(k)))  # Convert to lowercase symbol
        if haskey(options, sym_key)
            # Assign the value after any necessary conversion
            options[sym_key] = sym_key in [:preserve_central_symmetry, :preserve_vertical_symmetry, :preserve_horizontal_symmetry, :display] ? convert_switch(v) : v
        else
            error("Unknown parameter: $sym_key")
        end
    end

    # Ensure numerical options are correct type
    options[:smoothing_radius] = Float64(options[:smoothing_radius])
    options[:mu_perturbation] = Float64(options[:mu_perturbation])
    options[:sigma_perturbation] = Float64(options[:sigma_perturbation])
    options[:t_start] = Float64(options[:t_start])
    options[:t_end] = Float64(options[:t_end])
    options[:t_step] = Float64(options[:t_step])
    options[:nb_deepening] = round(Int, options[:nb_deepening])
    options[:nb_random_perturbations] = round(Int, options[:nb_random_perturbations])

    # Validate sizes of matrix options
    if size(options[:iu]) != (graph.J, graph.J)
        error("Iu must be of size (graph.J, graph.J)")
    end
    if size(options[:il]) != (graph.J, graph.J)
        error("Il must be of size (graph.J, graph.J)")
    end

    return options
end

# using Random

# This function adds #NbRandomPerturbations random links to the network and
# applies a Gaussian smoothing to prevent falling too quickly in a local optimum
function random_perturbation(param, graph, I0, res, options)
    size_perturbation = 0.1 * param[:K] / graph.J
    I1 = copy(I0)

    # Draw random perturbations
    link_list = shuffle(1:graph.J)[1:options[:nb_random_perturbations]]

    for i in link_list
        j = rand(1:length(graph.nodes[i]))
        neighbor = graph.nodes[i][j]
        I1[i, neighbor] = size_perturbation * exp(randn()) / exp(0.5)
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 = (I1 + I1') / 2  # make sure kappa is symmetric
    total_delta_i = sum(graph.delta_i .* I1)
    I1 *= param[:K] / total_delta_i  # rescale

    # Smooth network (optional)
    # I1 = smooth_network(param, graph, I1)

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
    I1 = (I1 + I1') / 2  # ensure kappa is symmetric
    total_delta_i = sum(graph.delta_i .* I1)
    I1 *= param[:K] / total_delta_i  # rescale

    return I1
end



# A simple way to perturb the network that simulates shaking the network in some 
# random direction and applying a Gaussian smoothing (see smooth_network()).
function shake_network(param, graph, I0, res, options)
    J = graph.J

    # ===================
    # RETRIEVE PARAMETERS
    # ===================

    smoothing_radius = options[:smoothing_radius]
    mu_perturbation = options[:mu_perturbation]
    sigma_perturbation = options[:sigma_perturbation]

    # ===============
    # PERTURB NETWORK
    # ===============

    theta = rand() * 2 * π
    rho = exp(mu_perturbation + sigma_perturbation * randn()) / exp(sigma_perturbation^2)
    direction = rho * cis(theta)  # cis(theta) is exp(i*theta) in Julia
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
            if options[:preserve_central_symmetry]
                weight .+= exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0 .+ direction_x).^2 .+ (edge_y .- y0 .+ direction_y).^2))
            end
            if options[:preserve_horizontal_symmetry]
                weight .+= exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0 .- direction_x).^2 .+ (edge_y .- y0 .+ direction_y).^2))
            end
            if options[:preserve_vertical_symmetry]
                weight .+= exp.(-0.5 / smoothing_radius^2 .* ((edge_x .- x0 .+ direction_x).^2 .+ (edge_y .- y0 .- direction_y).^2))
            end
            weight[.!feasible_edge] .= 0
            I1[i, neighbor] = sum(I0 .* weight) / sum(weight)
        end
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 = (I1 + I1') / 2  # make sure kappa is symmetric
    total_delta_i = sum(graph.delta_i .* I1)
    I1 *= param[:K] / total_delta_i  # rescale

    return I1
end


"""
    rebranch_network(param, graph, I0, res, options)

This function implements the rebranching algorithm described in the paper.
Links are reshuffled everywhere so that each node is better connected to its best neighbor
(those with the lowest price index for traded goods, i.e., more central places in the trading network).
"""
function rebranch_network(param, graph, I0, res, options)
    J = graph.J

    # ===============
    # PERTURB NETWORK
    # ===============

    I1 = copy(I0)

    # Rebranch each location to its lowest price parent
    for i = 1:J
        neighbors = graph.nodes[i]
        parents = neighbors[ res[:PCj][neighbors] .< res[:PCj][i] ]
        
        if length(parents) >= 2
            lowest_price_parent = sortperm(res[:PCj][parents])[1]
            lowest_price_parent = parents[lowest_price_parent]
            best_connected_parent = sortperm(I0[i, parents], rev=true)[1]
            best_connected_parent = parents[best_connected_parent]
            
            # swap roads
            I1[i, lowest_price_parent] = I0[i, best_connected_parent]
            I1[i, best_connected_parent] = I0[i, lowest_price_parent]
            I1[lowest_price_parent, i] = I0[best_connected_parent, i]
            I1[best_connected_parent, i] = I0[lowest_price_parent, i]
        end
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 = (I1 + I1') / 2  # make sure kappa is symmetric
    total_delta_i = sum(graph.delta_i .* I1)
    I1 *= param[:K] / total_delta_i  # rescale

    return I1
end


"""
    random_rebranch_network(param, graph, I0, res, options)

This function does the same as `rebranch_network` except that only a few nodes
(#NbRandomPerturbations) are selected for rebranching at random.
"""
function random_rebranch_network(param, graph, I0, res, options)
    J = graph.J

    # ===============
    # PERTURB NETWORK
    # ===============

    I1 = copy(I0)

    # Random selection of nodes to rebranch
    link_list = randperm(J)[1:options[:nb_random_perturbations]]

    # Rebranch each selected location to its lowest price parent
    for i in link_list
        neighbors = graph.nodes[i]
        parents = neighbors[res[:PCj][neighbors] .< res[:PCj][i]]
        
        if length(parents) >= 2
            lowest_price_parent_index = sortperm(res[:PCj][parents])[1]
            lowest_price_parent = parents[lowest_price_parent_index]
            best_connected_parent_index = sortperm(I0[i, parents], rev=true)[1]
            best_connected_parent = parents[best_connected_parent_index]
            
            # Swap roads
            I1[i, lowest_price_parent] = I0[i, best_connected_parent]
            I1[i, best_connected_parent] = I0[i, lowest_price_parent]
            I1[lowest_price_parent, i] = I0[best_connected_parent, i]
            I1[best_connected_parent, i] = I0[lowest_price_parent, i]
        end
    end

    # Make sure graph satisfies symmetry and capacity constraint
    I1 = (I1 + I1') / 2  # make sure kappa is symmetric
    total_delta_i = sum(graph.delta_i .* I1)
    I1 *= param[:K] / total_delta_i  # rescale

    return I1
end


"""
    hybrid(param, graph, I0, res, options)

This function attempts to adapt the spirit of Alder (2018)'s algorithm to
delete/add links to the network in a way that blends with our model.
"""
function hybrid(param, graph, I0, res, options)
    # ========================
    # COMPUTE GRADIENT WRT Ijk
    # ========================
    J = graph.J
    grad = zeros(J, J)

    if !param[:cong] # no cross-good congestion
        Pjkn = repeat(permute(res[:Pjn], [1, 3, 2]), outer=[1, J, 1])
        PQ = Pjkn .* res[:Qjkn].^(1 + param[:beta])
        grad = param[:gamma] * graph.delta_tau .* sum(PQ + permute(PQ, [2, 1, 3]), dims=3) .* I0.^-(1 + param[:gamma]) ./ graph.delta_i
        grad[.!graph.adjacency] .= 0
    else # cross-good congestion
        PCj = repeat(res[:PCj], outer=[1, J])
        matm = permutedims(repeat(param[:m], outer=[1, J, J]), [2, 1, 3])
        cost = sum(matm .* res[:Qjkn].^param[:nu], dims=3).^((param[:beta] + 1) / param[:nu])
        PQ = PCj .* cost
        grad = param[:gamma] * graph.delta_tau .* (PQ + PQ') .* I0.^-(1 + param[:gamma]) ./ graph.delta_i
        grad[.!graph.adjacency] .= 0
    end

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

    # ====================
    # COMPUTE NEW GRADIENT
    # ====================
    res = solve_allocation(param, graph, I1) # Assuming this function is defined elsewhere

    if !param[:cong] # no cross-good congestion
        Pjkn = repeat(permute(res[:Pjn], [1, 3, 2]), outer=[1, J, 1])
        PQ = Pjkn .* res[:Qjkn].^(1 + param[:beta])
        grad = param[:gamma] * graph.delta_tau .* sum(PQ + permute(PQ, [2, 1, 3]), dims=3) .* I0.^-(1 + param[:gamma]) ./ graph.delta_i
        grad[.!graph.adjacency] .= 0
    else # cross-good congestion
        PCj = repeat(res[:PCj], outer=[1, J])
        matm = permutedims(repeat(param[:m], outer=[1, J, J]), [2, 1, 3])
        cost = sum(matm .* res[:Qjkn].^param[:nu], dims=3).^((param[:beta] + 1) / param[:nu])
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



