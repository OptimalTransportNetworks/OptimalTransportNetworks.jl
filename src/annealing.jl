
using Random
using LinearAlgebra
using Optim
using Statistics
using Distributions

function annealing(param, graph, I0, options)
    J = graph.J
    gamma = param.gamma
    verbose = false
    TOL_I_BOUNDS = 1e-7

    Iu = options.Iu
    Il = options.Il

    if param.mobility || param.beta > 1
        Il = max(1e-6 * graph.adjacency, Il)
    end

    T = options.t_start
    T_min = options.t_end
    T_step = options.t_step
    nb_network_deepening = options.nb_deepening

    funcs = options.funcs

    best_score = -Inf
    best_I = I0

    old_score = best_score
    weight_old = 0.5
    counter = 0
    I1 = I0

    while T > T_min
        accepted = false

        if counter > 0
            I1 = perturbate(param, graph, I0, results, options)
        end

        k = 0
        x = x0
        while k <= nb_network_deepening - 1
            auxdata = create_auxdata(param, graph, I1)
            results, flag, x = solve_allocation_handle(x, auxdata, funcs, verbose)

            score = results.welfare

            if (!any(flag.status in [0, 1]) || isnan(score)) && param.verbose
                println("Optimization failed! k=$k, return flag=$flag.status")
                k = nb_network_deepening - 1
                score = -Inf
            end

            if score > best_score
                best_results = results
                best_I = I1
                best_score = score
            end

            if k < nb_network_deepening - 1
                if !param.cong
                    Pjkn = repeat(permutedims(results.Pjn, [1, 3, 2]), outer=[1, graph.J, 1])
                    PQ = Pjkn .* results.Qjkn .^ (1 + param.beta)
                    I1 = (graph.delta_tau ./ graph.delta_i .* sum(PQ + permutedims(PQ, [2, 1, 3]), dims=3)) .^ (1 / (1 + param.gamma))
                    I1[graph.adjacency .== false] .= 0
                else
                    PCj = repeat(results.PCj, outer=[1, graph.J])
                    matm = permutedims(repeat(param.m, outer=[1, graph.J, graph.J]), 1)
                    cost = sum(matm .* results.Qjkn .^ param.nu, dims=3) .^ ((param.beta + 1) / param.nu)
                    PQ = PCj .* cost
                    I1 = (graph.delta_tau ./ graph.delta_i .* (PQ + PQ')) .^ (1 / (param.gamma + 1))
                    I1[graph.adjacency .== false] .= 0
                end

                I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2, 1]))

                distance_lb = max(max(Il .- I1), 0)
                distance_ub = max(max(I1 .- Iu), 0)
                counter_rescale = 0
                while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
                    I1 = max(min(I1, Iu), Il)
                    I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2, 1]))

                    distance_lb = max(max(Il .- I1), 0)
                    distance_ub = max(max(I1 .- Iu), 0)
                    counter_rescale += 1
                end

                if counter_rescale == 100
                    println("Warning: road capacities I1 had to be rescaled $counter_rescale times to fit the bounds!")
                end
                



        
