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
function annealing(param, graph, I0, varargin)
    J = graph.J
    gamma = param.gamma
    verbose = false
    TOL_I_BOUNDS = 1e-7

    options = retrieve_options(param, graph, varargin)
    Iu = options.Iu
    Il = options.Il

    if param.mobility || param.beta > 1
        Il = max(1e-6 * graph.adjacency, Il)
    end

    T = options.t_start
    T_min = options.t_end
    T_step = options.t_step
    nb_network_deepening = options.nb_deepening

    switch options.perturbation_method
        case 'random'
            perturbate = random_perturbation(param, graph, I0, res, options)
        case 'shake'
            perturbate = shake_network(param, graph, I0, res, options)
        case 'rebranching'
            perturbate = rebranch_network(param, graph, I0, res, options)
        case 'random rebranching'
            perturbate = random_rebranch_network(param, graph, I0, res, options)
        case 'hybrid alder'
            perturbate = hybrid(param, graph, I0, res, options)
        otherwise
            error('unknown perturbation method')
    end

    x0 = []
    if param.custom
        x0 = [1e-6 * ones(graph.J * param.N, 1); zeros(graph.ndeg * param.N, 1); sum(param.Lj) / (graph.J * param.N) * ones(graph.J * param.N, 1)]
    end

    funcs = options.funcs
    if param.adigator && isempty(funcs)
        funcs = call_adigator(param, graph, I0, verbose)
    end

    best_score = -Inf
    best_I = I0

    old_score = best_score
    weight_old = 0.5
    counter = 0
    I1 = I0

    acceptance_str = ["rejected", "accepted"]

    while T > T_min
        accepted = false

        if counter > 0
            I1 = perturbate(param, graph, I0, results)
        end

        if options.display
            plot_graph(param, graph, I1)
        end

        k = 0
        x = x0
        while k <= nb_network_deepening - 1
            auxdata = create_auxdata(param, graph, I1)
            results, flag, x = solve_allocation_handle(x, auxdata, funcs, verbose)
            score = results.welfare

            if (~any(flag.status == [0, 1]) || isnan(score)) && param.verbose
                fprintf('optimization failed! k=%d, return flag=%d\n', k, flag.status)
                k = nb_network_deepening - 1
                score = -Inf
            end

            if score > best_score
                best_results = results
                best_I = I1
                best_score = score
            end

            if k < nb_network_deepening - 1
                if ~param.cong
                    Pjkn = repmat(permute(results.Pjn, [1 3 2]), [1 graph.J 1])
                    PQ = Pjkn .* results.Qjkn.^(1 + param.beta)
                    I1 = (graph.delta_tau ./ graph.delta_i .* sum(PQ + permute(PQ, [2 1 3]), 3)).^(1 / (1 + param.gamma))
                    I1(graph.adjacency == false) = 0
                else
                    PCj = repmat(results.PCj, [1 graph.J])
                    matm = shiftdim(repmat(param.m, [1 graph.J graph.J]), 1)
                    cost = sum(matm .* results.Qjkn.^param.nu, 3).^((param.beta + 1) / param.nu)
                    PQ = PCj .* cost
                    I1 = (graph.delta_tau ./ graph.delta_i .* (PQ + PQ')).^(1 / (param.gamma + 1))
                    I1(graph.adjacency == false) = 0
                end

                if param.custom
                    #  enter here how to update the infrastructure network I1 (if needed) in the custom case
                end

                I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))

                distance_lb = max(max(Il(:) - I1(:)), 0)
                distance_ub = max(max(I1(:) - Iu(:)), 0)
                counter_rescale = 0
                while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
                    I1 = max(min(I1, Iu), Il)
                    I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))
                    distance_lb = max(max(Il(:) - I1(:)), 0)
                    distance_ub = max(max(I1(:) - Iu(:)), 0)
                    counter_rescale = counter_rescale + 1
                end

                if counter_rescale == 100 && distance_lb + distance_ub > param.tol_kappa && param.verbose
                    fprintf('Warning! Could not impose bounds on network properly.\n')
                end
            end
            k = k + 1
        end

        acceptance_prob = exp(1e4 * (score - old_score) / (abs(old_score) * T))
        if rand() <= acceptance_prob || counter == 0
            I0 = I1
            old_score = score
            accepted = true
        end

        if options.display
            plot_graph(param, graph, I1)
        end

        if param.verbose
            fprintf('Iteration No.%d, score=%d, %s (best=%d)\n', counter, score, acceptance_str[1 + Int(accepted)], best_score)
        end

        counter = counter + 1
        T = T * T_step
    end

    has_converged = false
    I0 = best_I
    while !has_converged && counter < 100
        auxdata = create_auxdata(param, graph, I0)
        results, flag, x = solve_allocation_handle(x0, auxdata, funcs, verbose)

        score = results.welfare

        if options.display
            plot_graph(param, graph, I0, "Sizes", results.Lj)
        end

        if score > best_score
            best_results = results
            best_I = I0
            best_score = score
        end

        if !param.cong
            Pjkn = repmat(permute(results.Pjn, [1 3 2]), [1 graph.J 1])
            PQ = Pjkn .* results.Qjkn.^(1 + param.beta)
            I1 = (graph.delta_tau ./ graph.delta_i .* sum(PQ + permute(PQ, [2 1 3]), 3)).^(1 / (1 + param.gamma))
            I1(graph.adjacency == false) = 0
        else
            PCj = repmat(results.PCj, [1 graph.J])
            matm = shiftdim(repmat(param.m, [1 graph.J graph.J]), 1)
            cost = sum(matm .* results.Qjkn.^param.nu, 3).^((param.beta + 1) / param.nu)
            PQ = PCj .* cost
            I1 = (graph.delta_tau ./ graph.delta_i .* (PQ + PQ')).^(1 / (param.gamma + 1))
            I1(graph.adjacency == false) = 0
        end

        if param.custom
            # enter here how to update the infrastructure network I1 (if needed) in the custom case
        end

        I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))

        distance_lb = max(max(Il(:) - I1(:)), 0)
        distance_ub = max(max(I1(:) - Iu(:)), 0)
        counter_rescale = 0
        while distance_lb + distance_ub > TOL_I_BOUNDS && counter_rescale < 100
            I1 = max(min(I1, Iu), Il)
            I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))
            distance_lb = max(max(Il(:) - I1(:)), 0)
            distance_ub = max(max(I1(:) - Iu(:)), 0)
            counter_rescale = counter_rescale + 1
        end

        if counter_rescale == 100 && distance_lb + distance_ub > param.tol_kappa && param.verbose
            fprintf('Warning! Could not impose bounds on network properly.\n')
        end

        distance = sum(abs(I1(:) - I0(:))) / (J^2)
        I0 = weight_old * I0 + (1 - weight_old) * I1
        has_converged = distance < param.tol_kappa
        counter = counter + 1

        if param.verbose
            fprintf('Iteration No.%d - final iterations - distance=%d - welfare=%d\n', counter, distance, score)
        end
    end

    if param.verbose
        fprintf('Welfare = %d\n', score)
    end

    best_results.Ijk = best_I
end

function options = retrieve_options(param, graph, args)
    p = inputParser()

    checkSwitch = @(x) any(validatestring(x, ["on", "off"]))

    addParameter(p, "PerturbationMethod", "random rebranching", @(x) any(validatestring(x, ["shake", "random", "rebranching", "random rebranching", "hybrid alder"])))
    addParameter(p, "PreserveCentralSymmetry", "off", checkSwitch)
    addParameter(p, "PreserveVerticalSymmetry", "off", checkSwitch)
    addParameter(p, "PreserveHorizontalSymmetry", "off", checkSwitch)
    addParameter(p, "SmoothingRadius", 0.25, @isnumeric)
    addParameter(p, "MuPerturbation", log(0.3), @isnumeric)
    addParameter(p, "SigmaPerturbation", 0.05, @isnumeric)
    addParameter(p, "Display", "off", checkSwitch)
    addParameter(p, "TStart", 100, @isnumeric)
    addParameter(p, "TEnd", 1, @isnumeric)
    addParameter(p, "TStep", 0.9, @isnumeric)
    addParameter(p, "NbDeepening", 4, @isnumeric)
    addParameter(p, "NbRandomPerturbations", 1, @isnumeric)
    addParameter(p, "Funcs", [])
    addParameter(p, "Iu", Inf * ones(graph.J, graph.J), @(x) size(x) == [graph.J graph.J])
    addParameter(p, "Il", zeros(graph.J, graph.J), @(x) size(x) == [graph.J graph.J])

    parse(p, args...)

    options.perturbation_method = p.Results.PerturbationMethod
    options.preserve_central_symmetry = p.Results.PreserveCentralSymmetry == "on"
    options.preserve_vertical_symmetry = p.Results.PreserveVerticalSymmetry == "on"
    options.preserve_horizontal_symmetry = p.Results.PreserveHorizontalSymmetry == "on"
    options.smoothing_radius = p.Results.SmoothingRadius
    options.mu_perturbation = p.Results.MuPerturbation
    options.sigma_perturbation = p.Results.SigmaPerturbation
    options.display = p.Results.Display == "on"
    options.t_start = p.Results.TStart
    options.t_end = p.Results.TEnd
    options.t_step = p.Results.TStep
    options.nb_deepening = round(p.Results.NbDeepening)
    options.nb_random_perturbations = round(p.Results.NbRandomPerturbations)
    options.funcs = p.Results.Funcs
    options.Iu = p.Results.Iu
    options.Il = p.Results.Il
end

function I1 = random_perturbation(param, graph, I0, res, options)
    size_perturbation = 0.1 * param.K / graph.J

    I1 = I0

    link_list = randperm(graph.J, options.nb_random_perturbations)

    for i = link_list
        j = randi(length(graph.nodes[i].neighbors))
        I1(i, graph.nodes[i].neighbors(j)) = size_perturbation * exp(randn()) / exp(0.5)
    end

    I1 = (I1 + I1') / 2
    I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))
end

function I1 = shake_network(param, graph, I0, res, options)
    J = graph.J

    smoothing_radius = options.smoothing_radius
    mu_perturbation = options.mu_perturbation
    sigma_perturbation = options.sigma_perturbation

    I1 = zeros(J, J)
    vec_x = graph.x
    vec_y = graph.y
    feasible_edge = graph.adjacency

    edge_x = 0.5 * (vec_x[:, ones(J, 1)] + ones(J, 1) * vec_x')
    edge_y = 0.5 * (vec_y[:, ones(J, 1)] + ones(J, 1) * vec_y')

    theta = rand() * 2 * pi
    rho = exp(mu_perturbation + sigma_perturbation * randn()) / exp(sigma_perturbation^2)
    direction = rho * exp(1im * theta)
    direction_x = real(direction)
    direction_y = imag(direction)

    for i = 1:J
        for j = 1:length(graph.nodes[i].neighbors)
            x0 = edge_x[i, graph.nodes[i].neighbors(j)]
            y0 = edge_y[i, graph.nodes[i].neighbors(j)]

            weight = exp(-0.5 / smoothing_radius^2 * ((edge_x - x0).^2 + (edge_y - y0).^2))
            if options.preserve_central_symmetry
                weight = weight + exp(-0.5 / smoothing_radius^2 * ((edge_x - x0 + direction_x).^2 + (edge_y - y0 + direction_y).^2))
            end
            if options.preserve_horizontal_symmetry
                weight = weight + exp(-0.5 / smoothing_radius^2 * ((edge_x - x0 - direction_x).^2 + (edge_y - y0 + direction_y).^2))
            end
            if options.preserve_vertical_symmetry
                weight = weight + exp(-0.5 / smoothing_radius^2 * ((edge_x - x0 + direction_x).^2 + (edge_y - y0 - direction_y).^2))
            end
            weight(~feasible_edge) = 0
            I1(i, graph.nodes[i].neighbors(j)) = sum(sum(I0 .* weight)) / sum(weight(:))
        end
    end

    I1 = (I1 + I1') / 2
    I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))
end

function I1 = rebranch_network(param, graph, I0, res, options)
    J = graph.J

    I1 = I0

    for i = 1:graph.J
        neighbors = graph.nodes[i].neighbors
        parents = neighbors(res.PCj(neighbors) < res.PCj(i))

        if length(parents) >= 2
            [~, lowest_price_parent] = sort(res.PCj(parents))
            lowest_price_parent = parents(lowest_price_parent(1))
            [~, best_connected_parent] = sort(I0(i, parents), "descend")
            best_connected_parent = parents(best_connected_parent(1))

            I1(i, lowest_price_parent) = I0(i, best_connected_parent)
            I1(i, best_connected_parent) = I0(i, lowest_price_parent)
            I1(lowest_price_parent, i) = I0(i, best_connected_parent)
            I1(best_connected_parent, i) = I0(i, lowest_price_parent)
        end
    end

    I1 = (I1 + I1') / 2
    I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))
end

function I1 = random_rebranch_network(param, graph, I0, res, options)
    J = graph.J

    I1 = I0

    link_list = randperm(graph.J, options.nb_random_perturbations)

    for i = link_list
        neighbors = graph.nodes[i].neighbors
        parents = neighbors(res.PCj(neighbors) < res.PCj(i))

        if length(parents) >= 2
            [~, lowest_price_parent] = sort(res.PCj(parents))
            lowest_price_parent = parents(lowest_price_parent(1))
            [~, best_connected_parent] = sort(I0(i, parents), "descend")
            best_connected_parent = parents(best_connected_parent(1))

            I1(i, lowest_price_parent) = I0(i, best_connected_parent)
            I1(i, best_connected_parent) = I0(i, lowest_price_parent)
            I1(lowest_price_parent, i) = I0(i, best_connected_parent)
            I1(best_connected_parent, i) = I0(i, lowest_price_parent)
        end
    end

    I1 = (I1 + I1') / 2
    I1 = param.K * I1 / sum(reshape(graph.delta_i .* I1, [graph.J^2 1]))
end

function I2 = hybrid(param, graph, I0, res, options)
    if ~param.cong
        Pjkn = repmat(permute(res.Pjn, [1 3 2]), [1 graph.J 1])
        PQ = Pjkn .* res.Qjkn.^(1 + param.beta)
        grad = param.gamma * graph.delta_tau .* sum(PQ + permute(PQ, [2 1 3]), 3) .* I0.^-(1 + param.gamma) ./ graph.delta_i
        grad(graph.adjacency == false) = 0
    else
        PCj = repmat(results.PCj, [1 graph.J])
        matm = shiftdim(repmat(param.m, [1 graph.J graph.J]), 1)
        cost = sum(matm .* results.Qjkn.^param.nu, 3).^((param.beta + 1) / param.nu)
        PQ = PCj .* cost
        grad = param.gamma * graph.delta_tau .* (PQ + PQ') .* I0.^-(1 + param.gamma) ./ graph.delta_i
        grad(graph.adjacency == false) = 0
    end

    I1 = I0

    nremove = ceil(0.05 * graph.ndeg)
    [~, id] = sort(grad(tril(graph.adjacency == 1)), "ascend")
    remove_list = id(1:nremove)
    id = 1
    for j = 1:graph.J
        for k = 1:length(graph.nodes[j].neighbors)
            if any(id .== remove_list)
                I1(j, k) = 1e-6
                I1(k, j) = 1e-6
            end
            id = id + 1
        end
    end

    res = solve_allocation(param, graph, I1)
    if ~param.cong
        Pjkn = repmat(permute(res.Pjn, [1 3 2]), [1 graph.J 1])
        PQ = Pjkn .* res.Qjkn.^(1 + param.beta)
        grad = param.gamma * graph.delta_tau .* sum(PQ + permute(PQ, [2 1 3]), 3) .* I0.^-(1 + param.gamma) ./ graph.delta_i
        grad(graph.adjacency == false) = 0
    else
        PCj = repmat(results.PCj, [1 graph.J])
        matm = shiftdim(repmat(param.m, [1 graph.J graph.J]), 1)
        cost = sum(matm .* results.Qjkn.^param.nu, 3).^((param.beta + 1) / param.nu)
        PQ = PCj .* cost
        grad = param.gamma * graph.delta_tau .* (PQ + PQ') .* I0.^-(1 + param.gamma) ./ graph.delta_i
        grad(graph.adjacency == false) = 0
    end

    I2 = I1

    [~, id] = sort(grad(tril(graph.adjacency == 1)), "descend")
    add_link = id(1)

    id = 1
    for j = 1:graph.J
        for k = 1:length(graph.nodes[j].neighbors)
            if id == add_link
                I2(j, k) = param.K / (2 * graph.ndeg)
                I2(k, j) = param.K / (2 * graph.ndeg)
            end
            id = id + 1
        end
    end

    I2 = (I2 + I2') / 2
    I2 = param.K * I2 / sum(reshape(graph.delta_i .* I2, [graph.J^2 1]))
end