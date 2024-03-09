
# using Ipopt

function solve_allocation_mobility_ADiGator(x0, auxdata, funcs, verbose=true)

    graph = auxdata.graph
    param = auxdata.param

    if any(sum(param[:Zjn] .> 0, dims=2) .> 1) && param[:a] == 1
        error("Version of the code with more than 1 good per location and a=1 not supported yet.")
    end

    if isempty(x0)
        x0 = vcat(0, 1e-6 * ones(graph.J * param[:N]), zeros(graph.ndeg * param[:N]), 1 / graph.J * ones(graph.J), 1 / (graph.J * param[:N]) * ones(graph.J * param[:N]))
    end

    # -> could do using forward diff.
    # funcs[:objective] = (x) -> objective_mobility(x, auxdata)
    # funcs[:constraint] = (x) -> constraints_mobility(x, auxdata)
    # funcs[:gradient] = (x) -> objective_mobility_Grd(x, auxdata)
    # funcs[:jacobian] = (x) -> constraints_mobility_Jac(x, auxdata)
    # funcs[:hessian] = (x, sigma, lambda) -> objective_mobility_Hes(x, auxdata, sigma, lambda)
    objective = (x) -> objective_mobility(x, auxdata)
    constraints = (x) -> constraints_mobility(x, auxdata)
    gradient = (x) -> ForwardDiff.gradient(objective, x)
    jacobian = (x) -> ForwardDiff.jacobian(constraint, x)
    # hessian = (x) -> ForwardDiff.hessian(objective, x)

    ## Using FastDiffferentiation 
    # make_variables(:x,n)
    # f_exe! = make_function(objective, :x)
    # FastDifferentiation.hessian(objective, x)

    options = Dict(
        # Bounds on x
        "lb" => vcat(-Inf, 1e-8 * ones(graph.J * param[:N]), -Inf * ones(graph.ndeg * param[:N]), 1e-8 * ones(graph.J), 1e-8 * ones(graph.J * param[:N])),
        "ub" => vcat(Inf, Inf * ones(graph.J * param[:N]), Inf * ones(graph.ndeg * param[:N]), ones(graph.J), Inf * ones(graph.J * param[:N])),
        # On the constraints
        "cl" => vcat(-Inf * ones(graph.J), -Inf * ones(graph.J * param[:N]), -1e-8, -1e-8 * ones(graph.J)),
        "cu" => vcat(-1e-8 * ones(graph.J), -1e-8 * ones(graph.J * param[:N]), 1e-8, 1e-8 * ones(graph.J))
    )

    if verbose
        options[:print_level] = 5
    else
        options[:print_level] = 0
    end

    ## Old Matlab solution:
    # x, info = ipopt(x0, funcs, options)

    ## Ipopt Solution:
    n = funcs[:numvar]
    m = length(options["cu"])
    eval_g, eval_jac_g, eval_h = ipopt_prepare(x0) # defined in call_adigator.jl
    prob = Ipopt.CreateIpoptProblem(n, options["lb"], options["ub"], m, options["cl"], options["cu"],
                                    m^2, n^2, # dense formulation
                                    objective, constraint, eval_g, eval_jac_g, nothing) # eval_h
    prob.x = x0
    solvestat = Ipopt.IpoptSolve(prob)

    # TODO: try unrolling constraints? Use DynamicExpressions.jl? or Enzyme.jl?
    # or just use JuMP?

    # u = x[1]
    # Cjn = reshape(x[2:graph.J*param[:N]+1], graph.J, param[:N])
    # Qin = reshape(x[graph.J*param[:N]+2:graph.J*param[:N]+graph.ndeg*param[:N]+1], graph.ndeg, param[:N])
    # Lj = x[graph.J*param[:N]+graph.ndeg*param[:N]+2:graph.J*param[:N]+graph.ndeg*param[:N]+graph.J+1]
    # Cj = (sum(Cjn .^ ((param[:sigma] - 1) / param[:sigma]), dims=2)) .^ (param[:sigma] / (param[:sigma] - 1))
    # Ljn = reshape(x[graph.J*param[:N]+graph.ndeg*param[:N]+graph.J+2:end], graph.J, param[:N])
    # Yjn = param[:Zjn] .* Ljn .^ param[:a]


    # # JuMP Solution: see also build_model() function under constraints_mobility and thread at
    # # https://discourse.julialang.org/t/nonlinear-optimization-with-many-constraints-autodifferentiation-which-julia-solution/110678/15
    # model = Model(Ipopt.Optimizer)
    model = build_model(auxdata)
    # set_optimizer_attribute(model, "tol", 1e-5)  # For example, this sets the tolerance to 1e-6
    set_optimizer_attribute(model, "max_iter", 1000) 
    # set_attribute(model, "linear_solver", "paradiso")
    set_attribute(model,
      MOI.AutomaticDifferentiationBackend(),
      MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    # @variable(model, x[1:n])
    # # Register the objective and constraint functions with JuMP
    # # JuMP.register(model, :objective, n, objective, autodiff=true)
    # # JuMP.register(model, :constraint, n, constraint, autodiff=true)
    # # @objective(model, Min, objective(x))
    # @objective(model, Min, objective(x))
    # # for i in 1:length(constraint(x))
    # #     @constraint(model, constraint(x...)[i] <= 0)
    # # end
    # # @constraint(model, con[i=1:n], constraint(x)[i] <= 0)
    # # @constraint(model, constraint(x[1:n]) .<= 0)
    # @constraint(model, constraint(x...) .<= 0)
    # optimize!(model)
    # # x = value.(x)
    # # info = termination_status(model)

    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED
        # Create a dictionary to hold the variable names and their optimal values
        results = Dict(
            "u" => value(u),
            "Cjn" => value.(Cjn),
            "Qin" => value.(Qin),
            "Lj" => value.(Lj),
            "Ljn" => value.(Ljn)
        )
    
        # Populate the dictionary with variable names and their optimal values
        for var in all_variables(model)
            var_name = name(var)
            results[var_name] = value(var)
        end
    
        # Now `results` holds the optimal values for all variables
    else
        error("The solver did not find an optimal solution.")
    end

    # model = build_model(auxdata)
    # param = dict_to_namedtuple(auxdata[:param])
    # graph = auxdata[:graph]
    # kappa_ex = auxdata[:kappa_ex]
    # A = auxdata[:A]
    # optimize!(model)

    ## Try using Nonconvex.jl
    # using Nonconvex
    # import Nonconvex
    # using NonconvexIpopt
    # model = Nonconvex.Model(objective)
    # Nonconvex.addvar!(model, options["lb"], options["ub"], init = x0, integer = repeat([false], n))
    # Nonconvex.add_ineq_constraint!(model, constraint)
    # alg = IpoptAlg()
    # opts = IpoptOptions()
    # sp_model = Nonconvex.sparsify(model, hessian = true)
    # model = Nonconvex.forwarddiffy(constraint)
    # r = Nonconvex.optimize(model, alg, x0, options = opts)
    # r.minimum # objective value
    # r.minimizer # decision variables

    ## Try using ADNLPModels.jl
    # import ADNLPModels
    # nlp = ADNLPModels.ADNLPModel(objective, x0, options["lb"], options["ub"], constraint, options["cl"], options["cu"]) 

    flag = info
    results = recover_allocation_mobility_ADiGator(x, auxdata)
    results.Pjn = reshape(info.lambda[graph.J+1:graph.J+graph.J*param.N], graph.J, param.N)
    results.PCj = sum(results.Pjn.^(1-param.sigma), dims=2).^(1/(1-param.sigma))

    return results, flag, x
end

function recover_allocation_mobility_ADiGator(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph

    results = Dict()
    results["welfare"] = x[1]
    results["Cjn"] = reshape(x[2:graph.J*param.N+1], graph.J, param.N)
    results["Cj"] = sum(results["Cjn"].^((param.sigma-1)/param.sigma), dims=2).^(param.sigma/(param.sigma-1))
    results["Lj"] = x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+graph.ndeg*param.N+graph.J+1]
    results["Ljn"] = reshape(x[graph.J*param.N+graph.ndeg*param.N+graph.J+2:end], graph.J, param.N)
    results["Yjn"] = param.Zjn .* results["Ljn"].^param.a
    results["cj"] = results["Cj"] ./ results["Lj"]
    results["cj"][results["Lj"].==0] = 0
    results["hj"] = param.Hj ./ results["Lj"]
    results["hj"][results["Lj"].==0] = 0
    results["uj"] = param.u(results["cj"], results["hj"])
    results["Qin"] = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)

    results["Qjkn"] = zeros(graph.J, graph.J, param.N)
    id = 1
    for i in 1:graph.J
        for j in 1:length(graph.nodes[i])
            if graph.nodes[i][j] > i
                results["Qjkn"][i, graph.nodes[i][j], :] = max.(results["Qin"][id, :], 0)
                results["Qjkn"][graph.nodes[i][j], i, :] = max.(-results["Qin"][id, :], 0)
                id += 1
            end
        end
    end

    return results
end


# Please note that the functions `objective_mobility`, `objective_mobility_Grd`, `constraints_mobility`, `constraints_mobility_Jac`, and `objective_mobility_Hes` are not provided in the original MATLAB code, so they are not translated here. You will need to provide these functions in Julia. Also, the `ipopt` function in Julia may have a different syntax and usage than the `ipopt` function in MATLAB. You may need to adjust the code accordingly.