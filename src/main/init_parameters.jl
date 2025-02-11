

"""
    init_parameters(; kwargs...) -> Dict

Returns a `param` dict with the model parameters. These are independent of the graph structure and dimensions.

# Keyword Arguments
- `alpha::Float64=0.5`: Cobb-Douglas coefficient on final good c^alpha * h^(1-alpha)
- `beta::Float64=1`: Parameter governing congestion in transport cost
- `gamma::Float64=1`: Elasticity of transport cost relative to infrastructure
- `K::Float64=1`: Amount of concrete/asphalt (infrastructure budget)
- `sigma::Float64=5`: Elasticity of substitution across goods (CES)
- `rho::Float64=2`: Curvature in utility (c^alpha * h^(1-alpha))^(1-rho)/(1-rho)
- `a::Float64=0.8`: Curvature of the production function L^alpha
- `N::Int64=1`: Number of goods traded in the economy (used for checks in `create_graph()`)
- `labor_mobility::Any=false`: Switch for labor mobility (true/false or 'partial')
- `cross_good_congestion::Bool=false`: Switch for cross-good congestion
- `nu::Float64=2`: Elasticity of substitution b/w goods in transport costs if cross-good congestion
- `m::Vector{Float64}=ones(N)`: Vector of weights Nx1 in the cross congestion cost function
- `annealing::Bool=true`: Switch for the use of annealing at the end of iterations (only if gamma > beta)
- `verbose::Bool=true`: Switch to turn on/off text output (from Ipopt or other optimizers)
- `duality::Bool=true`: Switch to turn on/off duality whenever available (fixed labor and beta <= 1). 
- `warm_start::Bool=true`: Use the previous solution as a warm start for the next iteration
- `kappa_min::Float64=1e-5`: Minimum value for road capacities K
- `min_iter::Int64=20`: Minimum number of iterations
- `max_iter::Int64=200`: Maximum number of iterations
- `tol::Float64=1e-5`: Tolerance for convergence of road capacities K

# Optional Parameters
- `optimizer = Ipopt.Optimizer`: Optimizer to be used
- `optimizer_attr::Dict`: Dict of attributes passed to the optimizer (e.g. `Dict(:tol => 1e-5)`)
- `model_attr::Dict`: Dict of tuples (length 2) passed to the model (e.g. `Dict(:backend => (MOI.AutomaticDifferentiationBackend(), MathOptSymbolicAD.DefaultBackend()))` to use Symbolic AD)
- `model::Function`: For custom models => a function that taks an optimizer and an 'auxdata' structure as created by `create_auxdata()` as input and returns a fully parameterized JuMP model
- `recover_allocation::Function`: For custom models => a function that takes a solved JuMP model and 'auxdata' structure as input and returns the allocation variables. In particular, it should return a dict with symbol keys returning at least objects :welfare => scalar welfare measure, :Pjn => prices, :PCj => aggregate consumption, and :Qjkn => flows. 

# Examples
```julia
param = init_parameters(K = 10, labor_mobility = true)
```
"""
function init_parameters(; alpha = 0.5, beta = 1, gamma = 1, K = 1, sigma = 5, rho = 2, a = 0.8, N = 1, 
                         labor_mobility = false, cross_good_congestion=false, nu = 2, m = ones(N), 
                         annealing=true, 
                         verbose = true, duality = true, warm_start = true, 
                         kappa_min = 1e-5, min_iter = 20, max_iter = 200, tol = 1e-5, kwargs...)
    param = Dict()

    if !isempty(kwargs)
        for (key, value) in kwargs
            println("Non-standard parameter '$(key)' added")
            param[key] = value
        end
    end
    if haskey(param, :labour_mobility)
        labor_mobility = param[:labour_mobility]
    end

    param[:gamma] = gamma
    param[:alpha] = alpha
    param[:beta] = beta
    param[:K] = K
    param[:sigma] = sigma
    param[:rho] = rho
    param[:a] = a
    param[:nu] = nu
    if length(m) != N
        error("m must have length N = $N, but has length $(length(m))")
    end
    param[:N] = N
    param[:m] = m
    if labor_mobility === "partial" || labor_mobility === 0.5
        param[:mobility] = 0.5
    else
        param[:mobility] = labor_mobility === true
    end
    if param[:mobility] == true
        param[:rho] = 0;
    end
    param[:cong] = cross_good_congestion
    param[:annealing] = annealing
    param[:verbose] = verbose
    param[:duality] = duality
    param[:warm_start] = warm_start

    # Additional parameters for the numerical part
    param[:tol] = tol
    param[:kappa_min] = kappa_min
    param[:min_iter] = min_iter
    param[:max_iter] = max_iter

    # Define utility function, marginal utility of consumtion and inverse
    param[:u] = (c, h) -> ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^(1 - rho) / (1 - rho)
    param[:uprime] = (c, h) -> ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^-rho * ((c / alpha)^(alpha - 1) * (h / (1 - alpha))^(1 - alpha))
    param[:usecond] = (c, h) -> -rho * ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^(-rho - 1) * ((c / alpha)^(alpha - 1) * (h / (1 - alpha))^(1 - alpha))^2 + (alpha - 1) / alpha * ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^-rho * ((c / alpha)^(alpha - 2) * (h / (1 - alpha))^(1 - alpha))
    param[:uprimeinv] = (x, h) -> alpha * x^(-1 / (1 + alpha * (rho - 1))) * (h / (1 - alpha))^((1 - alpha) * (1 - rho) / (1 + alpha * (rho - 1)))
    param[:uprimeinvprime] = (x, h) -> -alpha / (1 + alpha * (rho - 1)) * x^(-1 / (1 + alpha * (rho - 1)) - 1) * (h / (1 - alpha))^((1 - alpha) * (1 - rho) / (1 + alpha * (rho - 1)))
    
    # Define production function 
    param[:F] = (L, a) -> L^a
    param[:Fprime] = (L, a) -> a * L^(a - 1)

    # # CHECK CONSISTENCY WITH ENDOWMENTS/PRODUCTIVITY (if applicable)
    # check_graph_param(param)

    return param
end

# Check if the parameters are consistent with the graph structure
function check_graph_param(graph, param)

    if haskey(param, :m) && length(param[:m]) != param[:N]
        @warn "m does not have the right length N = $(param[:N])."
    end

    if haskey(param, :omegaj) && !haskey(graph, :omegaj)
        graph[:omegaj] = param[:omegaj]
    end
    if haskey(graph, :omegaj) && length(graph[:omegaj]) != graph[:J]
        @warn "omegaj does not have the right length J = $(graph[:J])."
    end

    if haskey(param, :omegar) && !haskey(graph, :omegar)
        graph[:omegar] = param[:omegar]
    end
    if haskey(graph, :omegar) && length(graph[:omegar]) != graph[:nregions]
        @warn "omegar does not have the right length nregions = $(graph[:nregions])."
    end

    if haskey(param, :Lr) && !haskey(graph, :Lr)
        graph[:Lr] = param[:Lr]
    end
    if haskey(graph, :Lr) && length(graph[:Lr]) != graph[:nregions]
        @warn "Lr does not have the right length nregions = $(graph[:nregions])."
    end

    if haskey(param, :Lj) && !haskey(graph, :Lj) && param[:mobility] == 0
        graph[:Lj] = param[:Lj]
    end
    if haskey(graph, :Lj) && param[:mobility] == 0 && length(graph[:Lj]) != graph[:J]
        @warn "Lj does not have the right length J = $(graph[:J])."
    end
    
    if haskey(param, :Zjn) && !haskey(graph, :Zjn)
        graph[:Zjn] = param[:Zjn]
    end
    if haskey(graph, :Zjn) && size(graph[:Zjn]) != (graph[:J], param[:N])
        @warn "Zjn does not have the right size J ($(graph[:J])) x N ($(param[:N]))."
    end

    if haskey(param, :Hj) && !haskey(graph, :Hj)
        graph[:Hj] = param[:Hj]
    end
    if haskey(graph, :Hj) && length(graph[:Hj]) != graph[:J]
        @warn "Hj does not have the right length J = $(graph[:J])."
    end

    if haskey(param, :hj) && !haskey(graph, :hj)
        graph[:hj] = param[:hj]
    end
    if haskey(graph, :hj) && length(graph[:hj]) != graph[:J]
        @warn "hj does not have the right length J = $(graph[:J])."
    end
end