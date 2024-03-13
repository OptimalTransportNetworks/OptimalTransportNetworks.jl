
# ==============================================================
#  OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
#  by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
# ================================================= version 1.0.4

"""
    init_parameters(; kwargs...)

Returns a `param` structure with the model parameters.

# Arguments
- `alpha::Float64=0.5`: Cobb-Douglas coefficient on final good c^alpha * h^(1-alpha)
- `beta::Float64=1`: parameter governing congestion in transport cost
- `gamma::Float64=1`: elasticity of transport cost relative to infrastructure
- `K::Float64=1`: amount of concrete/asphalt
- `sigma::Float64=5`: elasticity of substitution across goods (CES)
- `rho::Float64=2`: curvature in utility (c^alpha * h^(1-alpha))^(1-rho)/(1-rho)
- `a::Float64=0.8`: curvature of the production function L^alpha
- `m::Vector{Float64}=ones(N,1)`: vector of weights Nx1 in the cross congestion cost function
- `N::Int64=1`: number of goods
- `nu::Float64=1`: elasticity of substitution b/w goods in transport costs if cross-good congestion
- `labor_mobility::String="off"`: switch for labor mobility ('on'/'off'/'partial')
- `cross_good_congestion::Bool=false`: switch for cross-good congestion
- `annealing::Bool=true`: switch for the use of annealing at the end of iterations
- `custom::Bool=false`: switch for the use of custom lagrangian function
- `verbose::Bool=true`: switch to turn on/off text output
- `duality::Bool=true`: switch to turn on/off duality whenever available
"""
function init_parameters(; alpha=0.5, beta=1, gamma=1, K=1, sigma=5, rho=2, a=0.8, N=1, m=ones(N,1), nu=1, 
                         labor_mobility=false, cross_good_congestion=false, annealing=true, custom=false, 
                         verbose=true, duality=true, warm_start = false, 
                         kappa_tol = 1e-7, kappa_min = 1e-5, kappa_min_iter = 20, kappa_max_iter = 200,
                         kwargs...)
    param = Dict()

    if !isempty(kwargs)
        for (key, value) in kwargs
            param[key] = value
        end
    end

    param[:gamma] = gamma
    param[:alpha] = alpha
    param[:beta] = beta
    param[:K] = K
    param[:sigma] = sigma
    param[:rho] = rho
    param[:a] = a
    param[:m] = m
    param[:N] = N
    param[:nu] = nu
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
    param[:custom] = custom
    param[:verbose] = verbose
    param[:duality] = duality
    param[:warm_start] = warm_start

    # Additional parameters for the numerical part
    param[:kappa_tol] = kappa_tol
    param[:kappa_min] = kappa_min
    param[:kappa_min_iter] = kappa_min_iter
    param[:kappa_max_iter] = kappa_max_iter

    if param[:mobility] == 0.5 || haskey(param, :nregions) || haskey(param, :region) || haskey(param, :omegar) || haskey(param, :Lr)
        if !haskey(param, :Lr)
            error("For partial mobility case need to provide a parameter 'Lr' containing a vector with the total populations of each region")        
        end
        if !haskey(param, :region)
            error("For partial mobility case need to provide a parameter 'region' containing an integer vector that maps each node of the graph to a region. The vector should have values in the range 1:nregions and be of length graph.J (=number of nodes).")        
        end
        if !haskey(param, :omegar)
            param[:omegar] = ones(length(param[:Lr]))
        end
        if !haskey(param, :nregions)
            param[:nregions] = length(param[:Lr])
        end
    end

    # Define utility function, marginal utility of consumtion and inverse
    param[:u] = (c, h) -> ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^(1 - rho) / (1 - rho)
    param[:uprime] = (c, h) -> ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^-rho * ((c / alpha)^(alpha - 1) * (h / (1 - alpha))^(1 - alpha))
    param[:usecond] = (c, h) -> -rho * ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^(-rho - 1) * ((c / alpha)^(alpha - 1) * (h / (1 - alpha))^(1 - alpha))^2 + (alpha - 1) / alpha * ((c / alpha)^alpha * (h / (1 - alpha))^(1 - alpha))^-rho * ((c / alpha)^(alpha - 2) * (h / (1 - alpha))^(1 - alpha))
    param[:uprimeinv] = (x, h) -> alpha * x^(-1 / (1 + alpha * (rho - 1))) * (h / (1 - alpha))^-((1 - alpha) * (rho - 1) / (1 + alpha * (rho - 1)))
    
    # Define production function 
    param[:F] = (L, a) -> L^a
    param[:Fprime] = (L, a) -> a * L^(a - 1)

    # CHECK CONSISTENCY WITH ENDOWMENTS/PRODUCTIVITY (if applicable)
    if haskey(param, :omegaj) && length(param[:omegaj]) != param[:J]
        @warn "omegaj does not have the right length J = $(param[:J])."
    end

    if haskey(param, :Lj) && param[:mobility] == 0 && length(param[:Lj]) != param[:J]
        @warn "Lj does not have the right length J = $(param[:J])."
    end
    # Zjn is a two-dimensional array (JxN), so using size() is appropriate for this check
    if haskey(param, :Zjn) && size(param[:Zjn]) != (param[:J], param[:N])
        @warn "Zjn does not have the right size J ($(param[:J])) x N ($(param[:N]))."
    end

    if haskey(param, :Hj) && length(param[:Hj]) != param[:J]
        @warn "Hj does not have the right length J = $(param[:J])."
    end

    if haskey(param, :hj) && length(param[:hj]) != param[:J]
        @warn "hj does not have the right length J = $(param[:J])."
    end
    
    return param
end
