
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
- `param::Dict=Dict()`: provide an already existing 'param' structure if you just want to change parameters.
"""
function init_parameters(; alpha=0.5, beta=1, gamma=1, K=1, sigma=5, rho=2, a=0.8, N=1, m=ones(N,1), nu=1, 
                         labor_mobility=false, cross_good_congestion=false, annealing=true, custom=false, 
                         verbose=true, duality=true, param=Dict(), kwargs...)
    p = isempty(param) ? Dict() : param
    if !isempty(kwargs)
        for (key, value) in kwargs
            p[key] = value
        end
    end
    param = Dict()

    param[:gamma] = get(p, :gamma, gamma)
    param[:alpha] = get(p, :alpha, alpha) 
    param[:beta] = get(p, :beta, beta)
    param[:K] = get(p, :K, K)
    param[:sigma] = get(p, :sigma, sigma)
    param[:rho] = get(p, :rho, rho)
    param[:a] = get(p, :a, a)
    param[:m] = get(p, :m, m)
    param[:N] = get(p, :N, N)
    param[:nu] = get(p, :nu, nu)
    labor_mobility = get(p, :labor_mobility, labor_mobility)
    if labor_mobility === "partial" || labor_mobility === 0.5
        param[:mobility] = 0.5
    else
        param[:mobility] = labor_mobility === true
    end
    if param[:mobility] == true
        param[:rho] = 0;
    end
    param[:cong] = get(p, :cross_good_congestion, cross_good_congestion)
    param[:annealing] = get(p, :annealing, annealing)
    param[:custom] = get(p, :custom, custom)
    param[:verbose] = get(p, :verbose, verbose)
    param[:duality] = get(p, :duality, duality)
    param[:warm_start] = get(p, :warm_start, false)

    # Additional parameters for the numerical part
    param[:kappa_tol] = get(p, :kappa_tol, 1e-7)
    param[:kappa_min] = get(p, :kappa_min, 1e-5)
    param[:kappa_min_iter] = get(p, :kappa_min_iter, 20)
    param[:kappa_max_iter] = get(p, :kappa_max_iter, 200)

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

    # Define utility function
    param[:u] = (c, h) -> ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(1-param[:rho])/(1-param[:rho])
    param[:uprime] = (c, h) -> ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(-param[:rho]) * ((c/param[:alpha])^(param[:alpha]-1) * (h/(1-param[:alpha]))^(1-param[:alpha]))
    param[:usecond] = (c, h) -> -param[:rho] * ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(-param[:rho]-1) * ((c/param[:alpha])^(param[:alpha]-1) * (h/(1-param[:alpha]))^(1-param[:alpha]))^2 + (param[:alpha]-1)/param[:alpha] * ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(-param[:rho]) * ((c/param[:alpha])^(param[:alpha]-2) * (h/(1-param[:alpha]))^(1-param[:alpha]))
    param[:uprimeinv] = (x, h) -> param[:alpha] * x^(-1/(1+param[:alpha]*(param[:rho]-1))) * (h/(1-param[:alpha]))^-((1-param[:alpha])*(param[:rho]-1)/(1+param[:alpha]*(param[:rho]-1)))

    # Define production function 
    param[:F] = (L, a) -> L^a
    param[:Fprime] = (L, a) -> a * L^(a-1)

    unmatched_keys = setdiff(keys(p), union(keys(param), [:optimizer_attr, :model_attr, :Zjn, :J, :omegaj, :hj, :Lj, :Hj]))
    # Check if non-supported keys
    if !isempty(unmatched_keys)
        # Print the error message indicating the unmatched keys
        @warn "Unsupported parameters:  $unmatched_keys"
    end

    # CHECK CONSISTENCY WITH ENDOWMENTS/PRODUCTIVITY (if applicable)
    # only happens when object param is specified
    if haskey(p, :omegaj)
        if length(p[:omegaj]) != p[:J]
            @warn "omegaj does not have the right length J = $(p[:J])."
        end
        param[:omegaj] = p[:omegaj]
    end

    if haskey(p, :Lj) && param[:mobility] == 0
        if length(p[:Lj]) != p[:J]
            @warn "Lj does not have the right length J = $(p[:J])."
        end
        param[:Lj] = p[:Lj]
    end

    # Zjn is a two-dimensional array (JxN), so using size() is appropriate for this check
    if haskey(p, :Zjn)
        if size(p[:Zjn]) != (p[:J], p[:N])
            @warn "Zjn does not have the right size J ($(p[:J])) x N ($(p[:N]))."
        end
        param[:Zjn] = p[:Zjn]
    end

    if haskey(p, :Hj)
        if length(p[:Hj]) != p[:J]
            @warn "Hj does not have the right length J = $(p[:J])."
        end
        param[:Hj] = p[:Hj]
    end

    if haskey(p, :hj)
        if length(p[:hj]) != p[:J]
            @warn "hj does not have the right length J = $(p[:J])."
        end
        param[:hj] = p[:hj]
    end
    
    return param
end
