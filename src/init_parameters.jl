
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
- `LaborMobility::String="off"`: switch for labor mobility ('on'/'off'/'partial')
- `CrossGoodCongestion::Bool=false`: switch for cross-good congestion
- `Annealing::Bool=true`: switch for the use of annealing at the end of iterations
- `Custom::Bool=false`: switch for the use of custom lagrangian function
- `Verbose::Bool=true`: switch to turn on/off text output
- `ADiGator::Bool=false`: use autodifferentiation with Adigator
- `Duality::Bool=true`: switch to turn on/off duality whenever available
- `param::Dict=Dict()`: provide an already existing 'param' structure if you just want to change parameters.
"""
function init_parameters(; alpha=0.5, beta=1, gamma=1, K=1, sigma=5, rho=2, a=0.8, N=1, m=ones(N,1), nu=1, LaborMobility="off", CrossGoodCongestion=false, Annealing=true, Custom=false, Verbose=true, ADiGator=false, Duality=true, param=Dict(), kwargs...)
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
    LaborMobility = get(p, :LaborMobility, LaborMobility)
    if LaborMobility == "partial"
        param[:mobility] = 0.5
    else
        param[:mobility] = LaborMobility == "on"
    end
    if param[:mobility] == true
        param[:rho] = 0;
    end
    param[:cong] = get(p, :CrossGoodCongestion, CrossGoodCongestion)
    param[:annealing] = get(p, :Annealing, Annealing)
    param[:custom] = get(p, :Custom, Custom)
    param[:verbose] = get(p, :Verbose, Verbose)
    param[:adigator] = get(p, :ADiGator, ADiGator)
    param[:duality] = get(p, :Duality, Duality)

    # Additional parameters for the numerical part
    param[:tol_kappa] = get(p, :TolKappa, 1e-7)
    param[:MIN_KAPPA] = get(p, :min_kappa, 1e-5)
    param[:MIN_KAPPA] = get(p, :minpop, 1e-3) 
    param[:MAX_ITER_KAPPA] = get(p, :max_iter_kappa, 200)
    param[:MAX_ITER_L] = get(p, :max_iter_l, 100)

    # Define utility function
    param[:u] = (c, h) -> ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(1-param[:rho])/(1-param[:rho])
    param[:uprime] = (c, h) -> ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(-param[:rho]) * ((c/param[:alpha])^(param[:alpha]-1) * (h/(1-param[:alpha]))^(1-param[:alpha]))
    param[:usecond] = (c, h) -> -param[:rho] * ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(-param[:rho]-1) * ((c/param[:alpha])^(param[:alpha]-1) * (h/(1-param[:alpha]))^(1-param[:alpha]))^2 + (param[:alpha]-1)/param[:alpha] * ((c/param[:alpha])^param[:alpha] * (h/(1-param[:alpha]))^(1-param[:alpha]))^(-param[:rho]) * ((c/param[:alpha])^(param[:alpha]-2) * (h/(1-param[:alpha]))^(1-param[:alpha]))
    param[:uprimeinv] = (x, h) -> param[:alpha] * x^(-1/(1+param[:alpha]*(param[:rho]-1))) * (h/(1-param[:alpha]))^-((1-param[:alpha])*(param[:rho]-1)/(1+param[:alpha]*(param[:rho]-1)))

    # Define production function 
    param[:F] = (L, a) -> L^a
    param[:Fprime] = (L, a) ->

    # CHECK CONSISTENCY WITH ENDOWMENTS/PRODUCTIVITY (if applicable)
    # only happens when object param is specified
    if haskey(p, :omegaj)
        if size(p[:omegaj]) != (param[:J], 1)
            @warn "$(basename(@__FILE__)).jl: omegaj does not have the right size Jx1."
        end
        param[:omegaj] = p[:omegaj]
    end

    if haskey(p, :Lj) && !param[:mobility]
        if size(p[:Lj]) != (param[:J], 1)
            @warn "$(basename(@__FILE__)).jl: Lj does not have the right size Jx1."
        end
        param[:Lj] = p[:Lj]
    end

    if haskey(p, :Zjn)
        if size(p[:Zjn]) != (param[:J], param[:N])
            @warn "$(basename(@__FILE__)).jl: Zjn does not have the right size JxN."
        end
        param[:Zjn] = p[:Zjn]
    end

    if haskey(p, :Hj)
        if size(p[:Hj]) != (param[:J], 1)
            @warn "$(basename(@__FILE__)).jl: Hj does not have the right size Jx1."
        end
        param[:Hj] = p[:Hj]
    end

    if haskey(p, :hj)
        if size(p[:hj]) != (param[:J], 1)
            @warn "$(basename(@__FILE__)).jl: hj does not have the right size Jx1."
        end
        param[:hj] = p[:hj]
    end

    return param
end


# Please note that this is a direct translation and the code may need to be adjusted to fit the specific needs of your project.

function dict_to_namedtuple(dict)
    if !(dict isa NamedTuple)
        return NamedTuple{Tuple(keys(dict))}(values(dict))
    else
        return dict
    end
end