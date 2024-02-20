
using Ipopt
using ForwardDiff

function solve_allocation_by_duality_ADiGator(x0, auxdata, funcs, verbose=true)

    # Recover parameters
    graph = auxdata.graph
    param = auxdata.param

    if isempty(x0)
        x0 = vec(ones(graph.J, param.N) * range(1, 2, length=graph.J) * range(1, 2, length=param.N)')
    end

    # Parametrize IPOPT
    options = Dict(
        :lb => 1e-6 * ones(graph.J * param.N),
        :ub => Inf * ones(graph.J * param.N),
        :print_level => verbose ? 5 : 0
    )

    # Run IPOPT
    x, info = ipopt(x0, funcs, options)

    # Return results
    flag = info
    results = recover_allocation(x, auxdata)
    results[:welfare] = funcs.objective(x)

    return results, flag, x
end

function recover_allocation(x, auxdata)

    # Extract parameters
    param = auxdata.param
    graph = auxdata.graph
    A = auxdata.A
    kappa_ex = auxdata.kappa_ex
    omegaj = param.omegaj
    kappa = auxdata.kappa

    # Population
    results = Dict()
    results["Lj"] = param.Lj

    # Extract price vectors
    results["Pjn"] = reshape(x, (graph.J, param.N))
    results["PCj"] = sum(results["Pjn"].^(1-param.sigma), dims=2).^(1/(1-param.sigma))

    # Calculate labor allocation
    results["Ljn"] = zeros(graph.J, param.N)
    if param.a < 1
        results["Ljn"] = ((results["Pjn"].*param.Zjn).^(1/(1-param.a))) ./ sum((results["Pjn"].*param.Zjn).^(1/(1-param.a)), dims=2) .* results["Lj"]
        results["Ljn"][param.Zjn .== 0] .= 0
    else
        max_id = argmax(results["Pjn"].*param.Zjn, dims=2)
        results["Ljn"][LinearIndices((1:graph.J, max_id))] .= param.Lj
    end
    results["Yjn"] = param.Zjn .* (results["Ljn"].^param.a)

    # Calculate consumption
    results["cj"] = param.alpha * (sum(results["Pjn"].^(1-param.sigma), dims=2).^(1/(1-param.sigma)) ./ omegaj).^(-1/(1+param.alpha*(param.rho-1))) .* (param.hj/(1-param.alpha)).^(-((1-param.alpha)*(param.rho-1))/(1+param.alpha*(param.rho-1)))
    zeta = omegaj .* ((results["cj"]/param.alpha).^param.alpha .* (param.hj/(1-param.alpha)).^(1-param.alpha)).^(-param.rho) .* ((results["cj"]/param.alpha).^(param.alpha-1) .* (param.hj/(1-param.alpha)).^(1-param.alpha))
    cjn = (results["Pjn"] ./ zeta).^(-param.sigma) .* results["cj"]
    results["Cj"] = results["cj"] .* param.Lj
    results["Cjn"] = cjn .* param.Lj

    # Non-tradeable per capita
    results["hj"] = param.hj

    # Vector of welfare per location
    results["uj"] = param.u(results["cj"], results["hj"])

    # Calculate Qin_direct (which is the flow in the direction of the edge)
    # and Qin_indirect (flow in edge opposite direction)
    Qin_direct = zeros(graph.ndeg, param.N)
    Qin_indirect = zeros(graph.ndeg, param.N)
    for n in 1:param.N
       Qin_direct[:,n] = max.(1/(1+param.beta)*kappa_ex.*(-A'*results["Pjn"][:,n] ./ (max.(A', 0)*results["Pjn"][:,n])), 0).^(1/param.beta)
    end
    for n in 1:param.N
       Qin_indirect[:,n] = max.(1/(1+param.beta)*kappa_ex.*(A'*results["Pjn"][:,n] ./ (max.(-A', 0)*results["Pjn"][:,n])), 0).^(1/param.beta)
    end

    # Calculate the flows Qin of dimension [Ndeg,N]
    results["Qin"] = zeros(graph.ndeg, param.N)

    for i in 1:param.N*graph.ndeg
        if Qin_direct[i] > Qin_indirect[i]
            results["Qin"][i] = Qin_direct[i]
        else
            results["Qin"][i] = -Qin_indirect[i]
        end
    end

    # Calculate the flows Qjkn of dimension [J,J,N]
    results["Qjkn"] = zeros(graph.J, graph.J, param.N)
    for n in 1:param.N
        Lambda = repeat(results["Pjn"][:,n], 1, graph.J)
        LL = max.(Lambda' .- Lambda, 0)
        LL[.!graph.adjacency] .= 0
        results["Qjkn"][:,:,n] = (1/(1+param.beta)*kappa.*LL ./ Lambda).^(1/param.beta)
    end

    return results
end


# Please note that the `ipopt` function is not available in Julia. You would need to use the `Ipopt.jl` package and adjust the code accordingly. The `ipopt` function in the MATLAB code seems to be a custom function, not a built-in MATLAB function. You would need to replace it with an equivalent function in Julia. The `Ipopt.jl` package in Julia provides similar functionality.