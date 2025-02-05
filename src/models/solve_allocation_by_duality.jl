"""
Solves the full allocation of Qjkn and Cj given a matrix of kappa (=I^gamma/delta_tau).
Solves the case without labor mobility with a duality approach.
Uses Ipopt.jl for optimization.

Arguments:
- x0: initial seed for the solver (lagrange multipliers P_j^n)
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A)
- verbose: {true | false} tells IPOPT to display results or not

Returns:
- results: structure of results (Cj, Qjkn, etc.)
- flag: flag returned by IPOPT
- x: returns the 'x' variable returned by IPOPT (useful for warm start)
"""
function solve_allocation_by_duality(x0, auxdata, verbose=true)
    # Extract parameters
    graph = auxdata.graph
    param = auxdata.param

    # Initialize x0 if not provided
    if isempty(x0)
        RN = param.N > 1 ? range(1, 2, length=param.N)' : 1
        x0 = vec(range(1, 2, length=graph.J) * RN)
    end

    obj = (x) -> objective_duality(x, auxdata)
    grad = (x, grad_f) -> gradient_duality(x, grad_f, auxdata)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_duality(x, rows, cols, obj_factor, lambda, values, auxdata)

    # Set up the optimization problem
    n = graph.J * param.N

    # Get the Hessian structure
    auxdata = (auxdata..., hess = hessian_structure_duality(auxdata))
    nnz_hess = length(auxdata.hess[1])

    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(1e-3, n),  # Variable lower bounds
        fill(Inf, n),   # Variable upper bounds
        0,              # No constraints
        Float64[],      # No constraint lower bounds
        Float64[],      # No constraint upper bounds
        0,              # Number of non-zeros in Jacobian (not used without constraints)
        nnz_hess,       # Number of non-zeros in Hessian
        obj,            # Objective function
        (args...) -> Float64[], # Constraints function (empty)
        grad,           # Gradient function
        (args...) -> Float64[], # Jacobian value function (empty)
        hess            # Hessian computation function
    )

    # Set Ipopt options
    Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "exact")
    Ipopt.AddIpoptIntOption(prob, "max_iter", 3000)
    Ipopt.AddIpoptIntOption(prob, "print_level", verbose ? 5 : 0)

    if haskey(param, :optimizer_attr)
        for (key, value) in param.optimizer_attr
            if value isa String
                Ipopt.AddIpoptStrOption(prob, String(key), value)
            elseif value isa Number
                if value isa Float64
                    Ipopt.AddIpoptNumOption(prob, String(key), value)
                else
                    Ipopt.AddIpoptIntOption(prob, String(key), value)
                end
            end
        end
    end

    # Solve the problem
    prob.x = x0
    status = Ipopt.IpoptSolve(prob)
    x = prob.x

    # Compute and return results
    results = recover_allocation_duality(x, auxdata)
    results = (results..., hj = graph.hj, Lj = graph.Lj, welfare = prob.obj_val, uj = param.u.(results.cj, graph.hj))
    return namedtuple_to_dict(results), status, x
end

# Objective function
function objective_duality(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph

    res = recover_allocation_duality(x, auxdata)
    Pjn = reshape(x, (graph.J, param.N))

    # Compute transportation cost
    cost = res.Qjkn .^ (1 + param.beta) ./ repeat(auxdata.kappa, 1, 1, param.N)
    cost[res.Qjkn .== 0] .= 0  # Deal with cases Qjkn=kappa=0

    # Compute constraint
    cons = res.cjn .* graph.Lj + dropdims(sum(res.Qjkn + cost - permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) - res.Yjn
    cons = sum(Pjn .* cons, dims=2)

    # Compute objective value
    f = sum(graph.omegaj .* graph.Lj .* param.u.(res.cj, graph.hj)) - sum(cons)
    return f # Negative objective because Ipopt minimizes?  
end

# Gradient function = negative constraint
function gradient_duality(x::Vector{Float64}, grad_f::Vector{Float64}, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    kappa = auxdata.kappa

    res = recover_allocation_duality(x, auxdata)

    # Compute transportation cost
    cost = res.Qjkn .^ (1 + param.beta) ./ repeat(kappa, 1, 1, param.N)
    cost[res.Qjkn .== 0] .= 0  # Deal with cases Qjkn=kappa=0

    # Compute constraint
    cons = res.cjn .* graph.Lj + dropdims(sum(res.Qjkn + cost - permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) - res.Yjn

    grad_f .= -cons[:]  # Flatten the array and store in grad_f
    return
end

# Hessian structure function
function hessian_structure_duality(auxdata)
    graph = auxdata.graph
    param = auxdata.param

    # Create the Hessian structure
    H_structure = tril(repeat(sparse(I(graph.J)), param.N, param.N) + kron(sparse(I(param.N)), sparse(graph.adjacency)))
        
    # Get the row and column indices of non-zero elements
    rows, cols, _ = findnz(H_structure)
    return rows, cols
end


# Hessian computation function
function hessian_duality(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    obj_factor::Float64,
    lambda::Vector{Float64},
    values::Union{Nothing,Vector{Float64}},
    auxdata
)
    if values === nothing
        r, c = auxdata.hess
        rows .= r
        cols .= c
    else
        # function hessian(x, auxdata, obj_factor, lambda)
        param = auxdata.param
        graph = auxdata.graph
        nodes = graph.nodes
        beta = param.beta
        m1dbeta = -1 / beta
        sigma = param.sigma
        a = param.a
        adam1 = a / (a - 1)
        J = graph.J
        Zjn = graph.Zjn
        Lj = graph.Lj
        omegaj = graph.omegaj  
        hess_str = auxdata.hess

        # Precompute elements
        res = recover_allocation_duality(x, auxdata)
        Pjn = res.Pjn
        PCj = res.PCj
        Qjkn = res.Qjkn

        # Prepare computation of production function 
        if a != 1
            psi = (Pjn .* Zjn) .^ (1 / (1 - a))
            Psi = sum(psi, dims=2)
        end

        # Now the big loop over the hessian terms to compute
        ind = 0
        
        # https://stackoverflow.com/questions/38901275/inbounds-propagation-rules-in-julia
        @inbounds for (jdnd, jn) in zip(hess_str[1], hess_str[2])
            ind += 1
            # First get the indices of the element and the respective derivative 
            j = (jn-1) % J + 1
            n = Int(ceil(jn / J))
            jd = (jdnd-1) % J + 1
            nd = Int(ceil(jdnd / J))
            # Get neighbors k
            neighbors = nodes[j]

            # Stores the derivative term
            term = 0.0

            # Starting with C^n_j = CG, derivative: C'G + CG'
            if jd == j # 0 for P^n_k
                Cprime = Lj[j]/omegaj[j] * (PCj[j] / Pjn[j, nd])^sigma / param.usecond(res.cj[j], graph.hj[j]) # param.uprimeinvprime(PCj[j]/omegaj[j], graph.hj[j])
                G = (Pjn[j, n] / PCj[j])^(-sigma)
                Gprime = sigma * (Pjn[j, n] * Pjn[j, nd])^(-sigma) * PCj[j]^(2*sigma-1) 
                if nd == n
                    Gprime -= sigma / PCj[j] * G^((sigma+1)/sigma)
                end
                term += Cprime * G + res.Cj[j] * Gprime
            end

            # Now terms sum_k((Q^n_{jk} + K_{jk}(Q^n_{jk})^(1+beta)) - Q^n_{kj})
            if nd == n 
                for k in neighbors
                    if jd == j # P^x_j
                        diff = Pjn[k, n] - Pjn[j, n]
                        if diff >= 0 # Flows in the direction of k
                            term += m1dbeta * (Pjn[k, n] / Pjn[j, n])^2 * Qjkn[j, k, n] / diff
                        else # Flows in the direction of j
                            term += m1dbeta * Qjkn[k, j, n] / abs(diff)
                        end
                    elseif jd == k # P^x_k
                        diff = Pjn[k, n] - Pjn[j, n]
                        if diff >= 0 # Flows in the direction of k
                            term -= m1dbeta * Pjn[k, n] / Pjn[j, n] * Qjkn[j, k, n] / diff
                        else # Flows in the direction of j
                            term -= m1dbeta * Pjn[j, n] / Pjn[k, n] * Qjkn[k, j, n] / abs(diff)
                        end
                    end
                end # End of k loop
            end

            # Finally: need to compute production function (X)
            if a != 1 && jd == j
                X = adam1 * Zjn[j, n] * Zjn[j, nd] * (psi[j, n] * psi[j, nd])^a / Psi[j]^(a+1) * Lj[j]^a
                if nd == n
                    X *= 1 - Psi[j] / psi[j, n]
                end
                term -= X
            end

            # Assign result
            values[ind] = -obj_factor * term # + 1e-5 # Somehow need this increment to make it work
        end
    end
    return
end

# Function to recover allocation from optimization variables
function recover_allocation_duality(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    omegaj = graph.omegaj
    kappa = auxdata.kappa
    Lj = graph.Lj
    hj1malpha = (graph.hj / (1-param.alpha)) .^ (1-param.alpha)

    # Extract price vectors
    Pjn = reshape(x, (graph.J, param.N))
    PCj = sum(Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma))

    # Calculate labor allocation
    if param.a < 1
        temp = (Pjn .* graph.Zjn) .^ (1 / (1 - param.a))
        Ljn = temp .* (Lj ./ sum(temp, dims=2))
        Ljn[graph.Zjn .== 0] .= 0
    else
        _, max_id = findmax(Pjn .* graph.Zjn, dims=2)
        Ljn = zeros(graph.J, param.N)
        Ljn[CartesianIndex.(1:graph.J, getindex.(max_id, 2))] .= Lj
    end
    Yjn = graph.Zjn .* (Ljn .^ param.a)

    # Calculate consumption
    cj = param.alpha * (PCj ./ omegaj) .^ 
         (-1 / (1 + param.alpha * (param.rho - 1))) .* 
         hj1malpha .^ (-(param.rho - 1)/(1 + param.alpha * (param.rho - 1)))
    
    # This is = PCj
    zeta = omegaj .* ((cj / param.alpha) .^ param.alpha .* hj1malpha) .^ (-param.rho) .* 
           ((cj / param.alpha) .^ (param.alpha - 1) .* hj1malpha)
    
    cjn = (Pjn ./ zeta) .^ (-param.sigma) .* cj
    Cj = cj .* Lj
    Cjn = cjn .* Lj

    # Calculate the flows Qjkn
    Qjkn = zeros(graph.J, graph.J, param.N)
    nadj = findall(.!graph.adjacency)
    @inbounds for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        # Note: max(P^n_k/P^n_j-1, 0) = max(P^n_k-P^n_j, 0)/P^n_j
        LL = max.(Lambda' - Lambda, 0) # P^n_k - P^n_j
        LL[nadj] .= 0
        Qjkn[:, :, n] = (1 / (1 + param.beta) * kappa .* LL ./ Lambda) .^ (1 / param.beta)
    end

    return (Pjn=Pjn, PCj=PCj, Ljn=Ljn, Yjn=Yjn, cj=cj, cjn=cjn, Cj=Cj, Cjn=Cjn, Qjkn=Qjkn)
end