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
    hess_str = hessian_structure_duality(auxdata)
    auxdata = (auxdata..., hess = hess_str, hess_ind = CartesianIndex.(hess_str[1], hess_str[2]))
    nnz_hess = length(hess_str[1])


    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(1e-3, n),  # Variable lower bounds
        fill(Inf, n),   # Variable upper bounds
        0,              # No constraints
        Float64[],      # No constraint lower bounds
        Float64[],      # No constraint upper bounds
        0,              # Number of non-zeros in Jacobian (not used without constraints)
        nnz_hess,       # Int((n * (n + 1)) / 2),  # Number of non-zeros in Hessian
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
    cons = res.cjn .* graph.Lj .+ dropdims(sum(res.Qjkn .+ cost .- permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) .- res.Yjn
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
    cons = res.cjn .* graph.Lj .+ dropdims(sum(res.Qjkn .+ cost .- permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) .- res.Yjn

    grad_f .= -cons[:]  # Flatten the array and store in grad_f
    return
end

# Hessian structure function
function hessian_structure_duality(auxdata)
    graph = auxdata.graph
    param = auxdata.param

    # Create the Hessian structure
    H_structure = sparse(tril(repeat(I(graph.J), param.N, param.N) + kron(I(param.N), graph.adjacency)))
        
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
        # # Return the sparsity structure
        # nz = length(findnz(tril(h))[1])
        # resize!(rows, nz)
        # resize!(cols, nz)
        # r, c, _ = findnz(tril(h))
        r, c = auxdata.hess
        rows .= r
        cols .= c
    else
        # function hessian(x, auxdata, obj_factor, lambda)
        param = auxdata.param
        graph = auxdata.graph
        kappa = auxdata.kappa

        res = recover_allocation_duality(x, auxdata)
        Lambda = repeat(x, 1, graph.J * param.N)
        lambda = reshape(x, (graph.J, param.N))

        # Compute price index
        P = sum(lambda .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma))
        mat_P = repeat(P, param.N, param.N * graph.J)

        # Create masks
        # This lets different products in the same location relate
        Iij = kron(ones(param.N, param.N), I(graph.J))
        # This is adjacency, but only for the same product (n == n')
        Inm = kron(I(param.N), graph.adjacency)

        # Compute terms for Hessian
        termA = -param.sigma * (repeat(P, param.N) .^ param.sigma .* x .^ (-(param.sigma + 1)) .* repeat(res.Cj, param.N))

        part1 = Iij .* Lambda .^ (-param.sigma) .* Lambda' .^ (-param.sigma) .* mat_P .^ (2 * param.sigma)

        termB = param.sigma * part1 ./ mat_P .* repeat(res.Cj, param.N, graph.J * param.N)
        
        termC = part1 .* repeat(graph.Lj ./ (graph.omegaj .* param.usecond.(res.cj, graph.hj)), param.N, graph.J * param.N)
        
        diff = Lambda' - Lambda
        mat_kappa = repeat(kappa, param.N, param.N)
        part1 = 1 / (param.beta * (1 + param.beta)^(1 / param.beta)) * Inm .* mat_kappa .^ (1 / param.beta)
        abs_diff_1betam1 = abs.(diff) .^ (1 / param.beta - 1)
        diffpos = diff .> 0
        diffneg = diff .< 0
        
        termD = part1 .* abs_diff_1betam1 .* 
                (diffpos .* Lambda' ./ Lambda .^ (1 + 1 / param.beta) + 
                 diffneg .* Lambda ./ Lambda' .^ (1 + 1 / param.beta))
        
        termE = -part1 .* abs_diff_1betam1 .* 
                (diffpos .* Lambda' .^ 2 ./ Lambda .^ (2 + 1 / param.beta) + 
                 diffneg ./ Lambda' .^ (1 / param.beta))
        termE = sum(termE, dims=2)

        # Compute labor term
        if param.a == 1
            X = 0
        else
            # This is Psi in your notes
            denom = sum((lambda .* graph.Zjn) .^ (1 / (1 - param.a)), dims=2)
            num = (x .* graph.Zjn[:]) .^ (1 / (1 - param.a))
            # Non-diagonal elements: using 1-a in denominator because output is subtracted
            X = param.a / (1 - param.a) * Iij .* (graph.Zjn[:] * graph.Zjn[:]') .* (num * num') .^ param.a .*    
                repeat(graph.Lj .^ param.a ./ denom .^ (1 + param.a), param.N, graph.J * param.N) 
            # The term that is multiplied to get the diagonal            
            X_diag = (num - repeat(denom, param.N)) ./ num
            # Adding the diagonal
            @inbounds for i in 1:length(X_diag)
                X[i, i] *= X_diag[i]
            end
        end
        # Compute Hessian
        h = -obj_factor * (Diagonal(termA[:]) + termB + termC + termD + Diagonal(termE[:]) + X)
        
        # Return lower triangular part
        values .= h[auxdata.hess_ind]
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
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        # Note: max(P^n_k/P^n_j-1, 0) = max(P^n_k-P^n_j, 0)/P^n_j
        LL = max.(Lambda' - Lambda, 0) # P^n_k - P^n_j
        LL[.!graph.adjacency] .= 0
        Qjkn[:, :, n] = (1 / (1 + param.beta) * kappa .* LL ./ Lambda) .^ (1 / param.beta)
    end

    return (Pjn=Pjn, PCj=PCj, Ljn=Ljn, Yjn=Yjn, cj=cj, cjn=cjn, Cj=Cj, Cjn=Cjn, Qjkn=Qjkn)
end