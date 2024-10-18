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
    return f # Negative objective because Ipopt minimizes  
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
        Iij = kron(ones(param.N, param.N), I(graph.J))
        Inm = kron(I(param.N), graph.adjacency)

        # Compute terms for Hessian
        termA = -param.sigma * (repeat(P, param.N) .^ param.sigma .* x .^ (-(param.sigma + 1)) .* repeat(res.Cj, param.N))
        
        termB = param.sigma * Iij .* Lambda .^ (-param.sigma) .* Lambda' .^ (-param.sigma) .* 
                mat_P .^ (2 * param.sigma - 1) .* repeat(res.Cj, param.N, graph.J * param.N)
        
        termC = Iij .* Lambda .^ (-param.sigma) .* Lambda' .^ (-param.sigma) .* mat_P .^ (2 * param.sigma) .* 
                repeat(graph.Lj ./ (graph.omegaj .* param.usecond.(res.cj, graph.hj)), param.N, graph.J * param.N)
        
        diff = Lambda' - Lambda
        mat_kappa = repeat(kappa, param.N, param.N)
        
        termD = 1 / (param.beta * (1 + param.beta)^(1 / param.beta)) * Inm .* mat_kappa .^ (1 / param.beta) .* 
                abs.(diff) .^ (1 / param.beta - 1) .* 
                ((diff .> 0) .* Lambda' ./ Lambda .^ (1 + 1 / param.beta) + 
                (diff .< 0) .* Lambda ./ Lambda' .^ (1 + 1 / param.beta))
        
        termE = -1 / (param.beta * (1 + param.beta)^(1 / param.beta)) * Inm .* mat_kappa .^ (1 / param.beta) .* 
                abs.(diff) .^ (1 / param.beta - 1) .* 
                ((diff .> 0) .* Lambda' .^ 2 ./ Lambda .^ (2 + 1 / param.beta) + 
                (diff .< 0) ./ Lambda' .^ (1 / param.beta))
        termE = sum(termE, dims=2)

        # Compute labor term
        if param.a == 1
            X = 0
        else
            denom = sum((lambda .* graph.Zjn) .^ (1 / (1 - param.a)), dims=2)
            Lambdaz = repeat(x .* graph.Zjn[:], 1, graph.J * param.N)
            X_nondiag = param.a / (1 - param.a) * Iij .* repeat(graph.Zjn[:], 1, graph.J * param.N) .* 
                        repeat(graph.Zjn[:]', graph.J * param.N) .* 
                        repeat(graph.Lj .^ param.a ./ denom .^ (1 + param.a), param.N, graph.J * param.N) .* 
                        Lambdaz .^ (param.a / (1 - param.a)) .* Lambdaz' .^ (param.a / (1 - param.a))
            X_diag = -param.a / (1 - param.a) * repeat((graph.Lj ./ denom) .^ param.a, param.N) .* 
                    graph.Zjn[:] ./ x .* (x .* graph.Zjn[:]) .^ (param.a / (1 - param.a))
            X = X_nondiag + Diagonal(X_diag[:])
        end
        # Compute Hessian
        h = -obj_factor * (Diagonal(termA[:]) + termB + termC + termD + Diagonal(termE[:]) + X)
        
        # Return lower triangular part
        r, c = auxdata.hess
        values .= h[CartesianIndex.(r, c)]
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

    # Extract price vectors
    Pjn = reshape(x, (graph.J, param.N))
    PCj = sum(Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma))

    # Calculate labor allocation
    if param.a < 1
        Ljn = ((Pjn .* graph.Zjn) .^ (1 / (1 - param.a))) ./ 
              repeat(sum((Pjn .* graph.Zjn) .^ (1 / (1 - param.a)), dims=2), 1, param.N) .* 
              repeat(Lj, 1, param.N)
        Ljn[graph.Zjn .== 0] .= 0
    else
        _, max_id = findmax(Pjn .* graph.Zjn, dims=2)
        Ljn = zeros(graph.J, param.N)
        Ljn[CartesianIndex.(1:graph.J, getindex.(max_id, 2))] .= Lj
    end
    Yjn = graph.Zjn .* (Ljn .^ param.a)

    # Calculate consumption
    cj = param.alpha * (sum(Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma)) ./ omegaj) .^ 
         (-1 / (1 + param.alpha * (param.rho - 1))) .* 
         (graph.hj / (1 - param.alpha)) .^ (-((1 - param.alpha) * (param.rho - 1) / (1 + param.alpha * (param.rho - 1))))
    
    zeta = omegaj .* ((cj / param.alpha) .^ param.alpha .* (graph.hj / (1 - param.alpha)) .^ (1 - param.alpha)) .^ (-param.rho) .* 
           ((cj / param.alpha) .^ (param.alpha - 1) .* (graph.hj / (1 - param.alpha)) .^ (1 - param.alpha))
    
    cjn = (Pjn ./ repeat(zeta, 1, param.N)) .^ (-param.sigma) .* repeat(cj, 1, param.N)
    Cj = cj .* Lj
    Cjn = cjn .* repeat(Lj, 1, param.N)

    # Calculate the flows Qjkn
    Qjkn = zeros(graph.J, graph.J, param.N)
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = max.(Lambda' - Lambda, 0)
        LL[.!graph.adjacency] .= 0
        Qjkn[:, :, n] = (1 / (1 + param.beta) * kappa .* LL ./ Lambda) .^ (1 / param.beta)
    end

    return (Pjn=Pjn, PCj=PCj, Ljn=Ljn, Yjn=Yjn, cj=cj, cjn=cjn, Cj=Cj, Cjn=Cjn, Qjkn=Qjkn)
end