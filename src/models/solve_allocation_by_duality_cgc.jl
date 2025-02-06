"""
Solves the full allocation of Qjkn and Cj given a matrix of kappa (=I^gamma/delta_tau).
Solves the case without labor mobility with a duality approach and cross-good congestion.
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
function solve_allocation_by_duality_cgc(x0, auxdata, verbose=true)
    # Extract parameters
    graph = auxdata.graph
    param = auxdata.param

    # Initialize x0 if not provided
    if isempty(x0)
        RN = param.N > 1 ? range(1, 2, length=param.N)' : 1
        x0 = vec(range(1, 2, length=graph.J) * RN)
    end

    obj = (x) -> objective_duality_cgc(x, auxdata)
    grad = (x, grad_f) -> gradient_duality_cgc(x, grad_f, auxdata)
    hess = (x, rows, cols, obj_factor, lambda, values) -> hessian_duality_cgc(x, rows, cols, obj_factor, lambda, values, auxdata)

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
    results = recover_allocation_duality_cgc(x, auxdata)
    results = (results..., hj = graph.hj, Lj = graph.Lj, welfare = prob.obj_val, uj = param.u.(results.cj, graph.hj))
    return namedtuple_to_dict(results), status, x
end

# Objective function
function objective_duality_cgc(x, auxdata)
    graph = auxdata.graph
    res = recover_allocation_duality_cgc(x, auxdata)

    # Compute constraint
    cons = res.Djn + dropdims(sum(res.Qjkn - permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) - res.Yjn
    cons = sum(res.Pjn .* cons, dims=2)

    # Compute objective value
    f = sum(graph.omegaj .* graph.Lj .* auxdata.param.u.(res.cj, graph.hj)) - sum(cons)

    return f # Negative objective because Ipopt minimizes?  
end

# Gradient function = negative constraint
function gradient_duality_cgc(x::Vector{Float64}, grad_f::Vector{Float64}, auxdata)

    res = recover_allocation_duality_cgc(x, auxdata)

    # Compute constraint
    cons = res.Djn + dropdims(sum(res.Qjkn - permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) - res.Yjn

    grad_f .= -cons[:]  # Flatten the array and store in grad_f
    return
end


# Hessian computation function
function hessian_duality_cgc(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    obj_factor::Float64,
    lambda::Vector{Float64}, # No constraints: thus no lagrange multipliers
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
        kappa = auxdata.kappa
        inexact_algo = param.duality == true # param.duality == 2 for exact algorithm
        beta = param.beta
        m1dbeta = -1 / beta
        sigma = param.sigma
        nu = param.nu
        m = param.m
        n1dnum1 = 1 / (nu - 1)
        numbetam1 = nu - beta - 1
        a = param.a
        adam1 = a / (a - 1)
        J = graph.J
        Zjn = graph.Zjn
        Lj = graph.Lj
        omegaj = graph.omegaj
        hess_str = auxdata.hess

        # Constant term: first part of Bprime
        cons = m1dbeta * n1dnum1 * numbetam1
        ## Constant in T'
        if inexact_algo
            cons2 = numbetam1 / (numbetam1+nu*beta)
        else
            # New: constant in T' -> More accurate!
            cons3 = m1dbeta * n1dnum1 * (numbetam1+nu*beta) / (1+beta)
        end

        # Precompute elements
        res = recover_allocation_duality_cgc(x, auxdata)
        Pjn = res.Pjn
        PCj = res.PCj
        Qjk = res.Qjk
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

            # Starting with D^n_j = (C+T)G, derivative: (C'+T')G + (C+T)G' = C'G + CG' + T'G + TG'
            # Terms involving T are computed below. Here forcus on C'G + CG'
            G = (Pjn[j, n] / PCj[j])^(-sigma)
            if jd == j # 0 for P^n_k
                Gprime = sigma * (Pjn[j, n] * Pjn[j, nd])^(-sigma) * PCj[j]^(2*sigma-1) 
                Cprime = Lj[j]/omegaj[j] * (PCj[j] / Pjn[j, nd])^sigma / param.usecond(res.cj[j], graph.hj[j]) # param.uprimeinvprime(PCj[j]/omegaj[j], graph.hj[j]) # 
                if nd == n
                    Gprime -= sigma / PCj[j] * G^((sigma+1)/sigma)
                end
                term += Cprime * G + res.Cj[j] * Gprime
            end

            # Now terms sum_k(Q^n_{jk} - Q^n_{kj}) as well as T'G + TG'
            for k in neighbors
                if Qjkn[j, k, n] > 0 # Flows in the direction of k
                    T = Qjk[j, k]^(1 + beta) / kappa[j, k]
                    PK0 = PCj[j] * (1 + beta) / kappa[j, k]
                    KPABprime1 = cons * ((Pjn[k, nd] - Pjn[j, nd])/m[nd])^n1dnum1 * (PK0 * Qjk[j, k]^beta)^(-nu*n1dnum1)
                    KPAprimeB1 = nd == n ? n1dnum1 / (Pjn[k, n] - Pjn[j, n]) : 0.0
                    if jd == j
                        KPprimeAB1 = m1dbeta * Pjn[j, nd]^(-sigma) * PCj[j]^(sigma-1)
                        term += Qjkn[j, k, n] * (KPprimeAB1 - KPAprimeB1 + KPABprime1) # Derivative of Qjkn
                        if inexact_algo
                            term += T * (((1+beta) * KPprimeAB1 + cons2 * KPABprime1) * G + Gprime) # T'G + TG'
                        else
                            term += T * ((1+beta) * KPprimeAB1 * G + Gprime) # T'G (first part) + TG'
                            term += cons3 * Qjkn[j, k, nd] / PCj[j] * G # Second part of T'G
                        end
                    elseif jd == k
                        term += Qjkn[j, k, n] * (KPAprimeB1 - KPABprime1) # Derivative of Qjkn
                        if inexact_algo
                            term -= T * cons2 * KPABprime1 * G # T'G: second part [B'(k) has opposite sign]
                        else
                            term -= cons3 * Qjkn[j, k, nd] / PCj[j] * G # Second part of T'G 
                        end
                    end
                end
                if Qjkn[k, j, n] > 0 # Flows in the direction of j
                    PK0 = PCj[k] * (1 + beta) / kappa[k, j]
                    KPABprime1 = cons * ((Pjn[j, nd] - Pjn[k, nd])/m[nd])^n1dnum1 * (PK0 * Qjk[k, j]^beta)^(-nu*n1dnum1)
                    KPAprimeB1 = nd == n ? n1dnum1 / (Pjn[j, n] - Pjn[k, n]) : 0.0 
                    if jd == k
                        KPprimeAB1 = m1dbeta * Pjn[k, nd]^(-sigma) * PCj[k]^(sigma-1)
                        term -= Qjkn[k, j, n] * (KPprimeAB1 - KPAprimeB1 + KPABprime1) # Derivative of Qkjn
                    elseif jd == j
                        term -= Qjkn[k, j, n] * (KPAprimeB1 - KPABprime1) # Derivative of Qkjn
                    end
                end
            end # End of k loop

            # Finally: need to compute production function (X)
            if a != 1 && jd == j
                X = adam1 * Zjn[j, n] * Zjn[j, nd] * (psi[j, n] * psi[j, nd])^a / Psi[j]^(a+1) * Lj[j]^a
                if nd == n
                    X *= 1 - Psi[j] / psi[j, n]
                end
                term -= X
            end

            # Assign result
            values[ind] = -obj_factor * term # + Random.rand() * 1e-6
        end
    end
    return
end

# Function to recover allocation from optimization variables
function recover_allocation_duality_cgc(x, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    omegaj = graph.omegaj
    kappa = auxdata.kappa
    Lj = graph.Lj
    m = param.m
    nu = param.nu
    beta = param.beta
    hj1malpha = (graph.hj / (1-param.alpha)) .^ (1-param.alpha)
    nadj = .!graph.adjacency

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

    # Calculate aggregate consumption
    cj = param.alpha * (PCj ./ omegaj) .^ 
         (-1 / (1 + param.alpha * (param.rho - 1))) .* 
         hj1malpha .^ (-(param.rho - 1)/(1 + param.alpha * (param.rho - 1)))
    Cj = cj .* Lj
        
    # Calculate the aggregate flows Qjk
    temp = kappa ./ ((1 + beta) * PCj)
    temp[nadj] .= 0
    Qjk = zeros(graph.J, graph.J)
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = max.(Lambda' - Lambda, 0) # P^n_k - P^n_j (non-negative flows)
        Qjk += m[n] * (LL / m[n]) .^ (nu/(nu-1))
    end
    Qjk .^= ((nu-1)/nu)
    Qjk .*= temp
    Qjk .^= (1/beta)

    # Calculate the flows Qjkn
    Qjkn = zeros(graph.J, graph.J, param.N)
    temp .= kappa ./ ((1 + beta) * PCj .* Qjk .^ (1+beta-nu))
    temp[nadj .| .!isfinite.(temp)] .= 0 # Because of the max() clause, Qjk may be zero
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = max.(Lambda' - Lambda, 0) # P^n_k - P^n_j (non-negative flows)
        Qjkn[:, :, n] = ((LL .* temp) / m[n]) .^ (1/(nu-1))
    end

    # Now calculating consumption bundle pre-transport cost
    temp .= Qjk .^ (1+beta) ./ kappa
    temp[nadj] .= 0
    Dj = Cj + sum(temp, dims = 2)
    Djn = Dj .* (Pjn ./ PCj) .^ (-param.sigma) 
    
    return (Pjn=Pjn, PCj=PCj, Ljn=Ljn, Yjn=Yjn, cj=cj, Cj=Cj, Dj=Dj, Djn=Djn, Qjk=Qjk, Qjkn=Qjkn)
end



# # Hessian Experimental:
#         # These are the lagrange multipliers = prices
#         Lambda = repeat(x, 1, graph.J * param.N) # P^n_j: each column is a price, the rows should be the derivatives (P^n'_k)
#         lambda = reshape(x, (graph.J, param.N))

#         # Compute price index
#         P = sum(lambda .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma))
#         mat_P = repeat(P, param.N, param.N * graph.J)

#         # Create masks
#         # This lets different products in the same location relate
#         Iij = kron(ones(param.N, param.N), I(graph.J))
#         # This is adjacency, but only for the same product (n == n')
#         Inm = kron(I(param.N), graph.adjacency)

#         # Compute Qjkn terms for Hessian
#         Qjknprime = zeros(graph.J, graph.J, param.N)





#         diff = Lambda' - Lambda # P^n'_k - P^n_j
#         mat_kappa = repeat(kappa, param.N, param.N)
#         mN = repeat(param.m, inner = graph.J)
#         # Rows are the derivatives, columns are P^n_j
#         PCjN = repeat(res.PCj, param.N)
#         A = mat_kappa ./ ((1+param.beta) * mN .* PCjN)'
#         # Adding term for (block-digonal) elements
#         temp = diff .* (x .^ (-param.sigma) .* PCjN .^ (param.sigma - 1))' .- 1
#         temp[Iij .== 0] .= 1
#         Aprime = A .* temp 
#         temp = diff .* Inm
#         temp[Inm .== 0] .= 1
#         A .*= temp

#         BA = A .^ (1/(nu-1))
#         BprimeAAprime = (1/(nu-1)) * A .^ (nu/(1-nu)) .* Aprime
#         BprimeAAprime[isnan.(BprimeAAprime)] .= 0

#         C = repeat(res.Qjk .^ ((nu-param.beta-1)/(nu-1)), param.N, param.N) 
#         Cprime = repeat(((nu-param.beta-1)/(nu-1)) * res.Qjk .^ (param.beta/(1-nu)), param.N, param.N)

#         Qjknprime = BA .* Cprime .* BprimeAAprime .* mN # Off-diagonal (n') part
#         Qjknprime[Inm] += BprimeAAprime[Inm] .* C[Inm]  # Adding diagonal
#         # Compute Qjkn Sums

#         # Derivatives of Qjk
#         # Result should be nedg * N
#         Qjkprime = zeros(graph.J, graph.J, param.N)
#         temp = kappa ./ ((1 + param.beta) * res.PCj)
#         for n in 1:param.N
#             Lambda = repeat(Pjn[:, n], 1, graph.J)
#             Qjkprime[:,:,n] = m[n] / param.beta * res.Qjk .^ ((nu-nu*param.beta-1)/(nu-1))
#             LL = Lambda' - Lambda # P^n_k - P^n_j
#             LL[.!graph.adjacency] .= 0
#             Qjkprime[:,:,n] .*= ((LL .* temp) / m[n]) .^ (1/(nu-1)) # A^(1/(nu-1))
#             Qjkprime[:,:,n] .*= temp / m[n] # A'(P^n_k)
#             tril()
#         end
#         Qjk = (Qjk .* temp) .^ (nu-1)/(nu*param.beta)

#         Qjkprime = res.Qjk[graph.adjacency]
        

#         # This is for Djn, the standard trade part (excluding trade costs)
#         termA = -param.sigma * (repeat(P, param.N) .^ param.sigma .* x .^ (-(param.sigma + 1)) .* repeat(res.Cj, param.N))
#         part1 = Iij .* Lambda .^ (-param.sigma) .* Lambda' .^ (-param.sigma) .* mat_P .^ (2 * param.sigma)
#         termB = param.sigma * part1 ./ mat_P .* repeat(res.Cj, param.N, graph.J * param.N)
#         termC = part1 .* repeat(graph.Lj ./ (graph.omegaj .* param.usecond.(res.cj, graph.hj)), param.N, graph.J * param.N)
#         Cjn = Diagonal(termA[:]) + termB + termC 

#         # Now comes the part of Djn relating to trade costs
#         Djn_costs = 


#         Djn = Cjn + Djn_costs


#         # This is related to the flows terms in the standard case
        
#         # Off-diagonal P^n_k terms
#         termD = 1 / (param.beta * (1 + param.beta)^(1 / param.beta)) * Inm .* mat_kappa .^ (1 / param.beta) .* 
#                 abs.(diff) .^ (1 / param.beta - 1) .* 
#                 ((diff .> 0) .* Lambda' ./ Lambda .^ (1 + 1 / param.beta) + 
#                 (diff .< 0) .* Lambda ./ Lambda' .^ (1 + 1 / param.beta))
#         # Diagonal P^n_j terms: sum across kappa
#         termE = -1 / (param.beta * (1 + param.beta)^(1 / param.beta)) * Inm .* mat_kappa .^ (1 / param.beta) .* 
#                 abs.(diff) .^ (1 / param.beta - 1) .* 
#                 ((diff .> 0) .* Lambda' .^ 2 ./ Lambda .^ (2 + 1 / param.beta) + 
#                 (diff .< 0) ./ Lambda' .^ (1 / param.beta))
#         termE = sum(termE, dims=2)
