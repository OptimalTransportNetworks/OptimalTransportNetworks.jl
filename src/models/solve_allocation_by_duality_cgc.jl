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
    param = auxdata.param
    graph = auxdata.graph

    res = recover_allocation_duality_cgc(x, auxdata)
    Pjn = reshape(x, (graph.J, param.N))

    # Compute constraint
    cons = res.Djn + dropdims(sum(res.Qjkn - permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) - res.Yjn
    cons = sum(Pjn .* cons, dims=2)

    # Compute objective value
    f = sum(graph.omegaj .* graph.Lj .* param.u.(res.cj, graph.hj)) - sum(cons)
    return f # Negative objective because Ipopt minimizes  
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
        kappa = auxdata.kappa
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
        # N = param.N
        hess_str = auxdata.hess

        # Precompute elements
        res = recover_allocation_duality_cgc(x, auxdata)
        Pjn = res.Pjn
        PDj = res.PDj

        # Prepare computation of production function 
        if a != 1
            Psi = sum((Pjn .* Zjn) .^ (1 / (1 - a)), dims=2)
            psi = (Pjn .* Zjn) .^ (1 / (1 - a))
        end

        # Now the big loop over the hessian terms to compute
        ind = 0
        # https://stackoverflow.com/questions/38901275/inbounds-propagation-rules-in-julia
        for (jdnd, jn) in zip(hess_str[1], hess_str[2])
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

            # Needed for both Q^n_{jk} and D^n_j
            P = PDj[j]^m1dbeta
            Pprime = m1dbeta * Pjn[j, nd]^(-sigma) * P^(1+beta - beta*sigma)

            # Starting with D^n_j = (C+T)G, derivative: (C'+T')G + (C+T)G'
            # T' can be additively split. Here I compute (C' + T'_1)G + (C+T)G'
            G = (Pjn[j, n] / PDj[j])^(-sigma)
            if jd == j # 0 for P^n_k
                T = 0.0 
                for k in neighbors # TODO: add to loop below ?? 
                    T += res.Qjk[j, k]^(1+beta) / kappa[j, k]
                end
                Cprime = 1/omegaj[j] * Pjn[j, nd]^(-sigma) * PDj[j]^sigma * param.usecond(PDj[j]/omegaj[j], graph.hj[j])
                Gprime = sigma * Pjn[j, n] * Pjn[j, nd]^(-sigma) * PDj[j]^(sigma-2) * G^((sigma+1)/sigma)
                if nd == n
                    Gprime -= sigma / PDj[j] * G^((sigma+1)/sigma)
                end
                # Adding all derivative terms apart from second part of T'
                term += (Cprime + (1+beta) * Pprime / P * T) * G + (res.Cj[j] + T) * Gprime
            end
            # Constant term for iterating through neighbors for computing T'_2 G
            cons = numbetam1 / (numbetam1 + nu*beta) * P^(1+beta) * G
            # Cconstant tern: first part of Bprime
            cons2 = m1dbeta * n1dnum1 * numbetam1

            # Now terms sum_k(Q^n_{jk} - Q^n_{kj}) as well as T'_2 G
            for k in neighbors
                # Constant Terms
                K0 = (1 + beta) * kappa[j, k]
                PK0 = PDj[j] * K0
                K = K0^m1dbeta

                # Terms with Derivatives
                A = ((Pjn[k, nd] - Pjn[j, nd])/m[nd])^(1/(nu-1)) # 0 for n' != n
                B = (res.Qjk[j, k] * PK0^m1dbeta)^((nu-beta-1)/(nu-1))

                # Computing the right derivative: Q^n_{jk}
                if jd == j # P^x_j
                    Bprime = cons2 * A * B^((numbetam1 - nu*beta)/numbetam1)
                    if nd == n # P^n_j
                        Aprime = -n1dnum1 / m[n] * A^(2-nu)
                        term += K * (Pprime*A*B + P*Aprime*B + P*A*Bprime)
                    else # P^n'_j (A' is zero)
                        term += K * (Pprime*A*B + P*A*Bprime)
                    end
                    # This computes the remaining derivative of D^n_j (T'_2 G)
                    term += cons * K / (1+beta) * Bprime * B^(-nu*beta/(numbetam1+nu*beta))
                elseif jd == k # P^x_k (P' is zero)
                    Bprime = -cons2 * A * B^((numbetam1 - nu*beta)/numbetam1) # simply the negative
                    if nd == n # P^n_k
                        Aprime = n1dnum1 / m[n] * A^(2-nu) # Simply the negative 
                        term += K * (P*Aprime*B + P*A*Bprime)
                    else # # P^n'_k (A' is zero)
                        term += K * P*A*Bprime
                    end
                    # This computes the remaining derivative of D^n_j (T'_2 G)
                    term += cons * K / (1+beta) * Bprime * B^(-nu*beta/(numbetam1+nu*beta))
                # else
                #     error("This should not occur, jd != j and jd != k")
                end

                # Computing the right derivative: Q^n_{kj} (other direction -> needs to be subtracted)
                # TODO: are therse terms already negative??
                   
                # Also need prices here 
                Pk = PDj[k]^m1dbeta
                Pkprime = m1dbeta * Pjn[k, nd]^(-sigma) * Pk^(1+beta - beta*sigma)

                # Constant Terms
                K0 = (1 + beta) * kappa[k, j]
                PK0 = PDj[k] * K0
                K = K0^m1dbeta

                # Terms with Derivatives
                A = ((Pjn[j, nd] - Pjn[k, nd])/m[nd])^(1/(nu-1)) # 0 for n' != n
                B = (res.Qjk[k, j] * PK0^m1dbeta)^((nu-beta-1)/(nu-1))

                # Computing the right derivative: Q^n_{kj}
                if jd == k # P^x_k
                    Bprime = cons2 * A * B^((numbetam1 - nu*beta)/numbetam1)
                    if nd == n # P^n_k
                        Aprime = -n1dnum1 / m[n] * A^(2-nu)
                        term -= K * (Pkprime*A*B + Pk*Aprime*B + Pk*A*Bprime)
                    else # P^n'_k (A' is zero)
                        term -= K * (Pkprime*A*B + Pk*A*Bprime)
                    end
                elseif jd == j # P^x_j (P' is zero)
                    Bprime = -cons2 * A * B^((numbetam1 - nu*beta)/numbetam1) # simply the negative
                    if nd == n # P^n_j
                        Aprime = n1dnum1 / m[n] * A^(2-nu) # Simply the negative 
                        term -= K * (Pk*Aprime*B + Pk*A*Bprime)
                    else # # P^n'_j (A' is zero)
                        term -= K * Pk*A*Bprime
                    end
                # else
                #    error("This should not occur, jd != k and jd != j")
                end
            end # End of k loop

            # Finally: need to compute production function (X)
            if a != 1 && jd == j
                X = adam1 * Zjn[j, n] * Zjn[j, nd] * (psi[j, n] * psi[j, nd])^a / Psi[j]^(a+1) * Lj[j]^a
                if nd == n
                    X *= 1 - Psi[j] / psi[j, n]
                end
                term += X
            end

            # Assign result
            values[ind] = -obj_factor * term
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
    hj1malpha = (graph.hj / (1-param.alpha)) .^ (1-param.alpha)

    # Extract price vectors
    Pjn = reshape(x, (graph.J, param.N))
    PDj = sum(Pjn .^ (1 - param.sigma), dims=2) .^ (1 / (1 - param.sigma))

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
    cj = param.alpha * (PDj ./ omegaj) .^ 
         (-1 / (1 + param.alpha * (param.rho - 1))) .* 
         hj1malpha .^ (-(param.rho - 1)/(1 + param.alpha * (param.rho - 1)))
    Cj = cj .* Lj
        
    # Calculate the aggregate flows Qjk
    Qjk = zeros(graph.J, graph.J)
    temp = ((1 + param.beta) * PDj) .^ (-nu/(nu-1))
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = Lambda' - Lambda # P^n_k - P^n_j
        LL[.!graph.adjacency] .= 0
        Qjk += m[n] * ((LL .* kappa) ./ m[n]) .^ (nu/(nu-1))
    end
    Qjk = (Qjk .* temp) .^ (nu-1)/(nu*param.beta)
    Qjk[.!graph.adjacency] .= 0


    # Calculate the flows Qjkn
    Qjkn = zeros(graph.J, graph.J, param.N)
    temp = (1 + param.beta) * PDj .* Qjk .^ (1+param.beta-nu)
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = Lambda' - Lambda # P^n_k - P^n_j
        LL[.!graph.adjacency] .= 0
        Qjkn[:, :, n] = ((LL .* kappa) ./ (m[n] * temp)) .^ (1/(nu-1))
    end

    # Now calculating consumption bundle pre-transport cost
    temp = Qjk .^ (1+param.beta) ./ kappa
    temp[.!graph.adjacency] .= 0
    Dj = Cj + sum(temp, dims = 2)
    Djn = Dj .* (Pjn ./ PDj) .^ (-param.sigma) 
    
    return (Pjn=Pjn, PDj=PDj, Ljn=Ljn, Yjn=Yjn, cj=cj, Cj=Cj, Dj=Dj, Djn=Djn, Qjk=Qjk, Qjkn=Qjkn)
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
#         PDjN = repeat(res.PDj, param.N)
#         A = mat_kappa ./ ((1+param.beta) * mN .* PDjN)'
#         # Adding term for (block-digonal) elements
#         temp = diff .* (x .^ (-param.sigma) .* PDjN .^ (param.sigma - 1))' .- 1
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
#         temp = kappa ./ ((1 + param.beta) * res.PDj)
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
