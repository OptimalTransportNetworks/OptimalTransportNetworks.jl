
using OptimalTransportNetworks
import OptimalTransportNetworks as otn
using LinearAlgebra
using SparseArrays
using ForwardDiff # Numeric Solutions

# Model Parameters
param = init_parameters(labor_mobility = false, K = 10, gamma = 1, beta = 1, verbose = true, 
                        N = 3, cross_good_congestion = true, nu = 2, rho = 0, duality = true) # , tol = 1e-4)
# Init network
map_size = 4
if map_size == 11
    graph = create_graph(param, 11, 11, type = "map") # create a map network of 11x11 nodes located in [0,10]x[0,10]
    # Customize graph
    graph[:Zjn] = fill(0.1, graph[:J], param[:N]) # set most places to low productivity
    Ni = find_node(graph, 6, 6) # Find index of the central node at (6,6)
    graph[:Zjn][Ni, 1] = 2 # central node more productive
    Ni = find_node(graph, 8, 2) 
    graph[:Zjn][Ni, 2] = 1 
    Ni = find_node(graph, 1, 10) 
    graph[:Zjn][Ni, 3] = 1 
end
if map_size == 4
    graph = create_graph(param, 4, 4, type = "map")  
    # Customize graph
    graph[:Zjn] = fill(0.1, graph[:J], param[:N]) 
    if param[:N] == 3
        Ni = find_node(graph, 2, 3) 
        graph[:Zjn][Ni, 1] = 2 
        Ni = find_node(graph, 2, 1) 
        graph[:Zjn][Ni, 2] = 1 
        Ni = find_node(graph, 4, 4) 
        graph[:Zjn][Ni, 3] = 1 
    else
        using Random: rand
        for i in 1:param[:N]
            Ni = find_node(graph, rand((1, 2, 3, 4)), rand((1, 2, 3, 4))) 
            graph[:Zjn][Ni, i] = rand() * 2
        end
    end
end
if map_size == 3
    graph = create_graph(param, 3, 3, type = "map")  
    # Customize graph
    graph[:Zjn] = fill(0.1, graph[:J], param[:N]) 
    Ni = find_node(graph, 2, 3) 
    graph[:Zjn][Ni, 1] = 2 
    Ni = find_node(graph, 2, 1) 
    graph[:Zjn][Ni, 2] = 1 
    Ni = find_node(graph, 3, 3) 
    graph[:Zjn][Ni, 3] = 1 
end

# Get Model
# param[:optimizer_attr] = Dict(:hsllib => "/usr/local/lib/libhsl.dylib", :linear_solver => "ma57") 
param[:duality] = true # Change to see dual/non-dual models

RN = param[:N] > 1 ? range(1, 2, length=param[:N])' : 1
x0 = vec(range(1, 2, length=graph[:J]) * RN)

edges = otn.represent_edges(otn.dict_to_namedtuple(graph))
I0 = Float64.(graph[:adjacency])
I0 *= param[:K] / sum(graph[:delta_i] .* I0) # 
# I0 = rescale_network!(param, graph, I0, Il, Iu)

# --------------
# INITIALIZATION

auxdata = otn.create_auxdata(otn.dict_to_namedtuple(param), otn.dict_to_namedtuple(graph), edges, I0)

# Recover allocation
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
    Qjk = fill(zero(eltype(temp)), graph.J, graph.J)
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = max.(Lambda' - Lambda, 0) # P^n_k - P^n_j (non-negative flows)
        Qjk += m[n] * (LL / m[n]) .^ (nu/(nu-1))
    end
    Qjk .^= ((nu-1)/nu)
    Qjk .*= temp
    Qjk .^= (1/beta)

    # Calculate the flows Qjkn
    temp = kappa ./ ((1 + beta) * PCj .* Qjk .^ (1+beta-nu))
    Qjkn = fill(zero(eltype(temp)), graph.J, graph.J, param.N)
    temp[nadj .| .!isfinite.(temp)] .= 0 # Because of the max() clause, Qjk may be zero
    for n in 1:param.N
        Lambda = repeat(Pjn[:, n], 1, graph.J)
        LL = max.(Lambda' - Lambda, 0) # P^n_k - P^n_j (non-negative flows)
        Qjkn[:, :, n] = ((LL .* temp) / m[n]) .^ (1/(nu-1))
    end

    # Now calculating consumption bundle pre-transport cost
    temp = Qjk .^ (1+beta) ./ kappa
    temp[nadj] .= 0
    Dj = Cj + sum(temp, dims = 2)
    Djn = Dj .* (Pjn ./ PCj) .^ (-param.sigma) 
    
    return (Pjn=Pjn, PCj=PCj, Ljn=Ljn, Yjn=Yjn, cj=cj, Cj=Cj, Dj=Dj, Djn=Djn, Qjk=Qjk, Qjkn=Qjkn)
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


function objective(x)
    return objective_duality_cgc(x, auxdata)
end

objective(x0)

# Gradient function = negative constraint
function gradient_duality_cgc(x::Vector{Float64}, auxdata)

    res = recover_allocation_duality_cgc(x, auxdata)

    # Compute constraint
    cons = res.Djn + dropdims(sum(res.Qjkn - permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) - res.Yjn

    return -cons[:]  # Flatten the array and store in grad_f
end

gradient_duality_cgc(x0, auxdata)

# Numeric Solution
ForwardDiff.gradient(objective, x0)

# Ratio
sum(abs.(gradient_duality_cgc(x0, auxdata) ./ ForwardDiff.gradient(objective, x0))) / length(x0)


# Now the Hessian 
function hessian_structure_duality_cgc(auxdata)
    graph = auxdata.graph
    param = auxdata.param

    # Create the Hessian structure
    H_structure = tril(repeat(sparse(I(graph.J)), param.N, param.N) + kron(sparse(ones(Int, param.N, param.N)), sparse(graph.adjacency))) # tril(repeat(sparse(I(graph.J)), param.N, param.N) + kron(sparse(I(param.N)), sparse(graph.adjacency)))
        
    # Get the row and column indices of non-zero elements
    rows, cols, _ = findnz(H_structure)
    return rows, cols
end

function hessian_duality_cgc(
    x::Vector{Float64},
    rows::Vector{Int64},
    cols::Vector{Int64},
    values::Vector{Float64},
    auxdata; 
    exact_algo = false
)
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

    # Constant term: first part of Bprime
    cons = m1dbeta * n1dnum1 * numbetam1

    # # Constant in T'
    if !exact_algo
        cons2 = numbetam1 / (numbetam1+nu*beta)
    else
        # New: constant in T' -> more accurate
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
    @inbounds for (jdnd, jn) in zip(rows, cols)
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
            Cprime = Lj[j]/omegaj[j] * (PCj[j] / Pjn[j, nd])^sigma / param.usecond(res.cj[j], graph.hj[j]) # param.uprimeinvprime(PCj[j]/omegaj[j], graph.hj[j])
            if nd == n
                Gprime -= sigma / PCj[j] * G^((sigma+1)/sigma)
            end
            term += Cprime * G + res.Cj[j] * Gprime
        end

        # Now terms sum_k(Q^n_{jk} - Q^n_{kj}) as well as T'G + TG'
        for k in neighbors
            # diff = Pjn[k, nd] - Pjn[j, nd]
            if Qjkn[j, k, n] > 0 # Flows in the direction of k
                T = Qjk[j, k]^(1 + beta) / kappa[j, k]
                PK0 = PCj[j] * (1 + beta) / kappa[j, k]
                KPABprime1 = cons * ((Pjn[k, nd] - Pjn[j, nd])/m[nd])^n1dnum1 * (PK0 * Qjk[j, k]^beta)^(-nu*n1dnum1)
                KPAprimeB1 = nd == n ? n1dnum1 / (Pjn[k, n] - Pjn[j, n]) : 0.0
                if jd == j
                    KPprimeAB1 = m1dbeta * Pjn[j, nd]^(-sigma) * PCj[j]^(sigma-1)
                    term += Qjkn[j, k, n] * (KPprimeAB1 - KPAprimeB1 + KPABprime1) # Derivative of Qjkn
                    if !exact_algo
                        term += T * (((1+beta) * KPprimeAB1 + cons2 * KPABprime1) * G + Gprime) # T'G + TG'
                    else
                        term += T * ((1+beta) * KPprimeAB1 * G + Gprime) # T'G (first part) + TG'
                        term += cons3 * Qjkn[j, k, nd] / PCj[j] * G # Second part of T'G
                    end
                elseif jd == k
                    term += Qjkn[j, k, n] * (KPAprimeB1 - KPABprime1) # Derivative of Qjkn
                    if !exact_algo
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
        values[ind] = -term # + 1e-6
    end
    return values
end

rows, cols = hessian_structure_duality_cgc(auxdata)
cind = CartesianIndex.(rows, cols)
values = zeros(length(cind))

hessian_duality_cgc(x0, rows, cols, values, auxdata, exact_algo = true)
ForwardDiff.hessian(objective, x0)[cind]

# findnz(sparse(ForwardDiff.hessian(objective, x0)))[1]
# extrema(abs.(findnz(sparse(ForwardDiff.hessian(objective, x0)))[3]))

ForwardDiff.hessian(objective, x0)
H = zeros(length(x0), length(x0))
H[cind] = hessian_duality_cgc(x0, rows, cols, values, auxdata, exact_algo = true)
H

# Ratios
hessian_duality_cgc(x0, rows, cols, values, auxdata, exact_algo = true) ./ ForwardDiff.hessian(objective, x0)[cind]
extrema(hessian_duality_cgc(x0, rows, cols, values, auxdata, exact_algo = true) ./ ForwardDiff.hessian(objective, x0)[cind])

# Differences
ad = abs.(hessian_duality_cgc(x0, rows, cols, values, auxdata, exact_algo = true) - ForwardDiff.hessian(objective, x0)[cind])
sum(ad)/length(ad)
sum(ad .> 0.01)
sort(ad, rev = true)[1:20]
cr = rows[ad .> 0.01] 
cc = cols[ad .> 0.01] 













