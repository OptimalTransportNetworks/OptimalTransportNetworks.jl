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
    Ni = find_node(graph, 2, 3) 
    graph[:Zjn][Ni, 1] = 2 
    Ni = find_node(graph, 2, 1) 
    graph[:Zjn][Ni, 2] = 1 
    Ni = find_node(graph, 4, 4) 
    graph[:Zjn][Ni, 3] = 1 
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
    Qjkn = fill(zero(eltype(Pjn)), graph.J, graph.J, param.N)
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


function objective(x)
    return objective_duality(x, auxdata)
end

objective(x0)

# Gradient function = negative constraint
function gradient_duality(x::Vector{Float64}, auxdata)
    param = auxdata.param
    graph = auxdata.graph
    kappa = auxdata.kappa

    res = recover_allocation_duality(x, auxdata)

    # Compute transportation cost
    cost = res.Qjkn .^ (1 + param.beta) ./ repeat(kappa, 1, 1, param.N)
    cost[res.Qjkn .== 0] .= 0  # Deal with cases Qjkn=kappa=0

    # Compute constraint
    cons = res.cjn .* graph.Lj + dropdims(sum(res.Qjkn + cost - permutedims(res.Qjkn, (2, 1, 3)), dims=2), dims = 2) - res.Yjn

    return -cons[:]  # Flatten the array and store in grad_f
end


gradient_duality(x0, auxdata)

# Numeric Solutions
ForwardDiff.gradient(objective, x0)

# Difference
gradient_duality(x0, auxdata) ./ ForwardDiff.gradient(objective, x0) 


# Now the Hessian 
function hessian_structure_duality(auxdata)
    graph = auxdata.graph
    param = auxdata.param

    # Create the Hessian structure
    H_structure = tril(repeat(sparse(I(graph.J)), param.N, param.N) + kron(sparse(I(param.N)), sparse(graph.adjacency)))
        
    # Get the row and column indices of non-zero elements
    rows, cols, _ = findnz(H_structure)
    return rows, cols
end

function hessian_duality(
    x::Vector{Float64},
    rows::Vector{Int64},
    cols::Vector{Int64},
    values::Vector{Float64},
    auxdata
)
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
        values[ind] = -term
    end
    return values
end

rows, cols = hessian_structure_duality(auxdata)
cind = CartesianIndex.(rows, cols)
values = zeros(length(cind))

hessian_duality(x0, rows, cols, values, auxdata)
ForwardDiff.hessian(objective, x0)[cind]

# findnz(sparse(ForwardDiff.hessian(objective, x0)))[1]
# extrema(abs.(findnz(sparse(ForwardDiff.hessian(objective, x0)))[3]))
# sum(abs.(ForwardDiff.hessian(objective, x0)) .> 0.01)

ForwardDiff.hessian(objective, x0)
H = zeros(length(x0), length(x0))
H[cind] = hessian_duality(x0, rows, cols, values, auxdata)
H

# Ratios
hessian_duality(x0, rows, cols, values, auxdata) ./ ForwardDiff.hessian(objective, x0)[cind]
extrema(hessian_duality(x0, rows, cols, values, auxdata) ./ ForwardDiff.hessian(objective, x0)[cind])






