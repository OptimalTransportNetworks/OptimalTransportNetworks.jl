
# using LinearAlgebra

function constraints_mobility(x, auxdata)

    # Extract parameters
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    kappa_ex = auxdata[:kappa_ex]
    A = auxdata[:A]

    # Extract optimization variables
    u = x[1]
    Cjn = reshape(x[2:graph.J*param.N+1], graph.J, param.N)
    Qin = reshape(x[graph.J*param.N+2:graph.J*param.N+graph.ndeg*param.N+1], graph.ndeg, param.N)
    Lj = x[graph.J*param.N+graph.ndeg*param.N+2:graph.J*param.N+graph.ndeg*param.N+graph.J+1]
    Cj = (sum(Cjn.^((param.sigma-1)/param.sigma), dims=2)).^(param.sigma/(param.sigma-1))
    Ljn = reshape(x[graph.J*param.N+graph.ndeg*param.N+graph.J+2:end], graph.J, param.N)
    Yjn = param.Zjn .* Ljn.^param.a

    # Utility constraint (Lj*u <= ... )
    cons_u = Lj .* u - (Cj / param.alpha).^param.alpha .* (param.Hj / (1 - param.alpha)).^(1 - param.alpha)

    # # balanced flow constraints
    # If `x` is numeric, create a matrix of zeros
    cons_Q = zeros(eltype(x), graph.J, param.N)
    for n in 1:param.N
        M = max.(A .* (ones(graph.J, 1) * sign.(Qin[:, n]')), 0)
        cons_Q[:, n] = Cjn[:, n] + A * Qin[:, n] - Yjn[:, n] + M * (abs.(Qin[:, n]).^(1 + param.beta) ./ kappa_ex)
    end
    # end

    # labor resource constraint
    cons_L = sum(Lj) - 1

    # Local labor availability constraints ( sum Ljn <= Lj )
    cons_Ljn = sum(Ljn, dims=2) .- Lj

    # return whole vector of constraints
    cons = vec(vcat(cons_u[:], cons_Q[:], cons_L, cons_Ljn))
    return cons
end


# Please note that in Julia, array indexing starts from 1, not 0 as in Matlab. Also, the `end` keyword in Julia is equivalent to Matlab's `end`. The `dims` argument in functions like `sum` and `max` specifies the dimensions to operate over. The `.` before functions like `*` and `^` is used for element-wise operations. The `vcat` function concatenates arrays vertically.

# Jump solution: 
function build_model(auxdata)
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    kappa_ex = auxdata[:kappa_ex]
    A = auxdata[:A]
    psigma = (param.sigma - 1) / param.sigma
    Hj = param.Hj

    # Model
    model = Model(Ipopt.Optimizer)

    # Variables + Bounds
    @variable(model, u)                                    # Overall utility
    # set_start_value(u, 0.0)
    @variable(model, Cjn[1:graph.J, 1:param.N] >= 1e-8)    # Good specific consumption
    # set_start_value.(Cjn, 1.0e-6)
    @variable(model, Qin[1:graph.ndeg, 1:param.N])         # Good specific flow
    # set_start_value.(Qin, 0.0)
    @variable(model, 1e-8 <= Lj[1:graph.J] <= 1)           # Total Labour
    # set_start_value.(Lj, 1 / graph.J)
    @variable(model, Ljn[1:graph.J] >= 1e-8)               # Good specific labour
    # set_start_value.(Ljn, 1 / (graph.J * param.N))
    @objective(model, Max, u)
    # Utility constraint (Lj*u <= ... )
    for j in 1:graph.J
        Cj = sum(Cjn[j, n]^psigma for n in 1:param.N)^(1 / psigma)
        @constraint(model, Lj[j] * u - (Cj / param.alpha)^param.alpha * (Hj[j] / (1 - param.alpha))^(1 - param.alpha) <= -1e-8)
    end
    # Cj = (sum(Cjn.^((param.sigma-1)/param.sigma), dims=2)).^(param.sigma/(param.sigma-1))
    # @constraint(model, Lj .* u .<= (Cj / param.alpha).^param.alpha .* (param.Hj / (1 - param.alpha)).^(1 - param.alpha))
    Yjn = param.Zjn .* Ljn.^param.a
    # balanced flow constraints
    # for n in 1:param.N
    #     M = max.(A .* (ones(graph.J, 1) * ifelse.(Qin[:, n]' .> 0, 1, -1)), 0) # sign.(Qin[:, n]') 
    #     @constraint(model, Cjn[:, n] + A * Qin[:, n] .<= Yjn[:, n] + M * (abs.(Qin[:, n]).^(1 + param.beta) ./ kappa_ex))
    # end
    @constraint(
        model, 
        [n in 1:param.N, j in 1:param.J],
        Cjn[j, n] + sum(A[j, i] * Qin[i, n] for i in 1:graph.ndeg) -
        Yjn[j, n] + sum(
            max(ifelse(Qin[i, n] > 0, A[j, i], -A[j, i]), 0) *
            abs(Qin[i, n])^(1 + param.beta) / kappa_ex[i]
            for i in 1:graph.ndeg
        ) <= -1e-8
    )
    # labor resource constraint
    @constraint(model, -1e-8 <= sum(Lj) - 1 <= 1e8)
    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, -1e-8 .<= sum(Ljn, dims=2) .- Lj .<= 1e-8)
    return model
end


function build_model(auxdata)
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    kappa_ex = auxdata[:kappa_ex]
    A = auxdata[:A]
    psigma = (param.sigma - 1) / param.sigma
    Hj = param.Hj

    # Model
    model = Model(Ipopt.Optimizer)

    # Variables + Bounds
    @variable(model, u)                                    # Overall utility
    # set_start_value(u, 0.0)
    @variable(model, Cjn[1:graph.J, 1:param.N] >= 1e-8)    # Good specific consumption
    # set_start_value.(Cjn, 1.0e-6)
    @variable(model, Qin[1:graph.ndeg, 1:param.N])         # Good specific flow
    # set_start_value.(Qin, 0.0)
    @variable(model, 1e-8 <= Lj[1:graph.J] <= 1)           # Total Labour
    # set_start_value.(Lj, 1 / graph.J)
    @variable(model, Ljn[1:graph.J] >= 1e-8)               # Good specific labour
    # set_start_value.(Ljn, 1 / (graph.J * param.N))
    @objective(model, Max, u)
    # Utility constraint (Lj*u <= ... )
    for j in 1:graph.J
        Cj = sum(Cjn[j, n]^psigma for n in 1:param.N)^(1 / psigma)
        @constraint(model, Lj[j] * u - (Cj / param.alpha)^param.alpha * (Hj[j] / (1 - param.alpha))^(1 - param.alpha) <= -1e-8)
    end
    # Cj = (sum(Cjn.^((param.sigma-1)/param.sigma), dims=2)).^(param.sigma/(param.sigma-1))
    # @constraint(model, Lj .* u .<= (Cj / param.alpha).^param.alpha .* (param.Hj / (1 - param.alpha)).^(1 - param.alpha))
    Yjn = param.Zjn .* Ljn.^param.a
    # balanced flow constraints
    # for n in 1:param.N
    #     M = max.(A .* (ones(graph.J, 1) * ifelse.(Qin[:, n]' .> 0, 1, -1)), 0) # sign.(Qin[:, n]') 
    #     @constraint(model, Cjn[:, n] + A * Qin[:, n] .<= Yjn[:, n] + M * (abs.(Qin[:, n]).^(1 + param.beta) ./ kappa_ex))
    # end
    @constraint(
        model, 
        [n in 1:param.N, j in 1:param.J],
        Cjn[j, n] + sum(A[j, i] * Qin[i, n] for i in 1:graph.ndeg) -
        Yjn[j, n] + sum(
            max(ifelse(Qin[i, n] > 0, A[j, i], -A[j, i]), 0) *
            abs(Qin[i, n])^(1 + param.beta) / kappa_ex[i]
            for i in 1:graph.ndeg
        ) <= -1e-8
    )
    # labor resource constraint
    @constraint(model, -1e-8 <= sum(Lj) - 1 <= 1e8)
    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, -1e-8 .<= sum(Ljn, dims=2) .- Lj .<= 1e-8)
    return model
end