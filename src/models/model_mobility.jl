# The primal case, with labor mobility, no cross-good congestion.

function model_mobility(optimizer, auxdata)
    
    # Parameters and data
    param = dict_to_namedtuple(auxdata[:param])
    graph = auxdata[:graph]
    kappa_ex = auxdata[:kappa_ex]
    A = auxdata[:A]
    psigma = (param.sigma - 1) / param.sigma
    Hj = param.Hj

    # Model
    model = Model(optimizer)
    set_string_names_on_creation(model, false)

    # Variables + bounds
    @variable(model, u)                                    # Overall utility
    @variable(model, Cjn[1:graph.J, 1:param.N] >= 1e-8)    # Good specific consumption
    @variable(model, Qin[1:graph.ndeg, 1:param.N])         # Good specific flow
    @variable(model, 1e-8 <= Lj[1:graph.J] <= 1)           # Total labour
    @variable(model, Ljn[1:graph.J] >= 1e-8)               # Good specific labour

    # If custom: set start values
    if param.custom
        set_start_value(u, 0.0)
        set_start_value.(Cjn, 1.0e-6)
        set_start_value.(Qin, 0.0)
        set_start_value.(Lj, 1 / graph.J)
        set_start_value.(Ljn, 1 / (graph.J * param.N))
    end 

    # Objective
    @objective(model, Max, u)

    # Utility constraint (Lj*u <= ... )
    for j in 1:graph.J
        Cj = sum(Cjn[j, n]^psigma for n in 1:param.N)^(1 / psigma)
        @constraint(model, Lj[j] * u - (Cj / param.alpha)^param.alpha * (Hj[j] / (1 - param.alpha))^(1 - param.alpha) <= -1e-8)
    end

    # balanced flow constraints
    # Yjn = @expression(model, param.Zjn .* Ljn .^ param.a) # Same thing
    @expression(model, Yjn[j=1:graph.J, n=1:param.N], param.Zjn[j, n] * Ljn[j, n]^param.a)
    @constraint(model, [n in 1:param.N, j in 1:param.J],
        Cjn[j, n] + sum(A[j, i] * Qin[i, n] for i in 1:graph.ndeg) -
        Yjn[j, n] + sum(
            max(ifelse(Qin[i, n] > 0, A[j, i], -A[j, i]), 0) *
            abs(Qin[i, n])^(1 + param.beta) / kappa_ex[i]
            for i in 1:graph.ndeg
        ) <= -1e-8
    )

    # Labor resource constraint
    @constraint(model, -1e-8 <= sum(Lj) - 1 <= 1e8)

    # Local labor availability constraints ( sum Ljn <= Lj )
    @constraint(model, -1e-8 .<= sum(Ljn, dims=2) .- Lj .<= 1e-8)

    return model
end

