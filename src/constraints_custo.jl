
"""
constraints_custom(x, auxdata)

Constraint function used by ADiGator in the custom case (in this example, corresponds
to primal case, no mobility, no cross-good congestion).

# Arguments
- `x::Vector`: vector of decision variables
- `auxdata::Dict`: dictionary containing auxiliary data

# Returns
- `cons::Vector`: vector of constraint values
"""
function constraints_custom(x, auxdata)

    # Extract parameters
    param = auxdata["param"]
    graph = auxdata["graph"]
    kappa_ex = auxdata["kappa_ex"]
    A = auxdata["A"]
    Lj = param["Lj"]

    # Recover variables
    Cjn = reshape(x[1:graph["J"]*param["N"]], graph["J"], param["N"])
    Qin = reshape(x[graph["J"]*param["N"]+1:graph["J"]*param["N"]+graph["ndeg"]*param["N"]], graph["ndeg"], param["N"])
    Ljn = reshape(x[graph["J"]*param["N"]+graph["ndeg"]*param["N"]+1:end], graph["J"], param["N"])
    Yjn = param["Zjn"] .* (Ljn .^ param["a"])

    # Balanced flow constraints
    cons_Q = zeros(graph["J"], param["N"])
    for n in 1:param["N"]
        M = max.(A .* (ones(graph["J"], 1) * sign.(Qin[:, n])'), 0) # Matrix of dimension [J,Ndeg] taking value 1 if node J sends a good through edge Ndeg and 0 else 
        cons_Q[:, n] = Cjn[:, n] + A*Qin[:, n] - Yjn[:, n] + M * (abs.(Qin[:, n]) .^ (1 + param["beta"]) ./ kappa_ex)
    end

    # Local labor constraints
    cons_Ljn = sum(Ljn, dims=2) .- Lj

    # return whole vector of constraints
    cons = [cons_Q[:]; cons_Ljn[:]]
    return cons
end


# Please note that in Julia, the `*` operator performs matrix multiplication, and the `.*` operator performs element-wise multiplication. The `.` before a function name (like `.^` or `./`) applies the function element-wise. The `[:]` syntax is used to convert a matrix into a vector by stacking its columns. The `max.` function is used for element-wise maximum. The `sum` function with `dims=2` sums over the second dimension (columns) of the matrix.