
function objective_cgc(x, auxdata)

    param = auxdata.param
    graph = auxdata.graph
    omegaj = param.omegaj

    cj = x[graph.J*param.N+2*graph.ndeg*param.N+1 : graph.J*param.N+2*graph.ndeg*param.N+graph.J]

    f = -sum(omegaj .* param.Lj .* ((cj / param.alpha) .^ param.alpha .* (param.hj / (1 - param.alpha)) .^ (1 - param.alpha)) .^ (1 - param.rho) / (1 - param.rho))

    return f
end


# Please note that in Julia, array indexing starts from 1, unlike Matlab where it starts from 0. Also, the `end` keyword in Julia is equivalent to Matlab's `end` in array slicing. The `.` operator is used for element-wise operations in Julia, similar to Matlab.