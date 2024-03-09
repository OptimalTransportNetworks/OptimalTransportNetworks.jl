# Please note that this translation assumes that the `adigatorOptions` and `adigatorGenFiles4Ipopt` functions have been appropriately translated to Julia and are available in the current scope. Also, the `create_auxdata` function should be translated and available in the current scope.

# Create the callbacks to be used by IPOPT
function ipopt_prepare(x, objective, constraints)

    # function find_nonzero_indices(A)
    #     rows = Int[]
    #     cols = Int[]
    #     for (index, value) in pairs(A)
    #         if value != 0
    #             push!(rows, index[1])
    #             push!(cols, index[2])
    #         end
    #     end
    #     return rows, cols
    # end
    function find_nonzero_indices(A)
        indices = findall(x -> x != 0, A)
        return getindex.(indices, 1), getindex.(indices, 2)
    end

    function eval_g(x::Vector{Float64}, g::Vector{Float64})
        # Bad: g    = zeros(2)  # Allocates new array
        # OK:  g[:] = zeros(2)  # Modifies 'in place'
        # g .= gradient(x)
        ForwardDiff.gradient!(g, objective, x)
    end

    jac = ForwardDiff.jacobian(constraints, x .+ 565.55)
    # Generate rows_jac and cols_jac vectors with indics of non-zero elements of jacobian
    rows_jac, cols_jac = find_nonzero_indices(jac)

    function eval_jac_g(
        x::Vector{Float64},
        rows::Vector{Int32},
        cols::Vector{Int32},
        values::Union{Nothing,Vector{Float64}},
    )
        if values === nothing
            rows[:] = rows_jac
            cols[:] = cols_jac
            # # Create dense sparsity structure
            # idx = 1
            # for i in 1:ncon
            #     for j in 1:nvar
            #         rows[idx] = i
            #         cols[idx] = j
            #         idx += 1
            #     end
            # end
        else
            # Compute Jacobian values
            ForwardDiff.jacobian!(values, constraints, x)
        end                    
    end

    hess = ForwardDiff.hessian(objective, x0)
    # Generate rows_jac and cols_jac vectors with indics of non-zero elements of jacobian
    rows_h, cols_h = find_nonzero_indices(hess)

    function hessian_lagrangian(x, lambda, sigma)
        H_f = ForwardDiff.hessian(objective, x)
        J_g = ForwardDiff.jacobian(constraints, x)
    
        # Start with the scaled Hessian of the objective function
        H_L = sigma .* H_f
    
        # Add the weighted sum of outer products of the Jacobian's rows
        for i in 1:size(J_g, 1)
            H_L .+= lambda[i] * J_g[i, :]' * J_g[i, :]
        end
    
        return H_L
    end

    
    function eval_h(x, nvar, obj_factor, lambda)
        # Create lower triangle sparsity structure 
        idx = 1
        for i in 1:nvar
            for j in 1:i
                rows[idx] = i
                cols[idx] = j
                idx += 1
            end
        end
    
        values = similar(rows)
        
        # Compute Hessian of Lagrangian 
        h = x -> obj_factor * objective(x) + transpose(lambda) * constraints(x)
        # -> very slow !!!
        ForwardDiff.hessian!(values, h, x)        
    end


    
end

# import FastDifferentiation as fd
# x = fd.make_variables(:x, length(x0))
# obj = h(x)
# sh = fd.sparse_hessian(obj, x0)