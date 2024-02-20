
"""
    objective_mobility_cgc(x, auxdata)

Objective function used by ADiGator in the primal case, with labor mobility and 
cross-good congestion.

# References
- "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

# License
This code is distributed under BSD-3 License.
"""
function objective_mobility_cgc(x, auxdata)
    return -x[1]
end


# Please note that in Julia, indexing is 1-based, so `x(1)` in Matlab becomes `x[1]` in Julia.