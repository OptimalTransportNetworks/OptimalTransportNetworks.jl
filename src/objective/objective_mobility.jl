
#=
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

objective ( x, auxdata ):
objective function used by ADiGator in the primal case, with labor mobility, 
no cross-good congestion.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
=#

function objective_mobility(x, auxdata)
    return -x[1]
end

# This Julia function takes two arguments, `x` and `auxdata`. `x` is expected to be an array, and the function returns the negative of the first element of `x`. The `auxdata` argument is not used in the function.