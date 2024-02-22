%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

cons = objective_cgc ( x, auxdata ):
objective function used by ADiGator in the primal case, no mobility, 
with cross-good congestion.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function f = objective_cgc(x,auxdata)

param=auxdata.param;
graph=auxdata.graph;
omegaj=param.omegaj;

cj=x(graph.J*param.N+2*graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N+graph.J);

f = -sum(omegaj.*param.Lj.*((cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(1-param.rho)/(1-param.rho));

end
