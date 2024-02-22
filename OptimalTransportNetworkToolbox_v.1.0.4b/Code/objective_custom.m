%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

cons = objective ( x, auxdata ):
objective function used by ADiGator in the custom case (in this example, corresponds
to primal case, no mobility, no cross-good congestion.)

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function f = objective_custom(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
omegaj=param.omegaj;

% Recover variables
Cjn=reshape(x(1:graph.J*param.N),[graph.J param.N]);
Cj=sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));
cj=Cj./param.Lj;
cj(param.Lj==0)=0;

% Define utility function
Uj=((cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(1-param.rho)/(1-param.rho);

% Write objective function
f = -sum(omegaj.*param.Lj.*Uj);
end
