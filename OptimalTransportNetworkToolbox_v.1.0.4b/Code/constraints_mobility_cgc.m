%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

cons = constraints_mobility_cgc ( x, auxdata ):
constraint function used by ADiGator in the primal case, with labor mobility 
and cross-good congestion.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2018, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function cons = constraints_mobility_cgc(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
kappa_ex=auxdata.kappa_ex;
A=auxdata.A;
m=param.m;

% Extract optimization variables
u=x(1);
Djn=reshape(x(1+1:1+graph.J*param.N),[graph.J param.N]); % Consumption per good pre-transport cost
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));  % Aggregate consumption pre-transport cost
Qin_direct=reshape(x(1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]); % Flow in the direction of the edge
Qin_indirect=reshape(x(1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]); % Flow in edge opposite direction
Lj=x(1+graph.J*param.N+2*graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J);
cj=x(1+graph.J*param.N+2*graph.ndeg*param.N+graph.J+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J+graph.J);
Ljn=reshape(x(1+graph.J*param.N+2*graph.ndeg*param.N+2*graph.J+1:end),[graph.J param.N]);
Yjn=param.Zjn.*(Ljn.^param.a);

% Utility constraint (Lj*u <= ... )
cons_u = Lj.*u-(cj.*Lj/param.alpha).^param.alpha.*(param.Hj/(1-param.alpha)).^(1-param.alpha);

% balanced flow constraint
cons_Q = Djn+A*Qin_direct-A*Qin_indirect-Yjn;

% labor resource constraint
cons_L = sum(Lj)-1;

% Final good constraint

% Create the matrix B_direct (resp. B_indirect) of transport cost along the direction of the edge 
% (resp. in edge opposite direction)
B_direct=(sum(repmat(m',[graph.ndeg 1]).*Qin_direct.^param.nu,2)).^((param.beta+1)/param.nu)./kappa_ex;
B_indirect=(sum(repmat(m',[graph.ndeg 1]).*Qin_indirect.^param.nu,2)).^((param.beta+1)/param.nu)./kappa_ex;
% Write final good constraint
cons_c = cj.*Lj+max(A,0)*B_direct+max(-A,0)*B_indirect-Dj;

% Local labor constraint
cons_Ljn = sum(Ljn,2) - Lj ;

% return whole vector of constraints
cons=[cons_u(:);cons_Q(:);cons_L;cons_c;cons_Ljn];

end
