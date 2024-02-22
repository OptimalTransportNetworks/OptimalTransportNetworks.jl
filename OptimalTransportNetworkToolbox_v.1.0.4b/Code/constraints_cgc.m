%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

cons = constraints_cgc ( x, auxdata ):
constraint function used by ADiGator in the primal case, no mobility, 
with cross-good congestion.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function cons = constraints_cgc(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
kappa_ex=auxdata.kappa_ex;
A=auxdata.A;
m=param.m;

% Recover labor allocation
Ljn=reshape(x(graph.J*param.N+2*graph.ndeg*param.N+graph.J+1:end),[graph.J param.N]);
Yjn=param.Zjn.*(Ljn.^param.a);

% Extract optimization variables
Qin_direct=reshape(x(graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]); % Flow in the direction of the edge
Qin_indirect=reshape(x(graph.J*param.N+graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]); % Flow in edge opposite direction
Djn=reshape(x(1:graph.J*param.N),[graph.J param.N]); % Consumption per good pre-transport cost
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % Aggregate consumption pre-transport cost
cj=x(graph.J*param.N+2*graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N+graph.J);

% Final good constraint

% Create the matrix B_direct (resp. B_indirect) of transport cost along the direction of the edge 
% (resp. in edge opposite direction)
B_direct=(sum(repmat(m',[graph.ndeg 1]).* Qin_direct.^param.nu,2)).^((param.beta+1)/param.nu)./kappa_ex;
B_indirect=(sum( repmat(m',[graph.ndeg 1]) .* Qin_indirect.^param.nu,2)).^((param.beta+1)/param.nu)./kappa_ex;

% Write final good constraint
cons_C = cj.*param.Lj+max(A,0)*B_direct+max(-A,0)*B_indirect-Dj;

% Balanced flow constraint
cons_Q=Djn+A*Qin_direct-A*Qin_indirect-Yjn;

% Labor allocation constraint
cons_Ljn=sum(Ljn,2)-param.Lj;

% return whole vector of constraints
cons=[cons_Q(:);cons_C;cons_Ljn];
end
