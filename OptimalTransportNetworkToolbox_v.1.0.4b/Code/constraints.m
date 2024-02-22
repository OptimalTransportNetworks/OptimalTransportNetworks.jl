%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

cons = constraints ( x, auxdata ):
constraint function used by ADiGator in the primal case, no mobility, 
no cross-good congestion.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function cons = constraints(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
kappa_ex=auxdata.kappa_ex;
A=auxdata.A;
Lj=param.Lj;

% Recover variables
Cjn=reshape(x(1:graph.J*param.N),[graph.J param.N]);
Qin=reshape(x(graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Ljn=reshape(x(graph.J*param.N+graph.ndeg*param.N+1:end),[graph.J param.N]);
Yjn=param.Zjn.*(Ljn.^param.a);

% Balanced flow constraints
cons_Q = zeros(graph.J,param.N);
for n=1:param.N
    M=max(A.*(ones(graph.J,1)*sign(Qin(:,n))'),0); % Matrix of dimension [J,Ndeg] taking value 1 if node J sends a good through edge Ndeg and 0 else 
    cons_Q(:,n) = Cjn(:,n)+A*Qin(:,n)-Yjn(:,n)+M*(abs(Qin(:,n)).^(1+param.beta)./kappa_ex);
end

% Local labor constraints
cons_Ljn = sum(Ljn,2) - Lj;

% return whole vector of constraints
cons=[cons_Q(:);cons_Ljn];
end
