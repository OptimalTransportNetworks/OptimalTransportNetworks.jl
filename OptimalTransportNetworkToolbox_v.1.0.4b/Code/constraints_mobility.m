%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

cons = constraints_mobility ( x, auxdata ):
constraint function used by ADiGator in the primal case, with labor mobility, 
no cross-good congestion.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2018, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function cons = constraints_mobility(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
kappa_ex=auxdata.kappa_ex;
A=auxdata.A;

% Extract optimization variables
u=x(1);
Cjn=reshape(x(1+1:1+graph.J*param.N),[graph.J param.N]);
Qin=reshape(x(1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Lj=x(1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+graph.ndeg*param.N+graph.J);
Cj=sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));
Ljn=reshape(x(1+graph.J*param.N+graph.ndeg*param.N+graph.J+1:end),[graph.J param.N]);
Yjn=param.Zjn.*Ljn.^param.a;

% Utility constraint (Lj*u <= ... )
cons_u = Lj.*u-(Cj/param.alpha).^param.alpha.*(param.Hj/(1-param.alpha)).^(1-param.alpha);

% balanced flow constraints
cons_Q = zeros(graph.J,param.N);
for n=1:param.N
    M=max(A.*(ones(graph.J,1)*sign(Qin(:,n))'),0);
    cons_Q(:,n) = Cjn(:,n)+A*Qin(:,n)-Yjn(:,n)+M*(abs(Qin(:,n)).^(1+param.beta)./kappa_ex);
end

% labor resource constraint
cons_L = sum(Lj)-1;

% Local labor availability constraints ( sum Ljn <= Lj )
cons_Ljn = sum(Ljn,2) - Lj ;

% return whole vector of constraints
cons=[cons_u(:);cons_Q(:);cons_L;cons_Ljn];

end
