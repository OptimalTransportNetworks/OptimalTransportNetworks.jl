%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

cons = objective_duality ( x, auxdata ):
objective function used by ADiGator in the dual case, no mobility, 
no cross-good congestion.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function f = objective_duality(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
kappa_ex=auxdata.kappa_ex;
A=auxdata.A;
omegaj=param.omegaj;
Lj=param.Lj;

% Extract price vector
Pjn=reshape(x,[graph.J param.N]);

% Calculate consumption
cj=param.alpha*(sum(Pjn.^(1-param.sigma),2).^(1/(1-param.sigma))./omegaj).^(-1/(1+param.alpha*(param.rho-1))).*...
(param.hj/(1-param.alpha)).^-((1-param.alpha)*(param.rho-1)/(1+param.alpha*(param.rho-1)));
zeta=omegaj.*((cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(-param.rho).*((cj/param.alpha).^(param.alpha-1).*(param.hj/(1-param.alpha)).^(1-param.alpha));
cjn=(Pjn./zeta(:,ones(1,param.N))).^(-param.sigma).*cj(:,ones(1,param.N));

% Calculate Q, Qin_direct (which is the flow in the direction of the edge)
% and Qin_indirect (flow in edge opposite direction)
Qin_direct=zeros(graph.ndeg,param.N);
Qin_indirect=zeros(graph.ndeg,param.N);
for n=1:param.N
        Qin_direct(:,n)=max(1/(1+param.beta)*kappa_ex.*(-A'*Pjn(:,n)./(max(A',0)*Pjn(:,n))),0).^(1/param.beta);
end
for n=1:param.N
        Qin_indirect(:,n)=max(1/(1+param.beta)*kappa_ex.*(A'*Pjn(:,n)./(max(-A',0)*Pjn(:,n))),0).^(1/param.beta);
end

% Calculate labor allocation
Ljn=((Pjn.*param.Zjn).^(1/(1-param.a)))./repmat(sum((Pjn.*param.Zjn).^(1/(1-param.a)),2),[1 param.N]).*repmat(Lj,[1 param.N]);
Ljn(param.Zjn==0)=0;
Yjn=param.Zjn.*(Ljn.^param.a);

% Create flow constraint
cons=zeros(graph.J,param.N);
cons = cjn.*repmat(param.Lj,[1 param.N])+A*Qin_direct-A*Qin_indirect-Yjn+...
    max(A,0)*(Qin_direct.^(1+param.beta)./repmat(kappa_ex,[1 param.N]))+...
    max(-A,0)*(Qin_indirect.^(1+param.beta)./repmat(kappa_ex,[1 param.N]));
cons=sum(Pjn.*cons,2);

% Lagrangian
f=sum(omegaj.*param.Lj.*((cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^...
    (1-param.alpha)).^(1-param.rho)/(1-param.rho)-cons);
end 
