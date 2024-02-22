%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,lambda]=solve_allocation_by_duality_ADiGator (...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case without labor mobility with
a duality approach.
It uses the autodifferentiation package Adigator to generate the functional inputs for IPOPT.

Arguments:
- x0: initial seed for the solver (lagrange multipliers)
- funcs: contains the Adigator generated functions (objective, gradient,
hessian, constraints, jacobian)
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A)
- verbose: {true | false} tells IPOPT to display results or not

Results:
- results: structure of results (Cj,Qjkn,etc.)
- flag: flag returned by IPOPT
- lambda: returns the 'lambda' variable returned by IPOPT (useful for warm start)

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function [results,flag,x]=solve_allocation_by_duality_ADiGator(x0,auxdata,funcs,verbose)

% ==================
% RECOVER PARAMETERS
% ==================

graph=auxdata.graph;
param=auxdata.param;

if nargin<4
    verbose=true;
end

if isempty(x0)
    x0=linspace(1,2,graph.J)'*linspace(1,2,param.N);
    x0=x0(:);
    % the JN terms are Pjn - start with a guess such that all lambda's are
    % different
end

% =================
% PARAMETRIZE IPOPT
% =================

% Init functions (note: this step is crucial since auxdata has changed
% since call to ADiGator!)
funcs.objective = @(x) objective_duality(x,auxdata);
funcs.gradient = @(x) objective_duality_Grd(x,auxdata);
funcs.hessian = @(x,sigma_hess,lambda_hess) objective_duality_Hes(x,auxdata,sigma_hess,lambda_hess);

% Options
% Bounds on optimization variables
options.lb = 1e-6*ones(graph.J*param.N,1); 
options.ub = inf*ones(graph.J*param.N,1);

% options.ipopt.hessian_approximation = 'limited-memory';
% options.ipopt.nlp_scaling_method = 'none'; % avoids an issue I was having with scaling: sometimes IPOPT would return different solutions for a convex problem depending on initial condition. Due to weird scaling behavior which led to accept bad solutions.

if verbose==true
    options.ipopt.print_level = 5;
else
    options.ipopt.print_level = 0;
end

% =========
% RUN IPOPT
% =========
[x,info]=ipopt(x0,funcs,options);

% ==============
% RETURN RESULTS
% ==============

% return allocation
flag=info;
results=recover_allocation(x,auxdata);
results.welfare=funcs.objective(x);       % Total economy welfare

end % end of function

function results = recover_allocation(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
kappa_ex=auxdata.kappa_ex;
omegaj=param.omegaj;
kappa=auxdata.kappa;

% Population
results.Lj=param.Lj;

% Extract price vectors
results.Pjn=reshape(x,[graph.J param.N]);
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma));

% Calculate labor allocation
results.Ljn = zeros(graph.J,param.N);
if param.a<1
    results.Ljn=((results.Pjn.*param.Zjn).^(1/(1-param.a)))./repmat(sum((results.Pjn.*param.Zjn).^(1/(1-param.a)),2),[1 param.N]).*repmat(results.Lj,[1 param.N]);
    results.Ljn(param.Zjn==0)=0;
else
    [~,max_id]=max(results.Pjn.*param.Zjn,[],2);    
    results.Ljn((1:graph.J)' + (max_id-1)*graph.J)=param.Lj;
end
results.Yjn=param.Zjn.*(results.Ljn.^param.a);

% Calculate consumption
results.cj=param.alpha*(sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma))./omegaj).^(-1/(1+param.alpha*(param.rho-1))).*...
(param.hj/(1-param.alpha)).^-((1-param.alpha)*(param.rho-1)/(1+param.alpha*(param.rho-1)));
zeta=omegaj.*((results.cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(-param.rho).*((results.cj/param.alpha).^(param.alpha-1).*(param.hj/(1-param.alpha)).^(1-param.alpha));
cjn=(results.Pjn./zeta(:,ones(1,param.N))).^(-param.sigma).*results.cj(:,ones(1,param.N));
results.Cj=results.cj.*param.Lj;
results.Cjn=cjn.*repmat(param.Lj,[1 param.N]);

% Non-tradeable per capita
results.hj=param.hj;

% Vector of welfare per location
results.uj=param.u(results.cj,results.hj);

% Calculate Qin_direct (which is the flow in the direction of the edge)
% and Qin_indirect (flow in edge opposite direction)
Qin_direct=zeros(graph.ndeg,param.N);
Qin_indirect=zeros(graph.ndeg,param.N);
for n=1:param.N
   Qin_direct(:,n)=max(1/(1+param.beta)*kappa_ex.*(-A'*results.Pjn(:,n)./(max(A',0)*results.Pjn(:,n))),0).^(1/param.beta);
end
for n=1:param.N
   Qin_indirect(:,n)=max(1/(1+param.beta)*kappa_ex.*(A'*results.Pjn(:,n)./(max(-A',0)*results.Pjn(:,n))),0).^(1/param.beta);
end

% Calculate the flows Qin of dimension [Ndeg,N]
results.Qin=zeros(graph.ndeg,param.N);

for i=1:param.N*graph.ndeg
    if Qin_direct(i)>Qin_indirect(i)
        results.Qin(i)=Qin_direct(i);
    else
        results.Qin(i)=-Qin_indirect(i);
    end
end
% Qin=reshape(Qin,[graph.ndeg param.N]);

% Calculate the flows Qjkn of dimension [J,J,N]
results.Qjkn=zeros(graph.J,graph.J,param.N);
for n=1:param.N
    Lambda=repmat(results.Pjn(:,n), [1 graph.J]);
    LL=max(Lambda'-Lambda,0);
    LL(~graph.adjacency)=0;
    results.Qjkn(:,:,n)=(1/(1+param.beta)*kappa.*LL./Lambda).^(1/param.beta);
end

end % end of function







