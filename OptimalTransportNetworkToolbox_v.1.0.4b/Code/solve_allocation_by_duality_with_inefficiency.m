%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,lambda]=solve_allocation_by_duality (...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case without labor mobility with
a duality approach.
It DOES NOT use the autodifferentiation package Adigator to generate the
functional inputs for IPOPT. It uses instead gradient, jacobian and hessian
computed by hand.

Arguments:
- x0: initial seed for the solver (lagrange multipliers P_j^n)
- verbose: {true | false} tells IPOPT to display results or not
- funcs: contains the Adigator generated functions (objective, gradient,
hessian, constraints, jacobian)
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A)

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

function [results,flag,x]=solve_allocation_by_duality_with_inefficiency(x0,auxdata,verbose)

% ==================
% RECOVER PARAMETERS
% ==================

graph=auxdata.graph;
param=auxdata.param;

if nargin<3
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


% Init functions  
funcs.objective = @(x) objective(x,auxdata);
funcs.gradient = @(x) gradient(x,auxdata);
funcs.hessian = @(x,sigma_hess,lambda_hess) hessian(x,auxdata,sigma_hess,lambda_hess);
funcs.hessianstructure = @() sparse(tril(repmat(eye(graph.J),[param.N param.N])+kron(eye(param.N),graph.adjacency)));

% Options
options.lb = 1e-3*ones(graph.J*param.N,1); % 0 is sometimes violated for some reason and I get negative -1e9 lambda's
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
[x,info]=ipopt(x0(:),funcs,options);

% ==============
% RETURN RESULTS
% ==============

% return allocation
flag=info;
results=recover_allocation(x,auxdata);

% compute missing fields
results.hj=param.hj;
results.Lj=param.Lj;
results.welfare=funcs.objective(x);
results.uj = param.u(results.cj,results.hj);

end % end of function


function f = objective(x,auxdata)
% recover parameters
param=auxdata.param;
graph=auxdata.graph;
kappa=auxdata.kappa;

res=recover_allocation(x,auxdata);

x=reshape(x,[graph.J param.N]);

cost=graph.delta_tau_spillover.*res.Qjkn.^(1+param.beta)./repmat(kappa,[1 1 param.N]);
cost(res.Qjkn==0)=0; % deal with cases Qjkn=kappa=0

cons=sum(x.*(res.cjn.*param.Lj(:,ones(param.N,1))+squeeze(sum(res.Qjkn+cost-permute(res.Qjkn,[2 1 3]),2))...
    -param.Zjn.*param.F(res.Ljn,param.a)),2);

f=sum(param.omegaj.*param.Lj.*param.u(res.cj,param.hj)-cons);
end % end of function

function g = gradient(x,auxdata)
% recover parameters
param=auxdata.param;
graph=auxdata.graph;
kappa=auxdata.kappa;

res=recover_allocation(x,auxdata);

cost=graph.delta_tau_spillover.*res.Qjkn.^(1+param.beta)./repmat(kappa,[1 1 param.N]);
cost(res.Qjkn==0)=0; % deal with cases Qjkn=kappa=0

cons=res.cjn.*param.Lj(:,ones(param.N,1))+squeeze(sum(res.Qjkn+cost-permute(res.Qjkn,[2 1 3]),2))...
    -param.Zjn.*param.F(res.Ljn,param.a);

g = -cons(:);
end % end of function


function h = hessian(x,auxdata,sigma_hess,lambda_hess)
% recover parameters
param=auxdata.param;
graph=auxdata.graph;
kappa=auxdata.kappa;

% Recover allocation and format
res=recover_allocation(x,auxdata);
Lambda=repmat(x,[1 graph.J*param.N]);
lambda=reshape(x,[graph.J param.N]);


% ---------------
% Precalculations

% Cj = param.Lj.*cj;
P=sum(lambda.^(1-param.sigma),2).^(1/(1-param.sigma)); % price index
mat_P=repmat(P,[param.N graph.J*param.N]); % matrix of price indices of size (JxN,JxN)
Iij=repmat(eye(graph.J),[param.N param.N]); % mask selecting only the (i,n;j,m) such that i=j
Inm=kron(eye(param.N),graph.adjacency); % mask selecting only the (i,n;j,m) such that j is in N(i) and n=m

% Compute first diagonal terms coming from C^n_j: numerator
termA=-param.sigma*(repmat(P,[param.N 1]).^param.sigma.*lambda(:).^-(param.sigma+1) .* repmat(res.Cj,[param.N 1])); % diagonal vector of size JxN

% Compute the non-diagonal terms from C^n_j: denominator
termB=param.sigma*Iij.*Lambda.^-param.sigma.*(Lambda').^-param.sigma.*mat_P.^(2*param.sigma-1).*repmat(res.Cj,[param.N param.N*graph.J]);

% Compute the non-diagonal terms from C^n_j: term in L_i c_i
termC=Iij.*Lambda.^-param.sigma.*(Lambda').^-param.sigma.*mat_P.^(2*param.sigma).*repmat(param.Lj./(param.omegaj.*param.usecond(res.cj,param.hj)),[param.N graph.J*param.N]);

% Compute the non-diagonal terms from the constraint
diff=Lambda'-Lambda;
% mat_kappa=repmat(kappa,[param.N param.N]);
spillover=max(graph.delta_tau_spillover,permute(graph.delta_tau_spillover,[2 1 3]));
mat_kappa=repmat(kappa,[param.N param.N])./repmat(reshape(spillover,[graph.J graph.J*param.N]),[param.N 1]);
mat_kappa(repmat(graph.adjacency==0,[param.N param.N]))=0;
termD=1/(param.beta*(1+param.beta)^(1/param.beta))*Inm.*mat_kappa.^(1/param.beta).*abs(diff).^(1/param.beta-1).*...
    ((diff>0).*Lambda'./Lambda.^(1+1/param.beta)+(diff<0).*Lambda./(Lambda').^(1+1/param.beta));

% Compute the diagonal terms from the constraint
termE=-1/(param.beta*(1+param.beta)^(1/param.beta))*Inm.*mat_kappa.^(1/param.beta).*abs(diff).^(1/param.beta-1).*...
    ((diff>0).*(Lambda').^2./Lambda.^(2+1/param.beta)+(diff<0).*1./(Lambda').^(1/param.beta));
termE=sum(termE,2);

% Compute the term coming from Lj
if param.a==1
    X=0;
else
    denom=sum((lambda.*param.Zjn).^(1/(1-param.a)),2);
    Lambdaz=repmat(lambda(:).*param.Zjn(:),[1 graph.J*param.N]);
    X_nondiag=param.a/(1-param.a)*Iij.*repmat(param.Zjn(:),[1 graph.J*param.N]).*repmat(param.Zjn(:)',[graph.J*param.N 1]).*repmat(param.Lj.^param.a./denom.^(1+param.a),[param.N graph.J*param.N]).*...
        Lambdaz.^(param.a/(1-param.a)).*(Lambdaz').^(param.a/(1-param.a));
    X_diag=-param.a/(1-param.a)*repmat((param.Lj./denom).^param.a,[param.N 1]).*param.Zjn(:)./lambda(:).*(lambda(:).*param.Zjn(:)).^(param.a/(1-param.a));
    X=X_nondiag+diag(X_diag);
end

% Return hessian
h=-sigma_hess*(diag(termA)+termB+termC+termD+diag(termE)+X);
h=sparse(tril(h));
end % end of function 


function results = recover_allocation(x,auxdata)

% Extract parameters
param=auxdata.param;
graph=auxdata.graph;
omegaj=param.omegaj;
kappa=auxdata.kappa;
Lj=param.Lj;

% Extract price vectors
Pjn=reshape(x,[graph.J param.N]);
results.Pjn=Pjn;
results.PCj=sum(Pjn.^(1-param.sigma),2).^(1/(1-param.sigma));

% Calculate labor allocation
if param.a<1
    results.Ljn=((Pjn.*param.Zjn).^(1/(1-param.a)))./repmat(sum((Pjn.*param.Zjn).^(1/(1-param.a)),2),[1 param.N]).*repmat(Lj,[1 param.N]);
    results.Ljn(param.Zjn==0)=0;
else
    [~,max_id]=max(Pjn.*param.Zjn,[],2);    
    results.Ljn = zeros(graph.J,param.N);
    results.Ljn((1:graph.J)' + (max_id-1)*graph.J)=param.Lj;
end
results.Yjn=param.Zjn.*(results.Ljn.^param.a);

% Calculate consumption
results.cj=param.alpha*(sum(Pjn.^(1-param.sigma),2).^(1/(1-param.sigma))./omegaj).^(-1/(1+param.alpha*(param.rho-1))).*...
(param.hj/(1-param.alpha)).^-((1-param.alpha)*(param.rho-1)/(1+param.alpha*(param.rho-1)));
zeta=omegaj.*((results.cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(-param.rho).*((results.cj/param.alpha).^(param.alpha-1).*(param.hj/(1-param.alpha)).^(1-param.alpha));
results.cjn=(Pjn./zeta(:,ones(1,param.N))).^(-param.sigma).*results.cj(:,ones(1,param.N));
results.Cj=results.cj.*param.Lj;
results.Cjn=results.cjn.*repmat(param.Lj,[1 param.N]);

% Calculate the flows Qjkn of dimension [J,J,N]
results.Qjkn=zeros(graph.J,graph.J,param.N);
for n=1:param.N
    Lambda=repmat(Pjn(:,n), [1 graph.J]);
    LL=max(Lambda'-Lambda,0);
    LL(~graph.adjacency)=0;
    spillover=max(graph.delta_tau_spillover(:,:,n)+graph.delta_tau_spillover(:,:,n)',1e-8);
    results.Qjkn(:,:,n)=(1/(1+param.beta)*(kappa./spillover).*LL./Lambda).^(1/param.beta);
end

end % end of function
