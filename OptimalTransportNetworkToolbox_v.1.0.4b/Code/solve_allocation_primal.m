%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,x]=solve_allocation_primal(...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case without labor mobility with
a primal approach (quasiconcave) (used when dual is not
twice-differentiable).
It DOES NOT use the autodifferentiation package Adigator to generate the 
functional inputs for IPOPT.

Arguments:
- x0: initial seed for the solver
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A)
- verbose: {true | false} tells IPOPT to display results or not


Results:
- results: structure of results (Cj,Qjkn,etc.)
- flag: flag returned by IPOPT
- x: returns the 'x' variable returned by IPOPT (useful for warm start)


-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}
function [results,flag,x]=solve_allocation_primal(x0,auxdata,verbose)

% ==================
% RECOVER PARAMETERS
% ==================

graph=auxdata.graph;
param=auxdata.param;

% check compatibility
if any(sum(param.Zjn>0,2)>1)
    error('%s.m: this code only supports one good at most per location. Use the ADiGator version instead.',mfilename);
end

if nargin<3
    verbose=true;
end

if isempty(x0)
    x0=[1e-6*ones(graph.J*param.N,1);zeros(graph.ndeg*param.N,1)];
        % only the case with at most one good produced per location is
        % coded when ADiGator is not used: no optimization on Ljn
end

% =================
% PARAMETRIZE IPOPT
% =================

% Init functions
funcs.objective = @(x) objective(x,auxdata);
funcs.gradient = @(x) gradient(x,auxdata);
funcs.constraints = @(x) constraints(x,auxdata);
funcs.jacobian = @(x) jacobian(x,auxdata);
funcs.jacobianstructure = @() sparse([eye(graph.J*param.N),kron(eye(param.N),auxdata.A~=0)]);

funcs.hessian = @(x,sigma,lambda) hessian(x,auxdata,sigma,lambda);
funcs.hessianstructure = @() sparse(tril([repmat(eye(graph.J),[param.N param.N]),zeros(graph.J*param.N,graph.ndeg*param.N);
          zeros(graph.ndeg*param.N,graph.J*param.N),eye(graph.ndeg*param.N)]));

% Options
options.lb = [1e-8*ones(graph.J*param.N,1);-inf*ones(graph.ndeg*param.N,1)];
options.ub = [inf*ones(graph.J*param.N,1);inf*ones(graph.ndeg*param.N,1)];
options.cl = -inf*ones(graph.J*param.N,1); % lower bound on constraint function
options.cu = 0*ones(graph.J*param.N,1); % upper bound on constraint function

% options.ipopt.hessian_approximation = 'limited-memory';
options.ipopt.max_iter = 2000;
% options.ipopt.linear_solver = 'mumps';
options.ipopt.ma57_pre_alloc = 3;
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

results.Pjn=reshape(info.lambda(1:graph.J*param.N),[graph.J param.N]); % vector of prices
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma)); % price of tradeable
results.welfare=-funcs.objective(x);

end % end of function


function results = recover_allocation(x,auxdata)
graph=auxdata.graph;
param=auxdata.param;

% Domestic absorption
results.Cjn=reshape(x(1:graph.J*param.N),[graph.J param.N]);

% Total availability of tradeable goods
results.Cj=sum(results.Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% Population
results.Lj=param.Lj;

% Working population
results.Ljn=(param.Zjn>0).*results.Lj(:,ones(param.N,1));

% Production
results.Yjn=param.Zjn.*results.Lj(:,ones(param.N,1)).^param.a;

% Consumption per capita
results.cj=results.Cj./results.Lj;
results.cj(results.Lj==0)=0;

% Non-tradeable per capita
results.hj=param.Hj./results.Lj;
results.hj(results.Lj==0)=0;

% Vector of welfare per location
results.uj=param.u(results.cj,results.hj);

% Trade flows
results.Qin=reshape(x(graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);

% recover the Qjkn's
results.Qjkn=zeros(graph.J,graph.J,param.N);
id=1;
for i=1:graph.J
    for j=1:length(graph.nodes{i}.neighbors)
        if graph.nodes{i}.neighbors(j)>i
            results.Qjkn(i,graph.nodes{i}.neighbors(j),:)=max(results.Qin(id,:),0);
            results.Qjkn(graph.nodes{i}.neighbors(j),i,:)=max(-results.Qin(id,:),0);            
            id=id+1;
        end
    end
end

end % end of function

function f = objective(x,auxdata)
param=auxdata.param;

results = recover_allocation(x,auxdata);

f = -sum(param.omegaj.*results.Lj.*param.u(results.cj,param.hj));

end

function g = gradient(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

results = recover_allocation(x,auxdata);

g=[-repmat(param.omegaj.*param.uprime(results.cj,param.hj),[param.N 1]).*results.Cjn(:).^(-1/param.sigma).*repmat(results.Cj.^(1/param.sigma),[param.N 1]);
    zeros(graph.ndeg*param.N,1)];

end

function cons = constraints(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
kappa_ex=auxdata.kappa_ex;

results = recover_allocation(x,auxdata);
Qin=reshape(x(graph.J*param.N+1:end),[graph.ndeg param.N]); % take Q in format [ndeg,N] not [J,J,N]

% balanced flow constraint
cons_Q = zeros(graph.J,param.N);
for n=1:param.N
    M=max(A.*(ones(graph.J,1)*sign(Qin(:,n))'),0);
    cons_Q(:,n) = results.Cjn(:,n)+A*Qin(:,n)-results.Yjn(:,n)+M*(abs(Qin(:,n)).^(1+param.beta)./kappa_ex);
end

% return whole vector of constraints
cons=cons_Q(:);
end

function j = jacobian(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
kappa_ex=auxdata.kappa_ex;

results = recover_allocation(x,auxdata);
Qin=reshape(x(graph.J*param.N+1:end),[graph.ndeg param.N]); % take Q in format [ndeg,N] not [J,J,N]

% balanced flow constraint
cons_Q = zeros(graph.J*param.N,graph.ndeg*param.N);
for n=1:param.N
    M=max(A.*(ones(graph.J,1)*sign(Qin(:,n))'),0);
    cons_Q((n-1)*graph.J+1:n*graph.J,(n-1)*graph.ndeg+1:n*graph.ndeg) = A+(1+param.beta)*M.*repmat((sign(Qin(:,n)).*abs(Qin(:,n)).^param.beta./kappa_ex)',[graph.J 1]);
end

cons_Q=[eye(graph.J*param.N),cons_Q];

% return full jacobian
j = sparse(cons_Q);
end

function h = hessian(x,auxdata,sigma_IPOPT,lambda_IPOPT)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
kappa_ex=auxdata.kappa_ex;

% recover variables
results = recover_allocation(x,auxdata);
Qin=reshape(x(graph.J*param.N+1:end),[graph.ndeg param.N]); % take Q in format [ndeg,N] not [J,J,N]
Pjn=reshape(lambda_IPOPT(1:graph.J*param.N),[graph.J param.N]);

Hcdiag=-sigma_IPOPT*(-1/param.sigma).*results.Cjn(:).^(-1/param.sigma-1)...
    .*repmat(param.omegaj.*results.Cj.^(1/param.sigma).*param.uprime(results.cj,param.hj),[param.N 1]);

CC=repmat(results.Cjn(:).^((-1)/param.sigma), [1 graph.J*param.N]);

Hcnondiag=-sigma_IPOPT*repmat(param.omegaj.*(1/param.sigma*param.uprime(results.cj,param.hj).*results.Cj.^(2/param.sigma-1)+1./param.Lj.*param.usecond(results.cj,param.hj).*results.Cj.^(2/param.sigma)),[param.N graph.J*param.N]).*CC.*(CC)';

mask=repmat(~eye(graph.J),[param.N param.N]);
Hcnondiag(mask)=0;

Hq=zeros(graph.ndeg*param.N,1);
if param.beta>0
    for n=1:param.N    
        Hq((n-1)*graph.ndeg + 1:n*graph.ndeg)=(1+param.beta)*param.beta*abs(Qin(:,n)).^(param.beta-1)./kappa_ex.*(sum(max((A.*repmat(Pjn(:,n),[1 graph.ndeg])).*repmat(sign(Qin(:,n))',[graph.J 1]),0),1))';
    end
end

h=sparse(tril([diag(Hcdiag)+Hcnondiag,zeros(graph.J*param.N,graph.ndeg*param.N);
               zeros(graph.ndeg*param.N,graph.J*param.N),diag(Hq)]));
end
