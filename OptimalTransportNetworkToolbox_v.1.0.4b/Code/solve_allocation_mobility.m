%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,x]=solve_allocation_mobility(...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case with labor mobility with
a primal approach (quasiconcave). 
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

Copyright (c) 2017-2018, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}
function [results,flag,x]=solve_allocation_mobility(x0,auxdata,verbose)

% ==================
% RECOVER PARAMETERS
% ==================

graph=auxdata.graph;
param=auxdata.param;
A=auxdata.A;

% check compatibility
if any(sum(param.Zjn>0,2)>1)
    error('%s.m: this code only supports one good at most per location.',mfilename);
end

if nargin<3
    verbose=true;
end

if isempty(x0)
    x0=[0;1e-6*ones(graph.J*param.N,1);1e-4*ones(2*graph.ndeg*param.N,1);1/graph.J*ones(graph.J,1)];
        % only the case with at most one good produced per location is
        % coded when ADiGator is not used: no optimization on Ljn
        % x=[u,Cjn,Qin,Lj]
end

% =================
% PARAMETRIZE IPOPT
% =================

% Init functions
funcs.objective = @(x) objective(x);
funcs.gradient = @(x) gradient(x,auxdata);
funcs.constraints = @(x) constraints(x,auxdata);
funcs.jacobian = @(x) jacobian(x,auxdata);

cons_u  = [ones(graph.J,1),repmat(eye(graph.J),[1 param.N]),zeros(graph.J,2*graph.ndeg*param.N),eye(graph.J)];
cons_Q = [zeros(graph.J*param.N,1),eye(graph.J*param.N),kron(eye(param.N),A~=0),kron(eye(param.N),A~=0),repmat(eye(graph.J),[param.N 1])];
cons_L = [zeros(1,1+graph.J*param.N+2*graph.ndeg*param.N),ones(1,graph.J)];
funcs.jacobianstructure = @() sparse([cons_u;cons_Q;cons_L]);

funcs.hessian = @(x,sigma,lambda) hessian(x,auxdata,sigma,lambda);
funcs.hessianstructure = @() sparse(tril([zeros(1,1+graph.J*param.N + 2*graph.ndeg*param.N+graph.J);
          zeros(graph.J*param.N,1),repmat(eye(graph.J),[param.N param.N]),zeros(graph.J*param.N,2*graph.ndeg*param.N+graph.J);
          zeros(graph.ndeg*param.N,1+graph.J*param.N),eye(graph.ndeg*param.N),zeros(graph.ndeg*param.N,graph.ndeg*param.N+graph.J);
          zeros(graph.ndeg*param.N,1+graph.J*param.N+graph.ndeg*param.N),eye(graph.ndeg*param.N),zeros(graph.ndeg*param.N,graph.J);
          ones(graph.J,1),zeros(graph.J,graph.J*param.N + 2*graph.ndeg*param.N),eye(graph.J)]));

% Options
options.lb = [-inf;1e-6*ones(graph.J*param.N,1);1e-6*ones(2*graph.ndeg*param.N,1);1e-8*ones(graph.J,1)];
options.ub = [inf;inf*ones(graph.J*param.N,1);inf*ones(2*graph.ndeg*param.N,1);1*ones(graph.J,1)];
options.cl = [-inf*ones(graph.J,1);-inf*ones(graph.J*param.N,1);0]; % lower bound on constraint function
options.cu = [0*ones(graph.J,1);1e-3*ones(graph.J*param.N,1);0]; % upper bound on constraint function

% options.ipopt.hessian_approximation = 'limited-memory';
options.ipopt.max_iter = 2000;
% options.ipopt.linear_solver = 'mumps';
% options.ipopt.ma57_pre_alloc = 3;
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

results.Pjn=reshape(info.lambda(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]); % prices Pjn
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma)); % price of tradeable

end % end of function


function results = recover_allocation(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

% Welfare
results.welfare=x(1);

% Domestic absorption 
results.Cjn=reshape(x(1+1:1+graph.J*param.N),[graph.J param.N]);

% Total availability of tradeable good
results.Cj=sum(results.Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% Population
results.Lj=x(1+graph.J*param.N+2*graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J);

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
results.Qin_direct=reshape(x(1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
results.Qin_indirect=reshape(x(1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);

results.Qin=zeros(graph.ndeg*param.N,1);

for i=1:param.N*graph.ndeg
    if results.Qin_direct(i)>results.Qin_indirect(i)
        results.Qin(i)=results.Qin_direct(i)-results.Qin_indirect(i);
    else
        results.Qin(i)=results.Qin_direct(i)-results.Qin_indirect(i);
    end
end
results.Qin=reshape(results.Qin,[graph.ndeg param.N]);

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


function f = objective(x)

f = -x(1);

end

function g = gradient(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

g = zeros(1+graph.J*param.N+2*graph.ndeg*param.N+graph.J,1);
g(1)=-1;

end

function cons = constraints(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;

u=x(1);
Cjn=reshape(x(1+1:1+graph.J*param.N),[graph.J param.N]);
Qin_direct=reshape(x(1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);
Lj=x(1+graph.J*param.N+2*graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J);
Cj=sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));
Yjn=param.Zjn.*Lj(:,ones(param.N,1)).^param.a;

% constraint Lj*u <= ...
cons_u = Lj.*u-(Cj/param.alpha).^param.alpha.*(param.Hj/(1-param.alpha)).^(1-param.alpha);

% % balanced flow constraint
% cons_Q = zeros(graph.J,param.N);
% for n=1:param.N
%     M=max(A.*(ones(graph.J,1)*sign(Qin(:,n))'),0);
%     cons_Q(:,n) = Cjn(:,n)+A*Qin(:,n)-Yjn(:,n)+M*(abs(Qin(:,n)).^(1+param.beta)./kappa_ex);
% end

% ------------------------
% Balanced flow constraint
cons_Q = zeros(graph.J,param.N);
for n=1:param.N    
    cons_Q(:,n) = Cjn(:,n)+Apos*(Qin_direct(:,n).^(1+param.beta)./kappa_ex)+Aneg*(Qin_indirect(:,n).^(1+param.beta)./kappa_ex)+A*Qin_direct(:,n)-A*Qin_indirect(:,n)-Yjn(:,n);
end


% labor resource constraint
cons_L = sum(Lj)-1;

% return whole vector of constraints
cons=[cons_u(:);cons_Q(:);cons_L];

end

function j = jacobian(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;

u=x(1);
Cjn=reshape(x(1+1:1+graph.J*param.N),[graph.J param.N]);
Qin_direct=reshape(x(1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);
Lj=x(1+graph.J*param.N+2*graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J);
Cj=sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% constraint Lj*u
cons_u=zeros(graph.J,graph.J*param.N);
for n=1:param.N
    cons_u(:,(n-1)*graph.J+1:n*graph.J)=diag(-Cjn(:,n).^-(1/param.sigma).*Cj.^(1/param.sigma).*(Cj/param.alpha).^(param.alpha-1).*(param.Hj/(1-param.alpha)).^(1-param.alpha));
end

cons_u = [Lj,cons_u,zeros(graph.J,2*graph.ndeg*param.N),u*eye(graph.J)];

% balanced flow constraint
cons_Q = zeros(graph.J*param.N,2*graph.ndeg*param.N);
for n=1:param.N    
    cons_Q((n-1)*graph.J+1:n*graph.J,(n-1)*graph.ndeg+1:n*graph.ndeg) = A+(1+param.beta)*Apos.*repmat((Qin_direct(:,n).^param.beta./kappa_ex)',[graph.J 1]);
    cons_Q((n-1)*graph.J+1:n*graph.J,graph.ndeg*param.N+(n-1)*graph.ndeg+1:graph.ndeg*param.N+n*graph.ndeg) = -A+(1+param.beta)*Aneg.*repmat((Qin_indirect(:,n).^param.beta./kappa_ex)',[graph.J,1]);
end

cons_Z=zeros(graph.J*param.N,graph.J);
for n=1:param.N
    cons_Z ((n-1)*graph.J+((1:graph.J)-1)*graph.J*param.N+(1:graph.J))=-param.a*param.Zjn(:,n).*Lj.^(param.a-1);
end

cons_Q=[zeros(graph.J*param.N,1),eye(graph.J*param.N),cons_Q,cons_Z];

% labor resource constraint
cons_L = [zeros(1,1+graph.J*param.N+2*graph.ndeg*param.N),ones(1,graph.J)];

% return full jacobian
j = sparse([cons_u;cons_Q;cons_L]);
end

function h = hessian(x,auxdata,sigma,lambda)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;

Cjn=reshape(x(1+1:1+graph.J*param.N),[graph.J param.N]);
Qin_direct=reshape(x(1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);
Lj=x(1+graph.J*param.N+2*graph.ndeg*param.N+1:1+graph.J*param.N+2*graph.ndeg*param.N+graph.J);
Cj=sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

omegaj=lambda(1:graph.J);
Pjn=reshape(lambda(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]);

% Hcdiag=-param.alpha^(1-param.alpha)*(param.sigma-1)/param.sigma*Cjn(:).^((param.sigma-1)/param.sigma)...
%     .*repmat(omegaj.*Cj.^(param.alpha+1/param.sigma-1).*(param.Hj/(1-param.alpha)).^(1-param.alpha),[param.N 1])...
%     +Pjn(:).*Cjn(:);
Hcdiag=repmat(1/param.sigma*omegaj.*param.uprime(Cj,param.Hj).*(Cj).^(1/param.sigma),[param.N 1]).*(Cjn(:)).^(-1/param.sigma-1);

% CC=repmat(Cjn(:).^((param.sigma-1)/param.sigma), [1 graph.J*param.N]);
% 
% Hcnondiag=-param.alpha^(1-param.alpha)*(param.alpha+1/param.sigma-1)*...
%     repmat(omegaj.*Cj.^(param.alpha+2/param.sigma-2).*(param.Hj/(1-param.alpha)).^(1-param.alpha),[param.N graph.J*param.N])...
%     .*CC.*(CC');

CC=repmat(Cjn(:).^(-1/param.sigma), [1 graph.J*param.N]);

Hcnondiag=-repmat(omegaj.*param.usecond(Cj,param.Hj).*(Cj).^(2/param.sigma)+1/param.sigma*omegaj.*param.uprime(Cj,param.Hj).*(Cj).^(2/param.sigma-1),[param.N graph.J*param.N]).*CC.*(CC');

mask=repmat(~eye(graph.J),[param.N param.N]);
Hcnondiag(mask)=0;

Hqpos=zeros(graph.ndeg*param.N,1);
Hqneg=zeros(graph.ndeg*param.N,1);
for n=1:param.N        
    Hqpos((n-1)*graph.ndeg + 1:n*graph.ndeg)=(1+param.beta)*param.beta*Qin_direct(:,n).^(param.beta-1)./kappa_ex.*(Apos'*Pjn(:,n));
    Hqneg((n-1)*graph.ndeg + 1:n*graph.ndeg)=(1+param.beta)*param.beta*Qin_indirect(:,n).^(param.beta-1)./kappa_ex.*(Aneg'*Pjn(:,n));    
end

Hl=diag( sum(-param.a*(param.a-1)*Pjn.*repmat(Lj.^(param.a-2),[1 param.N]),2) );

h=sparse(tril([zeros(1,1+graph.J*param.N + 2*graph.ndeg*param.N+graph.J);
          zeros(graph.J*param.N,1),diag(Hcdiag)+Hcnondiag,zeros(graph.J*param.N,2*graph.ndeg*param.N+graph.J);
          zeros(graph.ndeg*param.N,1+graph.J*param.N),diag(Hqpos),zeros(graph.ndeg*param.N,graph.ndeg*param.N+graph.J);
          zeros(graph.ndeg*param.N,1+graph.J*param.N+graph.ndeg*param.N),diag(Hqneg),zeros(graph.ndeg*param.N,graph.J);
          omegaj,zeros(graph.J,graph.J*param.N + 2*graph.ndeg*param.N),Hl]));
end
