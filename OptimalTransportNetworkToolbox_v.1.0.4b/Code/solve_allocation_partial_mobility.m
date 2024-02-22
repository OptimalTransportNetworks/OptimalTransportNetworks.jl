     %{
 =================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, 2017-19
 =================================== version 1.0.4

[results,flag,x]=solve_allocation_partial_mobility(...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case with partial labor mobility 
with a primal approach (quasiconcave). 
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

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal and Dorian Henricot
pfajgelbaum@ucla.edu, eschaal@crei.cat, dorian.henricot@barcelonagse.eu

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}
function [results,flag,x]=solve_allocation_partial_mobility(x0,auxdata,verbose)

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
    x0=[zeros(param.nregions,1);1e-6*ones(graph.J*param.N,1);zeros(graph.ndeg*param.N,1);1/graph.J*ones(graph.J,1)];
    % order [u_r(:);Cjn(:);Qin(:);Lj(:)];
end

% =================
% PARAMETRIZE IPOPT
% =================

% Init functions
funcs.objective = @(x) objective(x,auxdata);
funcs.gradient = @(x) gradient(x,auxdata);
funcs.constraints = @(x) constraints(x,auxdata);
funcs.jacobian = @(x) jacobian(x,auxdata);

% build location matrix;RxJ matrix with 1 if location j is in region r, 0 otherwise
location = zeros(param.nregions,graph.J);
for i=1:param.nregions
location(i,:)=(graph.region==i);
end

cons_u  = [location',repmat(eye(graph.J),[1 param.N]),zeros(graph.J,graph.ndeg*param.N),eye(graph.J)];
cons_Q = [zeros(graph.J*param.N,param.nregions),eye(graph.J*param.N),kron(eye(param.N),A~=0),repmat(eye(graph.J),[param.N 1])];
cons_L = [zeros(param.nregions,param.nregions+graph.J*param.N+graph.ndeg*param.N),location];
funcs.jacobianstructure = @() sparse([cons_u;cons_Q;cons_L]);


funcs.hessian = @(x,sigma,lambda) hessian(x,auxdata,sigma,lambda);
funcs.hessianstructure = @() sparse(tril([zeros(param.nregions,param.nregions+graph.J*param.N + graph.ndeg*param.N+graph.J);
          zeros(graph.J*param.N,param.nregions),repmat(eye(graph.J),[param.N param.N]),zeros(graph.J*param.N,graph.ndeg*param.N+graph.J);
          zeros(graph.ndeg*param.N,param.nregions+graph.J*param.N),eye(graph.ndeg*param.N),zeros(graph.ndeg*param.N,graph.J);
          location',zeros(graph.J,graph.J*param.N + graph.ndeg*param.N+graph.J)]));


% Options
options.lb = [-inf*ones(param.nregions,1);-100*ones(graph.J*param.N,1);-inf*ones(graph.ndeg*param.N,1);1e-8*ones(graph.J,1)];
options.ub = [inf*ones(param.nregions,1);inf*ones(graph.J*param.N,1);inf*ones(graph.ndeg*param.N,1);1*ones(graph.J,1)];
options.cl = [-inf*ones(graph.J,1);-inf*ones(graph.J*param.N,1);0*ones(param.nregions,1)]; % lower bound on constraint function
options.cu = [0*ones(graph.J,1);1e-3*ones(graph.J*param.N,1);0*ones(param.nregions,1)]; % upper bound on constraint function

% options.ipopt.hessian_approximation = 'limited-memory';

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


%update
function results = recover_allocation(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

% Welfare
results.welfare=sum(param.omegar.*param.Lr.*x(1:param.nregions));

% Domestic absorption 
results.Cjn=reshape(exp(x(param.nregions+1:param.nregions+graph.J*param.N)),[graph.J param.N]);

% Total availability of tradeable good
results.Cj=sum(results.Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% Population
results.Lj=x(param.nregions+graph.J*param.N+graph.ndeg*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N+graph.J);

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
results.uj=((results.cj/param.alpha).^param.alpha.*(results.hj/(1-param.alpha)).^(1-param.alpha));

% Trade flows
results.Qin=reshape(x(param.nregions+graph.J*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);

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

f = -sum(param.omegar.*param.Lr.*x(1:param.nregions)); 

end

function g = gradient(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

g = zeros(param.nregions+graph.J*param.N+graph.ndeg*param.N+graph.J,1);
g(1:param.nregions)= - param.omegar.*param.Lr;

end

function cons = constraints(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
kappa_ex=auxdata.kappa_ex;

ur  =x(1:param.nregions);
Cjn =reshape(exp(x(param.nregions+1:param.nregions+graph.J*param.N)),[graph.J param.N]);
Qin =reshape(x(param.nregions+graph.J*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Lj  =x(param.nregions+graph.J*param.N+graph.ndeg*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N+graph.J);
Cj  =sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));
Yjn =param.Zjn.*Lj(:,ones(param.N,1)).^param.a;

% constraint Lj*ur <= ...
cons_u = Lj.*ur(graph.region)-(Cj/param.alpha).^param.alpha.*(param.Hj/(1-param.alpha)).^(1-param.alpha);

% balanced flow constraint
cons_Q = zeros(graph.J,param.N);
for n=1:param.N
    M=max(A.*(ones(graph.J,1)*sign(Qin(:,n))'),0);
    cons_Q(:,n) = Cjn(:,n)+A*Qin(:,n)-Yjn(:,n)+M*(abs(Qin(:,n)).^(1+param.beta)./kappa_ex);
end

% build location matrix;RxJ matrix with 1 if location j is in region r, 0 otherwise
location = zeros(param.nregions,graph.J);
for i=1:param.nregions
location(i,:)=(graph.region==i);  
end

% labor resource constraint
cons_L = sum(location.*Lj',2)-param.Lr;

% return whole vector of constraints
cons=[cons_u(:);cons_Q(:);cons_L(:)];

end

function j = jacobian(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
kappa_ex=auxdata.kappa_ex;

ur  =x(1:param.nregions);
Cjn =reshape(exp(x(param.nregions+1:param.nregions+graph.J*param.N)),[graph.J param.N]);
Qin =reshape(x(param.nregions+graph.J*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Lj  =x(param.nregions+graph.J*param.N+graph.ndeg*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N+graph.J);
Cj  =sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% build location matrix;RxJ matrix with 1 if location j is in region r, 0 otherwise
location = zeros(param.nregions,graph.J);
for i=1:param.nregions
location(i,:)=(graph.region==i);
end

cons_ur = location'.*Lj;          % JxR matrix equal to Lj in column r if j is in region r, 0 otherwise
cons_uL = diag(ur(graph.region)); % JxJ diagonal matrix, with utility of regions corresponding to row/column j on the diagonal

% constraint Lj*u
cons_u=zeros(graph.J,graph.J*param.N);
for n=1:param.N
    cons_u(:,(n-1)*graph.J+1:n*graph.J)=diag(-Cjn(:,n).*Cjn(:,n).^-(1/param.sigma).*Cj.^(1/param.sigma).*(Cj/param.alpha).^(param.alpha-1).*(param.Hj/(1-param.alpha)).^(1-param.alpha));
end

cons_u = [cons_ur,cons_u,zeros(graph.J,graph.ndeg*param.N),cons_uL];

% balanced flow constraint
cons_Q = zeros(graph.J*param.N,graph.ndeg*param.N);
for n=1:param.N
    M=max(A.*(ones(graph.J,1)*sign(Qin(:,n))'),0);
    cons_Q((n-1)*graph.J+1:n*graph.J,(n-1)*graph.ndeg+1:n*graph.ndeg) = A+(1+param.beta)*M.*repmat((sign(Qin(:,n)).*abs(Qin(:,n)).^param.beta./kappa_ex)',[graph.J 1]);
end

cons_Z=zeros(graph.J*param.N,graph.J);
for n=1:param.N
    cons_Z ((n-1)*graph.J+((1:graph.J)-1)*graph.J*param.N+(1:graph.J))=-param.a*param.Zjn(:,n).*Lj.^(param.a-1);
end

cons_Q=[zeros(graph.J*param.N,param.nregions),diag(Cjn(:)),cons_Q,cons_Z];

% labor resource constraint
cons_L = [zeros(param.nregions,param.nregions+graph.J*param.N+graph.ndeg*param.N),location];

% return full jacobian
j = sparse([cons_u;cons_Q;cons_L]);
end

function h = hessian(x,auxdata,sigma,lambda)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
kappa_ex=auxdata.kappa_ex;

% variables
Cjn =reshape(exp(x(param.nregions+1:param.nregions+graph.J*param.N)),[graph.J param.N]);
Qin =reshape(x(param.nregions+graph.J*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Lj  =x(param.nregions+graph.J*param.N+graph.ndeg*param.N+1:param.nregions+graph.J*param.N+graph.ndeg*param.N+graph.J);
Cj  =sum(Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% define multipliers
omegaj=lambda(1:graph.J);
Pjn=reshape(lambda(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]);


% build location matrix;RxJ matrix with 1 if location j is in region r, 0 otherwise
location = zeros(param.nregions,graph.J);
for i=1:param.nregions
location(i,:)=(graph.region==i);
end

HuL = location'.*omegaj;            % JxR matrix equal to multplier omegaj in column r if j is in location r, 0 otherwise



Hcdiag=-param.alpha^(1-param.alpha)*(param.sigma-1)/param.sigma*Cjn(:).^((param.sigma-1)/param.sigma)...
    .*repmat(omegaj.*Cj.^(param.alpha+1/param.sigma-1).*(param.Hj/(1-param.alpha)).^(1-param.alpha),[param.N 1])...
    +Pjn(:).*Cjn(:);

CC=repmat(Cjn(:).^((param.sigma-1)/param.sigma), [1 graph.J*param.N]);

Hcnondiag=-param.alpha^(1-param.alpha)*(param.alpha+1/param.sigma-1)*...
    repmat(omegaj.*Cj.^(param.alpha+2/param.sigma-2).*(param.Hj/(1-param.alpha)).^(1-param.alpha),[param.N graph.J*param.N])...
    .*CC.*(CC');

mask=repmat(~eye(graph.J),[param.N param.N]);
Hcnondiag(mask)=0;

Hq=zeros(graph.ndeg*param.N,1);
for n=1:param.N    
    Hq((n-1)*graph.ndeg + 1:n*graph.ndeg)=(1+param.beta)*param.beta*abs(Qin(:,n)).^(param.beta-1)./kappa_ex.*(sum(max((A.*repmat(Pjn(:,n),[1 graph.ndeg])).*repmat(sign(Qin(:,n))',[graph.J 1]),0),1))';
end

Hl=diag( sum(-param.a*(param.a-1)*Pjn.*repmat(Lj.^(param.a-2),[1 param.N]),2) );

h=sparse(tril([zeros(param.nregions,param.nregions+graph.J*param.N + graph.ndeg*param.N+graph.J);
          zeros(graph.J*param.N,param.nregions),diag(Hcdiag)+Hcnondiag,zeros(graph.J*param.N,graph.ndeg*param.N+graph.J);
          zeros(graph.ndeg*param.N,param.nregions+graph.J*param.N),diag(Hq),zeros(graph.ndeg*param.N,graph.J);
          HuL,zeros(graph.J,graph.J*param.N + graph.ndeg*param.N),Hl]));
end
