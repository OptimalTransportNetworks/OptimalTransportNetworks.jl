%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,x]=solve_allocation_mobility_cgc(...): 
this function solves the full allocation of Q and C given a matrix of
kappa (=I^gamma/delta_tau). It solves the case with labor mobility with
a primal approach (quasiconcave) in the cross-good congestion case. 
It DOES NOT use the autodifferentiation package Adigator.

Arguments:
- x0: initial seed for the solver
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A...)
- verbose: {true | false} tells IPOPT to display results or not

Results:
- results: structure of results (C,Q,etc.)
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
function [results,flag,x]=solve_allocation_mobility_cgc(x0,auxdata,verbose)

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
    C=1e-6;
    L=1/graph.J;
    x0=[0;C/L*ones(graph.J,1);C*ones(graph.J*param.N,1);1e-6*ones(2*graph.ndeg*param.N,1);L*ones(graph.J,1)];
    % the version coded by hand optimizes on u (1x1), cj (Jx1), Djn (JxN) and Qin (2xndegxN) direct and indirect,
    % and Lj (Jx1). Only the case with at most one good produced per location is coded when ADiGator is not used: no optimization on Ljn
end

% =================
% PARAMETRIZE IPOPT
% =================

% Init functions
funcs.objective = @(x) objective(x,auxdata);
funcs.gradient = @(x) gradient(x,auxdata);
funcs.constraints = @(x) constraints(x,auxdata);
funcs.jacobian = @(x) jacobian(x,auxdata);
funcs.jacobianstructure = @() sparse( [ones(graph.J,1),eye(graph.J),zeros(graph.J,graph.J*param.N+2*graph.ndeg*param.N),eye(graph.J);
        zeros(graph.J,1),eye(graph.J),kron(ones(1,param.N),eye(graph.J)),kron(ones(1,param.N),max(auxdata.A,0)),kron(ones(1,param.N),max(-auxdata.A,0)),zeros(graph.J,graph.J);
        zeros(graph.J*param.N,1+graph.J),eye(graph.J*param.N),kron(eye(param.N),auxdata.A~=0),kron(eye(param.N),auxdata.A~=0),repmat(eye(graph.J),[param.N 1]);
        zeros(1,1+graph.J+graph.J*param.N+2*graph.ndeg*param.N),ones(1,graph.J)]);

funcs.hessian = @(x,sigma,lambda) hessian(x,auxdata,sigma,lambda);
funcs.hessianstructure = @() tril( [ sparse(1,1+graph.J+graph.J*param.N+2*graph.ndeg*param.N), sparse(ones(1,graph.J));
                                     sparse(graph.J,1),eye(graph.J),sparse(graph.J, graph.J*param.N+2*graph.ndeg*param.N+graph.J); 
                                     sparse(graph.J*param.N,1+graph.J), repmat(speye(graph.J),[param.N param.N]), sparse(graph.J*param.N,2*graph.ndeg*param.N+graph.J);
                                     sparse(graph.ndeg*param.N,1+graph.J+graph.J*param.N), repmat(speye(graph.ndeg), [param.N param.N] ), sparse(graph.ndeg*param.N,graph.ndeg*param.N+graph.J);
                                     sparse(graph.ndeg*param.N,1+graph.J+graph.J*param.N+graph.ndeg*param.N),repmat(speye(graph.ndeg), [param.N param.N] ),sparse(graph.ndeg*param.N,graph.J);
                                     sparse(ones(graph.J,1)),sparse(graph.J,graph.J+graph.J*param.N+2*graph.ndeg*param.N),speye(graph.J)]);

% funcs.hessianstructure = @() sparse(tril( [ zeros(1,1+graph.J+graph.J*param.N+2*graph.ndeg*param.N), ones(1,graph.J);
%                                             zeros(graph.J,1),eye(graph.J),zeros(graph.J, graph.J*param.N+2*graph.ndeg*param.N+graph.J); 
%                                             zeros(graph.J*param.N,1+graph.J), repmat(eye(graph.J),[param.N param.N]), zeros(graph.J*param.N,2*graph.ndeg*param.N+graph.J);
%                                             zeros(graph.ndeg*param.N,1+graph.J+graph.J*param.N), repmat(eye(graph.ndeg), [param.N param.N] ), zeros(graph.ndeg*param.N,graph.ndeg*param.N+graph.J);
%                                             zeros(graph.ndeg*param.N,1+graph.J+graph.J*param.N+graph.ndeg*param.N),repmat(eye(graph.ndeg), [param.N param.N] ),zeros(graph.ndeg*param.N,graph.J);
%                                             ones(graph.J,1),zeros(graph.J,graph.J+graph.J*param.N+2*graph.ndeg*param.N),eye(graph.J)]));

% Options
options.lb = [-inf;1e-6*ones(graph.J,1);1e-6*ones(graph.J*param.N,1);1e-6*ones(2*graph.ndeg*param.N,1);1e-8*ones(graph.J,1)];
options.ub = [inf;inf*ones(graph.J,1);inf*ones(graph.J*param.N,1);inf*ones(2*graph.ndeg*param.N,1);inf*ones(graph.J,1)];
options.cl = [-inf*ones(graph.J*(2+param.N),1);0]; % lower bound on constraint function
options.cu = 0*ones(graph.J*(2+param.N)+1,1); % upper bound on constraint function

% options.ipopt.hessian_approximation = 'limited-memory';
options.ipopt.max_iter = 2000;

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
% results.Pjn=reshape(info.lambda(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]); % Price vector
results.omegaj=info.lambda(1:graph.J);
results.Pjn=reshape(info.lambda(2*graph.J+1:2*graph.J+graph.J*param.N),[graph.J param.N]);
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma));  % Aggregate price vector  
results.PHj=param.alpha/(1-param.alpha).*results.PCj.*results.cj./results.hj;
results.W=info.lambda(2*graph.J+graph.J*param.N+1);

end % end of function

function results = recover_allocation(x,auxdata)
graph=auxdata.graph;
param=auxdata.param;
% kappa=auxdata.kappa;

% Total economy welfare
results.welfare=x(1);

% Aggregate consumption 
results.Cj=x(1+1:1+graph.J);

% Population
results.Lj=x(1+graph.J+graph.J*param.N+2*graph.ndeg*param.N+1:end);

% Consumption per capita 
results.cj=results.Cj./results.Lj;
results.cj(results.Lj==0)=0; % catch errors for non-populated places

% Non tradable good per capita
results.hj=param.Hj./results.Lj;
results.hj(results.Lj==0)=0;

% Vector of welfare per location
results.uj=((results.cj/param.alpha).^param.alpha.*(results.hj/(1-param.alpha)).^(1-param.alpha)).^(1-param.rho)/(1-param.rho);

 % Working population 
results.Ljn=(param.Zjn>0).*results.Lj(:,ones(param.N,1));

% Production
results.Yjn=param.Zjn.*results.Lj(:,ones(param.N,1)).^param.a;

% Domestic absorption of good n
results.Djn=max(0,reshape(x(1+graph.J+1:1+graph.J+graph.J*param.N),[graph.J param.N]));

% Total availability of final good 
results.Dj=sum(results.Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!

% Trade flows
results.Qin_direct=reshape(x(1+graph.J+graph.J*param.N+1:1+graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
results.Qin_indirect=reshape(x(1+graph.J+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);

results.Qin=zeros(graph.ndeg*param.N,1);

for i=1:param.N*graph.ndeg
    if results.Qin_direct(i)>results.Qin_indirect(i)
        results.Qin(i)=results.Qin_direct(i)-results.Qin_indirect(i);
    else
        results.Qin(i)=results.Qin_direct(i)-results.Qin_indirect(i);
    end
end
results.Qin=reshape(results.Qin,[graph.ndeg param.N]);

% recover the Q's
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

% recover u from optimizing variable
u=x(1);

f = -u;

end

function g = gradient(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

g=[-1;zeros(graph.J+graph.J*param.N+2*graph.ndeg*param.N+graph.J,1)];

end

function cons = constraints(x,auxdata)
% t0=clock();
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;

% -----------------
% Recover variables
u=x(1);
Cj=x(1+1:1+graph.J);
Djn=reshape(x(1+graph.J+1:1+graph.J+graph.J*param.N),[graph.J param.N]);
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!
Qin_direct=reshape(x(1+graph.J+graph.J*param.N+1:1+graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(1+graph.J+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);
Lj=x(1+graph.J+graph.J*param.N+2*graph.ndeg*param.N + 1:end);
Yjn=param.Zjn.*Lj(:,ones(param.N,1)).^param.a;

% --------------------
% Utility equalization

cons_u = u*Lj-param.u(Cj,param.Hj);

% -----------------------
% Final good availability

cost_direct=Apos*(sum(repmat(param.m',[graph.ndeg 1]).*Qin_direct.^param.nu,2).^((param.beta+1)/param.nu)./kappa_ex);
cost_indirect=Aneg*(sum(repmat(param.m',[graph.ndeg 1]).*Qin_indirect.^param.nu,2).^((param.beta+1)/param.nu)./kappa_ex);

cons_C=Cj+cost_direct+cost_indirect-Dj;

% ------------------------
% Balanced flow constraint
cons_Q = zeros(graph.J,param.N);
for n=1:param.N    
    cons_Q(:,n) = Djn(:,n)+A*Qin_direct(:,n)-A*Qin_indirect(:,n)-Yjn(:,n);
end

% ------------------------
% Total labor availability

cons_L=sum(Lj)-1;

% return whole vector of constraints
cons=[cons_u;cons_C;cons_Q(:);cons_L];
% fprintf('Time spent in constraints()=%2.5f secs.\n',etime(clock(),t0));
end

function J = jacobian(x,auxdata)
% t0=clock();
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;
 
% -----------------
% Recover variables

u=x(1);
Cj=x(1+1:1+graph.J);
Djn=reshape(x(1+graph.J+1:1+graph.J+graph.J*param.N),[graph.J param.N]);
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!
Qin_direct=reshape(x(1+graph.J+graph.J*param.N+1:1+graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(1+graph.J+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);
Lj=x(1+graph.J+graph.J*param.N+2*graph.ndeg*param.N + 1:end);

% ----------------------------------------
% Compute jacobian of welfare equalization
% ----------------------------------------

J1 = [Lj,-diag(param.uprime(Cj,param.Hj)),zeros(graph.J,graph.J*param.N+2*graph.ndeg*param.N),u*eye(graph.J)];

% -------------------------------------------
% Compute jacobian of final good availability
% -------------------------------------------

% part corresponding to Djn
JD=zeros(graph.J,graph.J*param.N);
for n=1:param.N
    JD(:,graph.J*(n-1)+1:graph.J*n)=-diag(Dj.^(1/param.sigma).*Djn(:,n).^(-1/param.sigma));
end

% part corresponding to Q

matm=repmat(param.m',[graph.ndeg 1]);

costpos=sum(matm.*Qin_direct.^param.nu,2).^((param.beta+1)/param.nu-1)./kappa_ex;
costneg=sum(matm.*Qin_indirect.^param.nu,2).^((param.beta+1)/param.nu-1)./kappa_ex; % kappa(j,k) = kappa(k,j) by symmetry

JQpos=zeros(graph.J,graph.ndeg*param.N);
JQneg=zeros(graph.J,graph.ndeg*param.N);
for n=1:param.N    
    vecpos=(1+param.beta)*costpos.*param.m(n).*Qin_direct(:,n).^(param.nu-1);    
    vecneg=(1+param.beta)*costneg.*param.m(n).*Qin_indirect(:,n).^(param.nu-1);
    
    JQpos(:,graph.ndeg*(n-1)+1:graph.ndeg*n)=Apos.*repmat(vecpos',[graph.J 1]);
    JQneg(:,graph.ndeg*(n-1)+1:graph.ndeg*n)=Aneg.*repmat(vecneg',[graph.J 1]);
end

J2 = [zeros(graph.J,1),eye(graph.J),JD,JQpos,JQneg,zeros(graph.J,graph.J)];

% ------------------------------------------------
% Compute jacobian of flow conservation constraint
% ------------------------------------------------

% part related to L
JL=zeros(graph.J*param.N,graph.J);
id=1:graph.J; 
for n=1:param.N       
    
    x=(n-1)*graph.J+1;
    y=1;
    offset=x-1+graph.J*param.N*(y-1);
     
    JL( offset + id+graph.J*param.N*(id-1)) = -param.a.*param.Zjn(:,n).*Lj.^(param.a-1);
end

J3 = [zeros(graph.J*param.N,1+graph.J),eye(graph.J*param.N),kron(eye(param.N),A),kron(eye(param.N),-A),JL];

% --------------------------------------------
% Compute jacobian of total labor availability
% --------------------------------------------

J4 = [zeros(1,1+graph.J+graph.J*param.N+2*graph.ndeg*param.N),ones(1,graph.J)];

% return full jacobian
J=sparse([J1;J2;J3;J4]);
% fprintf('Time spent in jacobian()=%2.5f secs.\n',etime(clock(),t0));
end

function H = hessian(x,auxdata,sigma_IPOPT,lambda_IPOPT)
% This code has been optimized to exploit the sparse structure of the
% hessian.

param=auxdata.param;
graph=auxdata.graph;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;

% -----------------
% Recover variables

u=x(1);
Cj=x(1+1:1+graph.J);
Djn=reshape(x(1+graph.J+1:1+graph.J+graph.J*param.N),[graph.J param.N]);
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!
Qin_direct=reshape(x(1+graph.J+graph.J*param.N+1:1+graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(1+graph.J+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);
Lj=x(1+graph.J+graph.J*param.N+2*graph.ndeg*param.N + 1:end);

omega=lambda_IPOPT(1:graph.J);
lambda=lambda_IPOPT(graph.J+1:2*graph.J);
Pjn=reshape(lambda_IPOPT(2*graph.J+1:2*graph.J+graph.J*param.N),[graph.J param.N]);

% preallocation of sparse matrix for maximum speed
sz=1+ graph.J + graph.J*param.N + 2*graph.ndeg*param.N + graph.J;
H=spalloc(sz,sz,graph.J + graph.J + graph.J*param.N^2 + 2*graph.ndeg*param.N^2 + graph.J);

% -----------------------------
% Part of Hessian related to Lu

id=1:graph.J;
x=1+graph.J+graph.J*param.N+2*graph.ndeg*param.N+1;
y=1;
offset=x-1+sz*(y-1);
H( offset + id + sz*(1-1) )=omega;

% -----------------------------------------
% Diagonal part of Hessian respective to Cj

HC = -omega.*param.usecond(Cj,param.Hj);
id=1:graph.J;
x=2;
y=2;
offset=x-1+sz*(y-1);
H( offset + id + sz*(id-1) ) = HC; % assign along the diagonal 

% -------------------------------------
% Diagonal of Hessian respective to Djn

HDdiag=repmat(lambda/param.sigma.*Dj.^(1/param.sigma),[param.N 1]).*Djn(:).^(-1/param.sigma - 1);

id=1:graph.J*param.N;
x=1+graph.J+1;
y=1+graph.J+1;
offset=x-1+sz*(y-1);
H( offset + id + sz*(id-1) ) = HDdiag;

% -------------------------------------
% Diagonal of Hessian respective to Qin

matm=repmat(param.m',[graph.ndeg 1]);
costpos=sum(matm.*Qin_direct.^param.nu,2); % ndeg x 1 vector of congestion cost
costneg=sum(matm.*Qin_indirect.^param.nu,2);

if param.nu>1 % if nu=1, diagonal term disappears    
    matpos=repmat( (1+param.beta)*(param.nu-1)*(Apos'*lambda).*costpos.^((param.beta+1)/param.nu-1)./kappa_ex, [1 param.N]).* repmat(param.m',[graph.ndeg 1]).* Qin_direct.^(param.nu-2);
    matneg=repmat( (1+param.beta)*(param.nu-1)*(Aneg'*lambda).*costneg.^((param.beta+1)/param.nu-1)./kappa_ex, [1 param.N]).* repmat(param.m',[graph.ndeg 1]).* Qin_indirect.^(param.nu-2);
    
    xpos = 1+graph.J+graph.J*param.N+1;
    ypos = 1+graph.J+graph.J*param.N+1;
    offset_pos = xpos-1+sz*(ypos-1);
    
    xneg = 1+graph.J+graph.J*param.N+graph.ndeg*param.N+1;
    yneg = 1+graph.J+graph.J*param.N+graph.ndeg*param.N+1;
    offset_neg = xneg-1+sz*(yneg-1);
    
    id = 1:graph.ndeg*param.N;
    
    H( offset_pos + id + sz*(id-1) ) = matpos(:);
    H( offset_neg + id + sz*(id-1) ) = matneg(:);
end

% ------------------------------------
% Diagonal of Hessian respective to Lj

HLL=-param.a*(param.a-1).*sum(Pjn.*param.Zjn,2).*Lj.^(param.a-2);

id=1:graph.J;
x=1+graph.J+graph.J*param.N+2*graph.ndeg*param.N+1;
y=1+graph.J+graph.J*param.N+2*graph.ndeg*param.N+1;
offset=x-1+sz*(y-1);
H( offset + id + sz*(id-1) ) = HLL;

% -----------------
% Nondiagonal parts

for n=1:param.N % row
    for m=1:param.N % col
        
        % -----------------
        % Respective to Djn
        
        HDnondiag=-lambda/param.sigma.*Dj.^(-(param.sigma-2)/param.sigma).*...
            Djn(:,n).^((-1)/param.sigma).*Djn(:,m).^((-1)/param.sigma);
        
        x = 1+graph.J + graph.J*(n-1)+1;
        y = 1+graph.J + graph.J*(m-1)+1;
        offset = x-1 + sz*(y-1);        
        id = 1:graph.J;
        H( offset + id + sz*(id-1) ) = H( offset + id + sz*(id-1) )+HDnondiag';
        
        % -----------------
        % Respective to Qin
        
        vecpos=(1+param.beta)*((1+param.beta)/param.nu-1)*param.nu*...
            (Apos'*lambda).*costpos.^((param.beta+1)/param.nu-2)./kappa_ex.*...
            param.m(n).*Qin_direct(:,n).^(param.nu-1).*...
            param.m(m).*Qin_direct(:,m).^(param.nu-1);
        
        vecneg=(1+param.beta)*((1+param.beta)/param.nu-1)*param.nu*...
            (Aneg'*lambda).*costneg.^((param.beta+1)/param.nu-2)./kappa_ex.*...
            param.m(n).*Qin_indirect(:,n).^(param.nu-1).*...
            param.m(m).*Qin_indirect(:,m).^(param.nu-1);
                
        xpos = 1+graph.J + graph.J*param.N + graph.ndeg*(n-1)+1;
        ypos = 1+graph.J + graph.J*param.N + graph.ndeg*(m-1)+1;        
        offset_pos = xpos-1 + sz*(ypos-1);
        
        xneg = 1+graph.J + graph.J*param.N + graph.ndeg*param.N + graph.ndeg*(n-1)+1;
        yneg = 1+graph.J + graph.J*param.N + graph.ndeg*param.N + graph.ndeg*(m-1)+1;        
        offset_neg = xneg-1 + sz*(yneg-1);
                
        id = 1:graph.ndeg;
        
        H( offset_pos + id + sz*(id-1) ) = H( offset_pos + id + sz*(id-1) )+vecpos';
        H( offset_neg + id + sz*(id-1) ) = H( offset_neg + id + sz*(id-1) )+vecneg';                
    end
end

% -------------------
% Return full hessian

H=tril( H );
end
