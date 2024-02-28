%{
 ==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,x]=solve_allocation_cgc(...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case without labor mobility with
a primal approach in the cross-good congestion case. 
It DOES NOT use the autodifferentiation package Adigator to generate the functional inputs for IPOPT.

Arguments:
- x0: initial seed for the solver
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A...)
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
function [results,flag,x]=solve_allocation_cgc(x0,auxdata,verbose)

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
    x0=[C/param.Lj(1)*ones(graph.J,1);C*ones(graph.J*param.N,1);1e-8*ones(2*graph.ndeg*param.N,1)]; % starting point
    % the version coded by hand optimizes on cj (Jx1), Cjn (JxN) and Qin (2xndegxN) direct and indirect,
    % only the case with at most one good produced per location is coded when ADiGator is not used: no optimization on Ljn
end

% =================
% PARAMETRIZE IPOPT
% =================

% Init functions
funcs.objective = @(x) objective(x,auxdata);
funcs.gradient = @(x) gradient(x,auxdata);
funcs.constraints = @(x) constraints(x,auxdata);
funcs.jacobian = @(x) jacobian(x,auxdata);
funcs.jacobianstructure = @() sparse([eye(graph.J),kron(ones(1,param.N),eye(graph.J)),kron(ones(1,param.N),max(auxdata.A,0)),kron(ones(1,param.N),max(-auxdata.A,0));
    zeros(graph.J*param.N,graph.J),eye(graph.J*param.N),kron(eye(param.N),auxdata.A~=0),kron(eye(param.N),auxdata.A~=0)]);

funcs.hessian = @(x,sigma,lambda) hessian(x,auxdata,sigma,lambda);
hessianstructure = [speye(graph.J),sparse(graph.J, graph.J*param.N+2*graph.ndeg*param.N); 
                    sparse(graph.J*param.N,graph.J), repmat(speye(graph.J),[param.N param.N]), sparse(graph.J*param.N,2*graph.ndeg*param.N);
                    sparse(graph.ndeg*param.N,graph.J+graph.J*param.N), repmat(speye(graph.ndeg), [param.N param.N] ), zeros(graph.ndeg*param.N,graph.ndeg*param.N);
                    sparse(graph.ndeg*param.N,graph.J+graph.J*param.N+graph.ndeg*param.N),repmat(speye(graph.ndeg), [param.N param.N] )];
funcs.hessianstructure = @() tril(hessianstructure);
                
% funcs.hessianstructure = @() sparse(tril( [eye(graph.J),zeros(graph.J, graph.J*param.N+2*graph.ndeg*param.N); 
%                                            zeros(graph.J*param.N,graph.J), repmat(eye(graph.J),[param.N param.N]), zeros(graph.J*param.N,2*graph.ndeg*param.N);
%                                            zeros(graph.ndeg*param.N,graph.J+graph.J*param.N), repmat(eye(graph.ndeg), [param.N param.N] ), zeros(graph.ndeg*param.N,graph.ndeg*param.N);
%                                            zeros(graph.ndeg*param.N,graph.J+graph.J*param.N+graph.ndeg*param.N),repmat(eye(graph.ndeg), [param.N param.N] )]));

% Options
options.lb = [1e-8*ones(graph.J,1);1e-6*ones(graph.J*param.N,1);1e-8*ones(2*graph.ndeg*param.N,1)];
options.ub = [inf*ones(graph.J,1);inf*ones(graph.J*param.N,1);inf*ones(2*graph.ndeg*param.N,1)];
options.cl = -inf*ones(graph.J*(1+param.N),1); % lower bound on constraint function
options.cu = 0*ones(graph.J*(1+param.N),1); % upper bound on constraint function

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

results.welfare=-funcs.objective(x);    % Total economy welfare
results.Pjn=reshape(info.lambda(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]);  % Price vector
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma)); % Aggregate price vector


end % end of function

function results = recover_allocation(x,auxdata)
graph=auxdata.graph;
param=auxdata.param;

% Consumption per capita
results.cj=x(1:graph.J);

% Total consumption
results.Cj=results.cj.*param.Lj;

% Domestic absorption of good n
results.Djn=max(0,reshape(x(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]));

% Total availability of tradeable good
results.Dj=sum(results.Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!

% Vector of welfare per location
results.uj=((results.cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(1-param.rho)/(1-param.rho);  

% Total population
results.Lj=param.Lj;

% Non-tradeable good per capita
results.hj=param.hj;

% Working population
results.Ljn=(param.Zjn>0).*results.Lj(:,ones(param.N,1));

% Production
results.Yjn=param.Zjn.*results.Lj(:,ones(param.N,1)).^param.a;

% trade flows
Qin_direct=reshape(x(graph.J+graph.J*param.N+1:graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(graph.J+graph.J*param.N+graph.ndeg*param.N+1:graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);

results.Qin=zeros(graph.ndeg*param.N,1);

for i=1:param.N*graph.ndeg
    if Qin_direct(i)>Qin_indirect(i)
        results.Qin(i)=Qin_direct(i)-Qin_indirect(i);
    else
        results.Qin(i)=Qin_indirect(i)-Qin_direct(i);
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

function f = objective(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

% recover cj from optimizing variable
cj=x(1:graph.J);

f = -sum(param.omegaj.*param.Lj.*param.u(cj,param.hj));

end

function g = gradient(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

% recover cj from optimizing variable
cj=x(1:graph.J);

g=[-param.omegaj.*param.Lj.*param.uprime(cj,param.hj);zeros(graph.J*param.N+2*graph.ndeg*param.N,1)];

end

function cons = constraints(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;

% -----------------
% Recover variables
cj=x(1:graph.J);
Djn=reshape(x(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]);
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!
Lj=param.Lj;
Yjn=param.Zjn.*Lj(:,ones(param.N,1)).^param.a;
Qin_direct=reshape(x(graph.J+graph.J*param.N+1:graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(graph.J+graph.J*param.N+graph.ndeg*param.N+1:graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);

% -----------------------
% Final good availability

cost_direct=Apos*(sum(repmat(param.m',[graph.ndeg 1]).*Qin_direct.^param.nu,2).^((param.beta+1)/param.nu)./kappa_ex);
cost_indirect=Aneg*(sum(repmat(param.m',[graph.ndeg 1]).*Qin_indirect.^param.nu,2).^((param.beta+1)/param.nu)./kappa_ex);

cons_C=cj.*param.Lj+cost_direct+cost_indirect-Dj;

% ------------------------
% Balanced flow constraint
cons_Q = zeros(graph.J,param.N);
for n=1:param.N    
    cons_Q(:,n) = Djn(:,n)+A*Qin_direct(:,n)-A*Qin_indirect(:,n)-Yjn(:,n);
end

% return whole vector of constraints
cons=[cons_C;cons_Q(:)];
end

function J = jacobian(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;
 
% -----------------
% Recover variables

Djn=reshape(x(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]);
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!
Qin_direct=reshape(x(graph.J+graph.J*param.N+1:graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(graph.J+graph.J*param.N+graph.ndeg*param.N+1:graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);

% -------------------------------------------
% Compute jacobian of final good availability
% -------------------------------------------

% part corresponding to Djn
JD=zeros(graph.J,graph.J*param.N);
for n=1:param.N
    JD(:,graph.J*(n-1)+1:graph.J*n)=-diag(Dj.^(1/param.sigma).*Djn(:,n).^(-1/param.sigma));
end

% part corresponding to Qjkn

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

J1 = [diag(param.Lj),JD,JQpos,JQneg];

% ------------------------------------------------
% Compute jacobian of flow conservation constraint
% ------------------------------------------------

J2 = [zeros(graph.J*param.N,graph.J),eye(graph.J*param.N),kron(eye(param.N),A),kron(eye(param.N),-A)];

% return full jacobian
J=sparse([J1;J2]);
end

function H = hessian(x,auxdata,sigma_IPOPT,lambda_IPOPT)
% This code has been optimized to exploit the sparse structure of the
% hessian.

param=auxdata.param;
graph=auxdata.graph;
A=auxdata.A;
Apos=auxdata.Apos;
Aneg=auxdata.Aneg;
kappa_ex=auxdata.kappa_ex;

% -----------------
% Recover variables
cj=x(1:graph.J);
Djn=reshape(x(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]);
Dj=sum(Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % total availability of final good, not consumption!
Qin_direct=reshape(x(graph.J+graph.J*param.N+1:graph.J+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);
Qin_indirect=reshape(x(graph.J+graph.J*param.N+graph.ndeg*param.N+1:graph.J+graph.J*param.N+2*graph.ndeg*param.N),[graph.ndeg param.N]);

% recover variables
lambda=lambda_IPOPT(1:graph.J);

% preallocation of sparse matrix for maximum speed
sz=graph.J + graph.J*param.N + 2*graph.ndeg*param.N;
H=spalloc(sz,sz,graph.J + graph.J*param.N^2 + 2*graph.ndeg*param.N^2);

% ----------------------------------------
% Diagonal part of Hessian respective to cj

Hc = -sigma_IPOPT*param.omegaj.*param.Lj.*param.usecond(cj,param.hj);
id=1:graph.J;
H( id + sz*(id-1) ) = Hc; % assign along the diagonal 

% -------------------------------------
% Diagonal of Hessian respective to Djn

HDdiag=repmat(lambda/param.sigma.*Dj.^(1/param.sigma),[param.N 1]).*Djn(:).^(-1/param.sigma - 1);

id=1:graph.J*param.N;
offset=graph.J+sz*graph.J;
H( offset + id + sz*(id-1) ) = HDdiag;

% -------------------------------------
% Diagonal of Hessian respective to Qin

matm=repmat(param.m',[graph.ndeg 1]);
costpos=sum(matm.*Qin_direct.^param.nu,2); % ndeg x 1 vector of congestion cost
costneg=sum(matm.*Qin_indirect.^param.nu,2);

if param.nu>1 % if nu=1, diagonal term disappears    
    matpos=repmat( (1+param.beta)*(param.nu-1)*(Apos'*lambda).*costpos.^((param.beta+1)/param.nu-1)./kappa_ex, [1 param.N]).* repmat(param.m',[graph.ndeg 1]).* Qin_direct.^(param.nu-2);
    matneg=repmat( (1+param.beta)*(param.nu-1)*(Aneg'*lambda).*costneg.^((param.beta+1)/param.nu-1)./kappa_ex, [1 param.N]).* repmat(param.m',[graph.ndeg 1]).* Qin_indirect.^(param.nu-2);

    id = 1:graph.ndeg*param.N;

    xpos = graph.J+graph.J*param.N+1;
    ypos = graph.J+graph.J*param.N+1;
    offset_pos = xpos-1+sz*(ypos-1);
    
    xneg = graph.J+graph.J*param.N+graph.ndeg*param.N+1;
    yneg = graph.J+graph.J*param.N+graph.ndeg*param.N+1;
    offset_neg = xneg-1+sz*(yneg-1);
    
    H( offset_pos + id + sz*(id-1) ) = matpos(:);
    H( offset_neg + id + sz*(id-1) ) = matneg(:);
end

% -----------------
% Nondiagonal parts

for n=1:param.N % row
    for m=1:param.N % col
        
        % -----------------
        % Respective to Cjn
        
        HDnondiag=-lambda/param.sigma.*Dj.^(-(param.sigma-2)/param.sigma).*...
            Djn(:,n).^((-1)/param.sigma).*Djn(:,m).^((-1)/param.sigma);
        
        x = graph.J + graph.J*(n-1)+1;
        y = graph.J + graph.J*(m-1)+1;
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
                
        xpos = graph.J + graph.J*param.N + graph.ndeg*(n-1)+1;
        ypos = graph.J + graph.J*param.N + graph.ndeg*(m-1)+1;        
        offset_pos = xpos-1 + sz*(ypos-1);
        
        xneg = graph.J + graph.J*param.N + graph.ndeg*param.N + graph.ndeg*(n-1)+1;
        yneg = graph.J + graph.J*param.N + graph.ndeg*param.N + graph.ndeg*(m-1)+1;        
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
