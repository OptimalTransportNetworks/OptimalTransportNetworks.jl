%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,x]=solve_allocation_cgc_ADiGator(...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case without labor mobility with
a primal approach in the cross-good congestion case. 
It uses the autodifferentiation package Adigator to generate the functional inputs for IPOPT.

Arguments:
- x0: initial seed for the solver
- verbose: {true | false} tells IPOPT to display results or not
- funcs: contains the Adigator generated functions (objective, gradient,
hessian, constraints, jacobian)
- auxdata: contains the model parameters (param, graph, kappa, kappa_ex, A)

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

function [results,flag,x]=solve_allocation_cgc_ADiGator(x0,auxdata,funcs,verbose)

% ==================
% RECOVER PARAMETERS
% ==================

graph=auxdata.graph;
param=auxdata.param;

if nargin<4
    verbose=true;
end

if isempty(x0)
    x0=[1e-6*ones(graph.J*param.N,1);zeros(2*graph.ndeg*param.N,1);1e-6*ones(graph.J,1);1e-6*ones(graph.J*param.N,1)]; % starting point
    % First JN variables are Cjn, the next Ndeg*N are Qin in the edge direction,
    % the next ndeg*N are Qin in the edge opposite direction, the next J are cj,
    % the last JN are Ljn
end

% ===============
% PRECALCULATIONS
% ===============

% extract parameters
graph=auxdata.graph;
param=auxdata.param;

% =================
% PARAMETRIZE IPOPT
% =================

% Init functions
funcs.objective = @(x) objective_cgc(x,auxdata);
funcs.gradient = @(x) objective_cgc_Grd(x,auxdata);
funcs.constraints = @(x) constraints_cgc(x,auxdata);
funcs.jacobian = @(x) constraints_cgc_Jac(x,auxdata);
funcs.hessian = @(x,sigma,lambda) objective_cgc_Hes(x,auxdata,sigma,lambda);

% Options
% Bounds on optimization variables
options.lb = [1e-8*ones(graph.J*param.N,1);1e-8*ones(2*graph.ndeg*param.N,1);1e-8*ones(graph.J,1);1e-8*ones(graph.J*param.N,1)];
options.ub = [inf*ones(graph.J*param.N,1);inf*ones(2*graph.ndeg*param.N+graph.J,1);inf*ones(graph.J*param.N,1)];
% Bounds on constraints
options.cl = -inf*ones(graph.J*param.N+2*graph.J,1); % lower bound on constraint function
options.cu = -1e-8*ones(graph.J*param.N+2*graph.J,1); % upper bound on constraint function

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

results=recover_allocation(param,graph,x);
results.welfare=-funcs.objective(x);    % Total economy welfare
results.Pjn=reshape(info.lambda(1:graph.J*param.N),[graph.J param.N]); % Price vector
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma)); % Price of tradeable

end % end of function

function results = recover_allocation(param,graph,x)

% Domestic absorption per good per location
results.Djn=reshape(x(1:graph.J*param.N),[graph.J param.N]); % Consumption per good pre-transport cost

% Total availability of tradeable good
results.Dj=sum(results.Djn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1)); % Aggregate consumption pre-transport cost

% Population
results.Lj=param.Lj;

% Consumption per capita
results.cj=x(graph.J*param.N+2*graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N+graph.J);

% Vector of welfare per location
results.uj=((results.cj/param.alpha).^param.alpha.*(param.hj/(1-param.alpha)).^(1-param.alpha)).^(1-param.rho)/(1-param.rho);  

% Total consumption of tradeable good
results.Cj=results.cj.*param.Lj;

% Working population
results.Ljn=reshape(x(graph.J*param.N+2*graph.ndeg*param.N+graph.J+1:end),[graph.J param.N]);

% Non-tradeable per capita
results.hj=param.hj;

% Production
results.Yjn=param.Zjn.*(results.Ljn.^param.a);

% Trade flows
Qin_direct=x(graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N);
Qin_indirect=x(graph.J*param.N+graph.ndeg*param.N+1:graph.J*param.N+2*graph.ndeg*param.N);
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
