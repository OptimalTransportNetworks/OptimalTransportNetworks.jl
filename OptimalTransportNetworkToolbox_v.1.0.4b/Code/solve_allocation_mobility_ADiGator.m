%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,x]=solve_allocation_mobility_ADiGator(...): 
this function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case with labor mobility with
a primal approach (quasiconcave). It uses the autodifferentiation package
Adigator to generate the functional inputs for IPOPT.

Arguments:
- x0: initial seed for the solver
- funcs: contains the Adigator generated functions (objective, gradient,
hessian, constraints, jacobian)
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

function [results,flag,x]=solve_allocation_mobility_ADiGator(x0,auxdata,funcs,verbose)

% ==================
% RECOVER PARAMETERS
% ==================

graph=auxdata.graph;
param=auxdata.param;

if any(sum(param.Zjn>0,2)>1) && param.a == 1
    error('%s.m: Version of the code with more than 1 good per location and a=1 not supported yet.\n',mfilename);
end

if nargin<4
    verbose=true;
end

if isempty(x0)
    x0=[0;1e-6*ones(graph.J*param.N,1);zeros(graph.ndeg*param.N,1);1/graph.J*ones(graph.J,1);1/(graph.J*param.N)*ones(graph.J*param.N,1)]; % starting point
    % x(1) is u, x(1+1:J*N+1) is Cjn, x(J*N+1+1: 1+J*N + ndeg*N) is
    % Qin, x(J*N+1 + ndeg*N+1: 1+J*N + ndeg*N+J) is Lj, the last JN terms are Ljn
end


% =================
% PARAMETRIZE IPOPT
% =================

% Update functions
funcs.objective = @(x) objective_mobility(x,auxdata);
funcs.gradient = @(x) objective_mobility_Grd(x,auxdata);
funcs.constraints= @(x) constraints_mobility(x,auxdata);
funcs.jacobian= @(x) constraints_mobility_Jac(x,auxdata);
funcs.hessian= @(x,sigma,lambda)objective_mobility_Hes(x,auxdata,sigma,lambda);

% Options
% Bounds on optimization variables
options.lb = [-inf;1e-8*ones*ones(graph.J*param.N,1);-inf*ones(graph.ndeg*param.N,1);1e-8*ones(graph.J,1);1e-8*ones*ones(graph.J*param.N,1)];
options.ub = [inf;inf*ones(graph.J*param.N,1);inf*ones(graph.ndeg*param.N,1);ones(graph.J,1);inf*ones(graph.J*param.N,1)];
% Bounds on constraints
options.cl = [-inf*ones(graph.J,1);-inf*ones(graph.J*param.N,1);-1e-8;-1e-8*ones(graph.J,1)]; 
options.cu = [-1e-8*ones(graph.J,1);-1e-8*ones(graph.J*param.N,1);1e-8;1e-8*ones(graph.J,1)]; 

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

results.Pjn=reshape(info.lambda(graph.J+1:graph.J+graph.J*param.N),[graph.J param.N]); % Price vector
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma)); % Price of tradeable 

end % end of function


function results = recover_allocation(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

% Aggreagte welfare
results.welfare=x(1);

% Domestic absorption
results.Cjn=reshape(x(1+1:1+graph.J*param.N),[graph.J param.N]);

% Total availability of tradeable good
results.Cj=sum(results.Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% Population
results.Lj=x(1+graph.J*param.N+graph.ndeg*param.N+1:1+graph.J*param.N+graph.ndeg*param.N+graph.J);

% Working population
results.Ljn=reshape(x(1+graph.J*param.N+graph.ndeg*param.N+graph.J+1:end),[graph.J param.N]);

% Production
results.Yjn=param.Zjn.*results.Ljn.^param.a;

% Consumption per capital
results.cj=results.Cj./results.Lj;
results.cj(results.Lj==0)=0;

% Non-tradeable per capita
results.hj=param.Hj./results.Lj;
results.hj(results.Lj==0)=0;

% Vector of welfare per location
results.uj=param.u(results.cj,results.hj);

% Trade flows
results.Qin=reshape(x(1+graph.J*param.N+1:1+graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);


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



