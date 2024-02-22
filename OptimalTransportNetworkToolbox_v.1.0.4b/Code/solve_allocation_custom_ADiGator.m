%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results,flag,x]=solve_allocation_custom(...): 
This function is designed to allow the user to customize the problem
specification. By default, it is encoded as the immobile labor, primal case
without cross good congestion.
This function solves the full allocation of Qjkn and Cj given a matrix of
kappa (=I^gamma/delta_tau). It solves the case without labor mobility with
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

function [results,flag,x]=solve_allocation_custom_ADiGator(x0,auxdata,funcs,verbose)

% ==================
% RECOVER PARAMETERS
% ==================

graph=auxdata.graph;
param=auxdata.param;

if nargin<4
    verbose=true;
end

if isempty(x0)
    x0=[1e-6*ones(graph.J*param.N,1);zeros(graph.ndeg*param.N,1);sum(param.Lj)/(graph.J*param.N)*ones(graph.J*param.N,1)];
    % JxN first terms are Cjn, next ndegxN terms are Qin, next JxN terms are Ljn
end



% =================
% PARAMETRIZE IPOPT
% =================

% Init functions
funcs.objective = @(x) objective_custom(x,auxdata);
funcs.gradient = @(x) objective_custom_Grd(x,auxdata);
funcs.constraints = @(x) constraints_custom(x,auxdata);
funcs.jacobian = @(x) constraints_custom_Jac(x,auxdata);
funcs.hessian = @(x,sigma,lambda) objective_custom_Hes(x,auxdata,sigma,lambda);

% Options
% Bounds on optimization variables
% variables.
options.lb = [1e-8*ones(graph.J*param.N,1);-inf*ones(graph.ndeg*param.N,1);1e-8*ones(graph.J*param.N,1)];
options.ub = [inf*ones(graph.J*param.N,1);inf*ones(graph.ndeg*param.N,1);inf*ones(graph.J*param.N,1)];
% Bounds on constraint functions
options.cl = [-inf*ones(graph.J*param.N,1);-inf*ones(graph.J,1)]; 
options.cu = [0*ones(graph.J*param.N,1);zeros(graph.J,1)]; 

% options.ipopt.hessian_approximation = 'limited-memory';
% options.ipopt.max_iter = 2000;

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

results.Pjn=reshape(info.lambda(1:graph.J*param.N),[graph.J param.N]); % Price vector
results.PCj=sum(results.Pjn.^(1-param.sigma),2).^(1/(1-param.sigma)); % Price of tradeable
results.welfare=-funcs.objective(x);   % Total economy welfare

end % end of function

function results = recover_allocation(x,auxdata)
param=auxdata.param;
graph=auxdata.graph;

% Recover parameters
results.Lj=param.Lj;

% Recover populations
results.Ljn=reshape(x(graph.J*param.N+graph.ndeg*param.N+1:end),[graph.J param.N]);

% Production
results.Yjn=param.Zjn.*(results.Ljn.^param.a);

% Domestic absorption
results.Cjn=reshape(x(1:graph.J*param.N),[graph.J param.N]);

% Total availability of tradeable good
results.Cj=sum(results.Cjn.^((param.sigma-1)/param.sigma),2).^(param.sigma/(param.sigma-1));

% Consumption per capita
results.cj=results.Cj./results.Lj;
results.cj(results.Lj==0)=0;

% Non-tradeable per capita
results.hj=param.hj;

% Vector of welfare per location
results.uj=param.u(results.cj,results.hj);

% Recover flows Qin in dimension [Ndeg,N]
results.Qin=reshape(x(graph.J*param.N+1:graph.J*param.N+graph.ndeg*param.N),[graph.ndeg param.N]);

% recover the flows Qjkn in dimension [J,J,N]
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




