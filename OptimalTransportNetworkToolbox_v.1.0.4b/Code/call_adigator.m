%{
 ==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

call_adigator.m: call package ADiGator to produce the gradient, jacobian and hessian
to be used by IPOPT

Arguments:
- auxdata: structure created by create_auxdata() that contains all the model's parameters
- verbose: switch whether or not the function should display output

Output:
- funcs: funcs structure used as input in IPOPT (returns [] if error)

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}


function funcs = call_adigator(param,graph,I,verbose)

funcs=[];

% ----------------
% Check parameters

if nargin<4
    verbose=false;
end

if ~param.adigator
    fprintf('%s.m: Please enable ADiGator with init_parameters() before.\n',mfilename);
    return;
end

% -------------
% Call ADiGator

auxdata=create_auxdata(param,graph,I);

if param.mobility && param.cong && ~param.custom
    setup.numvar=1+graph.J*param.N+2*graph.ndeg*param.N+graph.J+graph.J+graph.J*param.N;
    setup.objective='objective_mobility_cgc';
    setup.auxdata=auxdata;
    setup.order = 2;
    setup.constraint='constraints_mobility_cgc';
elseif ~param.mobility && param.cong && ~param.custom
    setup.numvar=graph.J*param.N+2*graph.ndeg*param.N+graph.J+graph.J*param.N;
    setup.objective='objective_cgc';
    setup.auxdata=auxdata;
    setup.order = 2;
    setup.constraint='constraints_cgc';
elseif param.mobility && ~param.cong && ~param.custom
    setup.numvar=1+graph.J*param.N+graph.ndeg*param.N+graph.J+graph.J*param.N;
    setup.objective='objective_mobility';
    setup.auxdata=auxdata;
    setup.order = 2;
    setup.constraint='constraints_mobility';
elseif (~param.mobility && ~param.cong && ~param.custom) && (param.beta<=1 && param.a<1)
    setup.numvar=graph.J*param.N;
    setup.objective='objective_duality';
    setup.auxdata=auxdata;
    setup.order = 2;
elseif (~param.mobility && ~param.cong && ~param.custom) && (param.beta>1 || param.a == 1)
    setup.numvar=graph.J*param.N+graph.ndeg*param.N+graph.J*param.N;
    setup.objective='objective';
    setup.auxdata=auxdata;
    setup.order = 2;
    setup.constraint='constraints';
elseif param.custom
    setup.numvar=size(x0,1);
    setup.objective='objective_custom';
    setup.auxdata=auxdata;
    setup.order = 2;
    setup.constraint='constraints_custom';
end

if verbose
    fprintf('\n-------------------\n');
    fprintf('CALLING ADIGATOR...\n\n');
end

setup.options = adigatorOptions('ECHO',verbose,'OVERWRITE',1);

funcs = adigatorGenFiles4Ipopt(setup);
end


