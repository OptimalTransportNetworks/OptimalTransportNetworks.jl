%{
 =================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, 2017-19
 =================================== version 1.0.4

[results,flag,x]=solve_allocation(...): 
this function is a wrapper that calls the relevant functions to solve for
the economic allocation given a certain matrix of infrastructure
investments in all cases (convex/nonconvex,mobile/immobile labor,cross-good
congestion or none, etc.)

Arguments:
- param: structure that contains the model's parameters
- graph: structure that contains the underlying graph (created by
create_graph function)
- I: JxJ symmetric matrix of infrastructure investments
- verbose: (optional) {true | false} tells IPOPT to display results or not
- x0: (optional) initial seed for the solver
- funcs: (optional) funcs structure returned by ADiGator to avoid calling
it avoid. Can be obtained with the function call_adigator().

Output:
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
function [results,flag,x]=solve_allocation(param,graph,I,verbose,x0,funcs)

% ==================
% CHECK PARAMETERS
% ==================

auxdata=create_auxdata(param,graph,I);

if nargin<6 && param.adigator % if funcs is not specified, call ADiGator
    funcs=call_adigator(param,graph,I,false);    
end

if nargin<5
    x0=[]; % set x0=[] for default init point 
end

if nargin<4
    verbose=false;
end

% ================
% SOLVE ALLOCATION
% ================

if param.adigator && param.mobility~=0.5 % IF USING ADIGATOR
    if param.mobility && param.cong && ~param.custom % implement primal with mobility and congestion
        [results,flag,x] = solve_allocation_mobility_cgc_ADiGator(x0,auxdata,funcs,verbose);
    elseif ~param.mobility && param.cong && ~param.custom % implement primal with congestion
        [results,flag,x] = solve_allocation_cgc_ADiGator(x0,auxdata,funcs,verbose);
    elseif param.mobility && ~param.cong && ~param.custom % implement primal with mobility
        [results,flag,x] = solve_allocation_mobility_ADiGator(x0,auxdata,funcs,verbose);
    elseif (~param.mobility && ~param.cong && ~param.custom) && (param.beta<=1 && param.a <1 && param.duality)% implement dual
        [results,flag,x] = solve_allocation_by_duality_ADiGator(x0,auxdata,funcs,verbose);
    elseif (~param.mobility && ~param.cong && ~param.custom) && (param.beta>1 || param.a ==1) % implement primal
        [results,flag,x] = solve_allocation_ADiGator(x0,auxdata,funcs,verbose);
    elseif param.custom % run custom optimization
        [results,flag,x] = solve_allocation_custom_ADiGator(x0,auxdata,funcs,verbose);
    end
else % IF NOT USING ADIGATOR
    if ~param.cong
        if param.mobility==0
            if param.beta<=1 && param.duality % dual is only twice differentiable if beta<=1
                [results,flag,x] = solve_allocation_by_duality(x0,auxdata,verbose);
            else % otherwise solve the primal
                [results,flag,x] = solve_allocation_primal(x0,auxdata,verbose);
            end
        elseif param.mobility==1 % always solve the primal with labor mobility
            [results,flag,x] = solve_allocation_mobility(x0,auxdata,verbose);
        elseif param.mobility==0.5
            [results,flag,x] = solve_allocation_partial_mobility(x0,auxdata,verbose);
        end
    else
        if param.mobility==0
            [results,flag,x] = solve_allocation_cgc(x0,auxdata,verbose);
        elseif param.mobility==1
            [results,flag,x] = solve_allocation_mobility_cgc(x0,auxdata,verbose);
        elseif param.mobility==0.5
            [results,flag,x] = solve_allocation_partial_mobility_cgc(x0,auxdata,verbose);
        end
    end
end
