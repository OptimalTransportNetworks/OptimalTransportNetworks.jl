%{
 ==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

optimal_network.m: solve for the optimal network by solving the inner problem (dual if no mobility and
no cross good congestion, primal otherwise) and the outer problem by iterating over the FOCs

Arguments:
- param: structure that contains the model's parameters
- graph: structure that contains the underlying graph (created by
create_graph function)
- I0: (optional) provides the initial guess for the iterations (matrix JxJ)
- Il: (optional) exogenous lower bound on infrastructure levels (matrix JxJ)
- Iu: (optional) exogenous upper bound on infrastructure levels (matrix JxJ)
- verbose: (optional) tell IPOPT to display results
- x0: (optional) provide initial condition for IPOPT

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}


function results = optimal_network(param,graph,I0,Il,Iu,verbose,x0)

J=graph.J;
save_before_it_crashes=false;
TOL_I_BOUNDS=1e-7;
error_status=false;

% ---------------------------------------------
% CHECK PARAMETERS FOR ERRORS OR MISSING VALUES

if param.a>1
    error('%s.m: Model with increasing returns to scale not convex.\n',mfilename);
end

if param.nu<1 && param.cong
    error('%s.m: nu has to be larger or equal to one for the problem to be guaranteed convex.\n',mfilename);
end

debug_file_str = 'debug.mat';


if nargin<3 || isempty(I0) % we impose a constraint kl<=kappa for some exercises
    % Init guesses to uniform network unless provided as argument
    I0 = zeros(J,J); % start with uniform allocation
    
    for i=1:J
        for j=1:length(graph.nodes{i}.neighbors)
            I0(i,graph.nodes{i}.neighbors(j))=1;
        end
    end
    I0=param.K*I0/sum(reshape(graph.delta_i.*I0,[graph.J^2 1])); % make sure that we satisfy the capacity constraint
end


if nargin<4 || isempty(Il) % we impose a constraint kl<=kappa for some exercises
    Il=zeros(graph.J,graph.J);
end

if nargin<5 || isempty(Iu)% we impose a constraint kappa<=ku for some exercises
    Iu=inf*ones(graph.J,graph.J);
end

if nargin<6 % if verbose is not specified
    verbose=false; % do not display IPOPT information for every iteration
end

if nargin<7 % if x0 is not specified
    x0=[]; % the solver will use default initial point
end

if param.mobility || param.beta>1 || param.cong % use with the primal version only
    Il=max(1e-6*graph.adjacency,Il); % the primal approach requires a non-zero lower bound on kl, otherwise derivatives explode
end

% --------------
% INITIALIZATION

% CUSTOMIZATION 1: provide here the initial point of optimization for custom case
if param.custom
    x0=[1e-6*ones(graph.J*param.N,1);zeros(graph.ndeg*param.N,1);sum(param.Lj)/(graph.J*param.N)*ones(graph.J*param.N,1)];
    % Based on the primal case with immobile and no cross-good congestion, to
    % be customized
end
% END OF CUSTOMIZATION 1


% -------------
% Call ADiGator

funcs=[];
if param.adigator    
    funcs = call_adigator(param,graph,I0,param.verbose);
end

% =======================================
% SET FUNCTION HANDLE TO SOLVE ALLOCATION

if param.adigator && param.mobility~=0.5 % IF USING ADIGATOR
    if param.mobility==1 && param.cong && ~param.custom % implement primal with mobility and congestion
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_mobility_cgc_ADiGator(x0,auxdata,funcs,verbose);
    elseif param.mobility==0 && param.cong && ~param.custom % implement primal with congestion
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_cgc_ADiGator(x0,auxdata,funcs,verbose);
    elseif param.mobility==1 && ~param.cong && ~param.custom % implement primal with mobility
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_mobility_ADiGator(x0,auxdata,funcs,verbose);
    elseif (param.mobility==0 && ~param.cong && ~param.custom) && (param.beta<=1 && param.a <1 && param.duality)% implement dual
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_by_duality_ADiGator(x0,auxdata,funcs,verbose);
    elseif (param.mobility==0 && ~param.cong && ~param.custom) && (param.beta>1 || param.a ==1) % implement primal
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_ADiGator(x0,auxdata,funcs,verbose);
    elseif param.custom % run custom optimization
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_custom_ADiGator(x0,auxdata,funcs,verbose);
    end
else % IF NOT USING ADIGATOR
    if ~param.cong
        if param.mobility==0
            if param.beta<=1 && param.duality% dual is only twice differentiable if beta<=1
                solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_by_duality(x0,auxdata,verbose);
            else % otherwise solve the primal
                solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_primal(x0,auxdata,verbose);
            end
        elseif param.mobility==1 % always solve the primal with labor mobility
            solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_mobility(x0,auxdata,verbose);
        elseif param.mobility==0.5 
            solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_partial_mobility(x0,auxdata,verbose);
        end
    else
        if param.mobility==0
            solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_cgc(x0,auxdata,verbose);
        elseif param.mobility==1
            solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_mobility_cgc(x0,auxdata,verbose);
        elseif param.mobility==0.5
            solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_partial_mobility_cgc(x0,auxdata,verbose);
        end
    end
end



% ==========
% RESOLUTION
% ==========

% -------
% ITERATE

has_converged=false;
counter=0;
weight_old=0.5;
I1=zeros(graph.J,graph.J);

if param.verbose
    fprintf('\n---------------------------\n');
    fprintf('STARTING MAIN LOOP...\n\n');
end

while ( ~has_converged && counter<param.MAX_ITER_KAPPA ) || counter<=20
    
    skip_update=false;
    
    % Create auxdata structure that contains all info for IPOPT/ADiGator
    auxdata=create_auxdata(param,graph,I0);
    
    if save_before_it_crashes==true
        save(debug_file_str,'param','graph','kappa','x0','I0','I1','counter');
    end
    
    % Solve allocation
    t0=clock();
    [results,flag,x1] = solve_allocation_handle(x0,auxdata,funcs,verbose);    
    t1=clock();
    
    if ~any( flag.status == [0,1] )
        if ~isempty(x0) % if error happens with warm start, then try again starting cold
            x0=[];
            skip_update=true;
        else
            error('%s.m: IPOPT returned with error code %d.',mfilename,flag.status);
        end
    else
        x0=x1;
    end
    
    % Compute new I1
    
    if ~param.cong % no cross-good congestion
        Pjkn=repmat(permute(results.Pjn,[1 3 2]),[1 graph.J 1]);
        PQ=Pjkn.*results.Qjkn.^(1+param.beta);
        PQ=PQ+permute(PQ,[2 1 3]);
        PQ=sum(PQ,3);
        I1=(graph.delta_tau./graph.delta_i.*PQ).^(1/(1+param.gamma));
        I1( graph.adjacency==false )=0;
        I1(PQ==0)=0;
        I1(graph.delta_i==0)=0;
    else % cross-good congestion
        PCj=repmat(results.PCj,[1 graph.J]);
        matm=shiftdim(repmat(param.m,[1 graph.J graph.J]),1);
        cost=sum(matm.*results.Qjkn.^param.nu,3).^((param.beta+1)/param.nu);
        PQ=PCj.*cost;
        PQ=PQ+PQ';
        I1=(graph.delta_tau./graph.delta_i.*PQ).^(1/(param.gamma+1));
        I1( graph.adjacency==false )=0;
        I1(PQ==0)=0;
        I1(graph.delta_i==0)=0;
    end
    
    % CUSTOMIZATION 2: updating network
    if param.custom
        % enter here how to update the infrastructure network I1 (if needed) in the custom case
    end
    % END OF CUSTOMIZATION 2
  
    
    % Take care of scaling and bounds
    I1=param.K*I1/sum(reshape(graph.delta_i.*I1,[graph.J^2 1])); % rescale so that sum(delta*kappa^1/gamma)=K
    
    distance_lb=max(max(Il(:)-I1(:)),0);
    distance_ub=max(max(I1(:)-Iu(:)),0);
    counter_rescale=0;
    while distance_lb+distance_ub>TOL_I_BOUNDS && counter_rescale<100
        I1=max(min(I1,Iu),Il); % impose the upper and lower bounds
        I1=param.K*I1/sum(reshape(graph.delta_i.*I1,[graph.J^2 1])); % rescale again
        distance_lb=max(max(Il(:)-I1(:)),0);
        distance_ub=max(max(I1(:)-Iu(:)),0);
        counter_rescale=counter_rescale+1;
    end
    
    if counter_rescale==100 && distance_lb+distance_ub>param.tol_kappa && param.verbose
        fprintf('%s.m: Warning! Could not impose bounds on network properly.\n',mfilename);
    end
    
    % Check convergence
    
    distance=max(abs(I1(:)-I0(:)))/(param.K/mean(graph.delta_i(graph.adjacency==1)));
    has_converged=distance<param.tol_kappa;
    counter=counter+1;
    
    if param.verbose
        fprintf('Iteration No.%d distance=%d duration=%2.1f secs. Welfare=%d\n',counter,distance,etime(t1,t0),results.welfare);
    end
    
    if (~has_converged || counter<=20) && skip_update==false
        I0 = weight_old*I0+(1-weight_old)*I1;
    end
    
end

% ==============
% POST TREATMENT
% ==============

% -----------------------
% CHECK FINAL CONVERGENCE

if counter<=param.MAX_ITER_KAPPA && ~has_converged && param.verbose
    fprintf('%s.m: reached MAX iterations with convergence at %d.\n',mfilename,distance);
    error_status=true;
end

% --------------
% RETURN RESULTS

results.Ijk = I0;

if param.verbose && error_status==false
    fprintf('\nCOMPUTATION RESULTED WITH SUCCESS.\n');
    fprintf('----------------------------------\n');
end

% ---------
% ANNEALING
if param.gamma>param.beta && param.annealing
    results=annealing(param,graph,I0,'Funcs',funcs);
end

end
