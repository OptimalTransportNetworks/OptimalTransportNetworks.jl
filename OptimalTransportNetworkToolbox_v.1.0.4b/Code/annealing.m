%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[results]=annealing ( param, graph, I0, varargin ):
runs the simulated annealing method starting from network I0

Arguments:
- param: structure that contains the model's parameters
- graph: structure that contains the underlying graph (created by
create_map, create_rectangle or create_triangle functions)
- I0: (optional) provides the initial guess for the iterations
- varargin: various optional arguments as follows

'PerturbationMethod'={'shake','random','rebranching','random rebranching','hybrid alder'}: method to be used to perturbate the network
( random is purely random, works horribly; shake applies a gaussian blur
along a random direction, works alright; ewbranching (default) is the algorithm desxribed
in Appendix A.4 in the paper, works nicely )

'PreserveCentralSymmetry={'on','off'}; % only applies to shake method
'PreserveVerticalSymmetry={'on','off'}; % only applies to shake method
'PreserveHorizontalSymmetry={'on','off'}; % only applies to shake method
'SmoothingRadius= default 0.25; % parameters of the gaussian blur
'MuPerturbation'=default log(.3); % parameters of the gaussian blur
'SigmaPerturbation'=default 0.05; % parameters of the gaussian blur
'Display'={'on','off'} % display the graph as we go
'TStart'= default 100 % initial temperature
'TEnd'= default 1; % final temperature
'TStep'=0.9 % speed of cooling
'NbDeepening'=4 % number of FOC iterations between candidate draws
'NbRandomPerturbations'=1 % number of links to be randomly affected
('random' and 'random rebranching' only)
'Funcs': funcs structure computed by ADiGator in order to skip rederivation
'Iu': JxJ matrix of upper bounds on network infrastructure Ijk
'Il': JxJ matrix of lower bounds on network infrastructure Ijk

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal 
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}


function best_results=annealing(param,graph,I0,varargin)
% Retrieve economy's parameters
J=graph.J;
gamma=param.gamma;
verbose=false;
TOL_I_BOUNDS=1e-7;

% Retrieve optional parameters
options = retrieve_options(param,graph,varargin);
Iu=options.Iu;
Il=options.Il;

if param.mobility || param.beta>1% use with the primal version only
    Il=max(1e-6*graph.adjacency,Il); % the primal approach in the mobility case requires a non-zero lower bound on kl, otherwise derivatives explode
end

% Parameters for simulated annealing algorithm
T=options.t_start;
T_min=options.t_end;%1e-3
T_step=options.t_step;
nb_network_deepening=options.nb_deepening; % number of iterations for local network deepening

% -------------------
% PERTURBATION METHOD

switch options.perturbation_method
    case 'random'
        perturbate = @(param,graph,kappa0,res) random_perturbation(param,graph,I0,res,options);
    case 'shake'
        perturbate = @(param,graph,I0,res) shake_network(param,graph,I0,res,options);
    case 'rebranching'
        perturbate = @(param,graph,I0,res) rebranch_network(param,graph,I0,res,options);
    case 'random rebranching'
        perturbate = @(param,graph,I0,res) random_rebranch_network(param,graph,I0,res,options);
    case 'hybrid alder'
        perturbate = @(param,graph,I0,res) hybrid(param,graph,I0,res,options);    
    otherwise
        error('%s.m: unknown perturbation method %s.\n',mfilename,options.perturbation_method);
end

% ----------
% INIT STUFF

% initial point in the IPOPT optimization
x0=[]; % automatic initialization except for custom case
% CUSTOMIZATION 1: provide here the initial point of optimization for custom case
if param.custom % Enter here the initial point of optimization for custom
    x0=[1e-6*ones(graph.J*param.N,1);zeros(graph.ndeg*param.N,1);sum(param.Lj)/(graph.J*param.N)*ones(graph.J*param.N,1)];
    % Based on the primal case with immobile and no cross-good congestion, to
    % be customized
end
% END OF CUSTOMIZATION 1

% =======================================
% SET FUNCTION HANDLE TO SOLVE ALLOCATION

if param.adigator && param.mobility~=0.5 % IF USING ADIGATOR
    if param.mobility && param.cong && ~param.custom % implement primal with mobility and congestion
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_mobility_cgc_ADiGator(x0,auxdata,funcs,verbose);
    elseif ~param.mobility && param.cong && ~param.custom % implement primal with congestion
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_cgc_ADiGator(x0,auxdata,funcs,verbose);
    elseif param.mobility && ~param.cong && ~param.custom % implement primal with mobility
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_mobility_ADiGator(x0,auxdata,funcs,verbose);
    elseif (~param.mobility && ~param.cong && ~param.custom) && (param.beta<=1 && param.a <1 && param.duality)% implement dual
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_by_duality_ADiGator(x0,auxdata,funcs,verbose);
    elseif (~param.mobility && ~param.cong && ~param.custom) && (param.beta>1 || param.a ==1) % implement primal
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_ADiGator(x0,auxdata,funcs,verbose);
    elseif param.custom % run custom optimization
        solve_allocation_handle = @(x0,auxdata,funcs,verbose) solve_allocation_custom_ADiGator(x0,auxdata,funcs,verbose);
    end
else % IF NOT USING ADIGATOR
    if ~param.cong
        if param.mobility==0
            if param.beta<=1 && param.duality % dual is only twice differentiable if beta<=1
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

% ===================================================
% GENERATE AUTODIFFERENTIATED FUNCTIONS WITH ADIGATOR
% ===================================================

funcs = options.funcs;
if param.adigator && isempty(funcs)
    funcs = call_adigator(param,graph,I0,verbose);
end

% =========
% ANNEALING
% =========

if param.verbose
    fprintf('\n-------------------------------\n');
    fprintf('STARTING SIMULATED ANNEALING...\n\n');
end

best_score=-inf;
best_I=I0;

old_score=best_score;
weight_old=0.5;
counter=0;
I1=I0;
% rng(0); % set the seed of random number generator for replicability
acceptance_str = {'rejected','accepted'};

while T>T_min
    accepted=false;
    
    if param.verbose
        fprintf('Perturbating network (Temperature=%3.1f)...\n',T);
    end
    
    if counter>0
        I1=perturbate(param,graph,I0,results);
    end
    
    if options.display
        plot_graph( param,graph,I1);
    end
    
    if param.verbose
        fprintf('Iterating on FOCs...\n');
    end
    
    k=0;
    x=x0; % use default initial condition for allocation
    while k<=nb_network_deepening-1
        % Create auxdata structure for IPOPT/ADiGator
        auxdata=create_auxdata(param,graph,I1);        
        
        % Solve allocation
        [results,flag,x]=solve_allocation_handle(x,auxdata,funcs,verbose);
        
        score=results.welfare;
        
        if (~any( flag.status == [0,1] ) || isnan(score)) && param.verbose% optimization failed
            fprintf('%s.m: optimization failed! k=%d, return flag=%d\n',mfilename,k,flag.status);
            k=nb_network_deepening-1;
            score=-inf;
        end
        
        if score>best_score
            best_results=results;            
            best_I=I1;
            best_score=score;
        end
        
        % Deepen network
        if k<nb_network_deepening-1
            if ~param.cong % no cross-good congestion
                Pjkn=repmat(permute(results.Pjn,[1 3 2]),[1 graph.J 1]);
                PQ=Pjkn.*results.Qjkn.^(1+param.beta);
                I1=(graph.delta_tau./graph.delta_i.*sum(PQ+permute(PQ,[2 1 3]),3)).^(1/(1+param.gamma));
                I1( graph.adjacency==false )=0;
            else % cross-good congestion
                PCj=repmat(results.PCj,[1 graph.J]);
                matm=shiftdim(repmat(param.m,[1 graph.J graph.J]),1);
                cost=sum(matm.*results.Qjkn.^param.nu,3).^((param.beta+1)/param.nu);
                PQ=PCj.*cost;
                I1=(graph.delta_tau./graph.delta_i.*(PQ+PQ')).^(1/(param.gamma+1));
                I1( graph.adjacency==false )=0;
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
        end
        k=k+1;
    end
    
    % Probabilistically accept perturbation    
    acceptance_prob=exp(1e4*(score-old_score)/(abs(old_score)*T)); % set such that a 1% worse allocation is accepted with proba exp(-.1)=0.36 at beginning
    if rand()<=acceptance_prob || counter==0
        I0=I1;
        old_score=score;
        accepted=true;
    end
    
    % --------------
    % DISPLAY STATUS
    
    if options.display
        plot_graph( param,graph,I1,'Sizes',results.Lj);
    end
    
    if param.verbose
        fprintf('Iteration No.%d, score=%d, %s (best=%d)\n',counter,score,acceptance_str{1+double(accepted)},best_score);
    end
    
    counter=counter+1;
    T=T*T_step;
end

% Last deepening before returning found optimum
has_converged=false;
I0=best_I;
while ~has_converged && counter<100
    % Update auxdata
    auxdata=create_auxdata(param,graph,I0);        
        
    % Solve allocation
    [results,flag,x]=solve_allocation_handle(x0,auxdata,funcs,verbose);
    
    score=results.welfare;
    
    if options.display
        plot_graph( param,graph,I0,'Sizes',results.Lj );
    end
    
    if score>best_score
        best_results = results;        
        best_I=I0;
        best_score=score;
    end
    
    % DEEPEN NETWORK
    if ~param.cong % no cross-good congestion
        Pjkn=repmat(permute(results.Pjn,[1 3 2]),[1 graph.J 1]);
        PQ=Pjkn.*results.Qjkn.^(1+param.beta);
        I1=(graph.delta_tau./graph.delta_i.*sum(PQ+permute(PQ,[2 1 3]),3)).^(1/(1+param.gamma));
        I1( graph.adjacency==false )=0;
    else % cross-good congestion
        PCj=repmat(results.PCj,[1 graph.J]);
        matm=shiftdim(repmat(param.m,[1 graph.J graph.J]),1);
        cost=sum(matm.*results.Qjkn.^param.nu,3).^((param.beta+1)/param.nu);
        PQ=PCj.*cost;
        I1=(graph.delta_tau./graph.delta_i.*(PQ+PQ')).^(1/(param.gamma+1));
        I1( graph.adjacency==false )=0;
    end
    
    % CUSTOMIZATION 2': updating network
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
    
    % UPDATE AND DISPLAY RESULTS
    distance=sum(abs(I1(:)-I0(:)))/(J^2);
    I0 = weight_old*I0+(1-weight_old)*I1;
    has_converged=distance<param.tol_kappa;
    counter=counter+1;
    
    if param.verbose
        fprintf('Iteration No.%d - final iterations - distance=%d - welfare=%d\n',counter,distance,score);
    end
    
end

if param.verbose
    fprintf('Welfare = %d\n',score);
end

best_results.Ijk=best_I;

end %% end of function

%%
%{
function I1=random_perturbation(param,graph,I0,res,options)
This function add #NbRandomPerturbations random links to the network and 
applies a Gaussian smoothing to prevent falling too quickly in a local optimum 
(see smooth_network()).
%}
function I1=random_perturbation(param,graph,I0,res,options)
size_perturbation=0.1*param.K/graph.J;

I1=I0;

% Draw random perturbations 
link_list=randperm(graph.J,options.nb_random_perturbations);

for i=link_list    
    j=randi(length(graph.nodes{i}.neighbors));
    I1(i,graph.nodes{i}.neighbors(j))=size_perturbation*exp(randn())/exp(0.5);
end

% Make sure graph satisfies symmetry and capacity constraint
I1=(I1+I1')/2; % make sure kappa is symmetric
I1=param.K*I1/sum(reshape(graph.delta_i.*I1,[graph.J^2 1])); % rescale

% % Smooth network (optional)
% I1=smooth_network(param,graph,I1);
end   %% End of function

%%
%{
function I1=smooth_network(param,graph,I0)
This function produces a smoothed version of the network by applying a
Gaussian smoother (i.e., a Gaussian kernel estimator). The main benefit is
that it makes the network less sparse and the variance of investments gets
reduced.
%}
%%
function I1=smooth_network(param,graph,I0)
J=graph.J;
smoothing_radius=0.3;

I1=zeros(J,J);
feasible_edge=zeros(J,J);
vec_x=zeros(J,1);
vec_y=zeros(J,1);
for i=1:J
    vec_x(i) = graph.x(i);
    vec_y(i) = graph.y(i);
    for j=1:length(graph.nodes{i}.neighbors);
        feasible_edge(i,graph.nodes{i}.neighbors(j))=1;
    end
end

edge_x=0.5*(vec_x(:,ones(J,1))+ones(J,1)*vec_x');
edge_y=0.5*(vec_y(:,ones(J,1))+ones(J,1)*vec_y');

% Proceed to Gaussian kernel smoothing
for i=1:graph.J
    for j=1:length(graph.nodes{i}.neighbors)
        x0=edge_x(i,graph.nodes{i}.neighbors(j));
        y0=edge_y(i,graph.nodes{i}.neighbors(j));
        
        weight=exp(-0.5/smoothing_radius^2*((edge_x-x0).^2+(edge_y-y0).^2));
        weight(~feasible_edge)=0;
        I1(i,graph.nodes{i}.neighbors(j))=sum(sum(I0.*weight))/sum(weight(:));
    end
end

% Make sure graph satisfies symmetry and capacity constraint
I1=(I1+I1')/2; % make sure kappa is symmetric
I1=param.K*I1/sum(reshape(graph.delta_i.*I1,[graph.J^2 1])); % rescale
end %% End of function

%%
%{
function I1=shake_network(param,graph,I0,res,options)
A simple way to perturb the network that simulates shaking the network in some 
random direction and applying a Gaussian smoothing (see smooth_network()).
%}

%%
function I1=shake_network(param,graph,I0,res,options)
J=graph.J;

% ===================
% RETRIEVE PARAMETERS
% ===================

smoothing_radius=options.smoothing_radius;
mu_perturbation=options.mu_perturbation;
sigma_perturbation=options.sigma_perturbation;

% ===============
% PERTURB NETWORK
% ===============

theta=rand()*2*pi;
rho=exp(mu_perturbation+sigma_perturbation*randn())/exp(sigma_perturbation^2);
direction = rho*exp(1i*theta);
direction_x = real(direction);
direction_y = imag(direction);

I1=zeros(J,J);
vec_x=graph.x;
vec_y=graph.y;
feasible_edge=graph.adjacency;

edge_x=0.5*(vec_x(:,ones(J,1))+ones(J,1)*vec_x');
edge_y=0.5*(vec_y(:,ones(J,1))+ones(J,1)*vec_y');

% Proceed to Gaussian kernel smoothing
for i=1:graph.J
    for j=1:length(graph.nodes{i}.neighbors)
        x0=edge_x(i,graph.nodes{i}.neighbors(j));
        y0=edge_y(i,graph.nodes{i}.neighbors(j));
        
        weight=exp(-0.5/smoothing_radius^2*((edge_x-x0-direction_x).^2+(edge_y-y0-direction_y).^2));
        if options.preserve_central_symmetry==true
            weight=weight+exp(-0.5/smoothing_radius^2*((edge_x-x0+direction_x).^2+(edge_y-y0+direction_y).^2));
        end
        if options.preserve_horizontal_symmetry==true
            weight=weight+exp(-0.5/smoothing_radius^2*((edge_x-x0-direction_x).^2+(edge_y-y0+direction_y).^2));
        end
        if options.preserve_vertical_symmetry==true
            weight=weight+exp(-0.5/smoothing_radius^2*((edge_x-x0+direction_x).^2+(edge_y-y0-direction_y).^2));
        end
        weight(~feasible_edge)=0;
        I1(i,graph.nodes{i}.neighbors(j))=sum(sum(I0.*weight))/sum(weight(:));
    end
end

% Make sure graph satisfies symmetry and capacity constraint
I1=(I1+I1')/2; % make sure kappa is symmetric
I1=param.K*I1/sum(reshape(graph.delta_i.*I1,[graph.J^2 1])); % rescale
end   %% End of function

%%
%{
function I1=rebranch_network(param,graph,I0,res,options)
This function implements the rebranching algorithm described in the paper. 
Links are reshuffled everywhere so that each nodes is better connected to its best neighbor
(those with lowest price index for traded goods, i.e., more central places in the trading network).
%}
function I1=rebranch_network(param,graph,I0,res,options)
J=graph.J;

% ===============
% PERTURB NETWORK
% ===============

I1=I0;

% Rebranch each location to its lowest price parent
for i=1:graph.J
    neighbors = graph.nodes{i}.neighbors;
    parents = neighbors ( res.PCj(neighbors) < res.PCj(i) );
    
    if length(parents)>=2
        [~,lowest_price_parent] = sort( res.PCj(parents) );
        lowest_price_parent = parents(lowest_price_parent(1));
        [~,best_connected_parent] = sort( I0(i,parents) , 'descend' );
        best_connected_parent = parents(best_connected_parent(1));
        
        % swap roads
        I1( i,lowest_price_parent ) = I0( i,best_connected_parent);
        I1( i,best_connected_parent ) = I0( i,lowest_price_parent);
        I1( lowest_price_parent,i ) = I0( i,best_connected_parent);
        I1( best_connected_parent,i ) = I0( i,lowest_price_parent);
        
    end
end

% Make sure graph satisfies symmetry and capacity constraint
I1=(I1+I1')/2; % make sure kappa is symmetric
I1=param.K*I1/sum(reshape(graph.delta_i.*I1,[graph.J^2 1])); % rescale
end % End of function

%%
%{
function I1=random_rebranch_network(param,graph,I0,res,options)
This function does the same as rebranch_network except that only a few nodes (#NbRandomPerturbations)
are selected for rebranching at random.
%}
function I1=random_rebranch_network(param,graph,I0,res,options)
J=graph.J;

% ===============
% PERTURB NETWORK
% ===============

I1=I0;

link_list=randperm(graph.J,options.nb_random_perturbations);

% Rebranch each location to its lowest price parent
for i=link_list
    neighbors = graph.nodes{i}.neighbors;
    parents = neighbors ( res.PCj(neighbors) < res.PCj(i) );
    
    if length(parents)>=2
        [~,lowest_price_parent] = sort( res.PCj(parents) );
        lowest_price_parent = parents(lowest_price_parent(1));
        [~,best_connected_parent] = sort( I0(i,parents) , 'descend' );
        best_connected_parent = parents(best_connected_parent(1));
        
        % swap roads
        I1( i,lowest_price_parent ) = I0( i,best_connected_parent);
        I1( i,best_connected_parent ) = I0( i,lowest_price_parent);
        I1( lowest_price_parent,i ) = I0( i,best_connected_parent);
        I1( best_connected_parent,i ) = I0( i,lowest_price_parent);
        
    end
end

% Make sure graph satisfies symmetry and capacity constraint
I1=(I1+I1')/2; % make sure kappa is symmetric
I1=param.K*I1/sum(reshape(graph.delta_i.*I1,[graph.J^2 1])); % rescale
end % End of function



%%
%{
function I1=hybrid(param,graph,I0,res,options)
This function attempts to adapt the spirit of Alder (2018)'s algorithm to
delete/add links to the network in a way that blends with our model.
%}
function I2=hybrid(param,graph,I0,res,options)

% ========================
% COMPUTE GRADIENT WRT Ijk
% ========================

if ~param.cong % no cross-good congestion
    Pjkn=repmat(permute(res.Pjn,[1 3 2]),[1 graph.J 1]);
    PQ=Pjkn.*res.Qjkn.^(1+param.beta);
    grad = param.gamma*graph.delta_tau.*sum(PQ+permute(PQ,[2 1 3]),3).*I0.^-(1+param.gamma)./graph.delta_i;
    grad( graph.adjacency==false )=0;
else % cross-good congestion
    PCj=repmat(results.PCj,[1 graph.J]);
    matm=shiftdim(repmat(param.m,[1 graph.J graph.J]),1);
    cost=sum(matm.*results.Qjkn.^param.nu,3).^((param.beta+1)/param.nu);
    PQ=PCj.*cost;
    grad=param.gamma*graph.delta_tau.*(PQ+PQ').*I0.^-(1+param.gamma)./graph.delta_i;
    grad( graph.adjacency==false )=0;
end

% ============
% REMOVE LINKS: remove 5% worst links
% ============

I1=I0;

nremove=ceil(0.05*graph.ndeg); % remove 5% of the links
[B,id]=sort(grad(tril(graph.adjacency==1)),'ascend');
remove_list=id(1:nremove);
id=1;
for j=1:graph.J
    for k=1:length(graph.nodes{j}.neighbors)
        if any(id==remove_list) % if link is in the list to remove
            I1(j,k)=1e-6;
            I1(k,j)=1e-6;
        end
        id=id+1;
    end
end

% ====================
% COMPUTE NEW GRADIENT
% ====================

res = solve_allocation(param,graph,I1);

if ~param.cong % no cross-good congestion
    Pjkn=repmat(permute(res.Pjn,[1 3 2]),[1 graph.J 1]);
    PQ=Pjkn.*res.Qjkn.^(1+param.beta);
    grad = param.gamma*graph.delta_tau.*sum(PQ+permute(PQ,[2 1 3]),3).*I0.^-(1+param.gamma)./graph.delta_i;
    grad( graph.adjacency==false )=0;
else % cross-good congestion
    PCj=repmat(results.PCj,[1 graph.J]);
    matm=shiftdim(repmat(param.m,[1 graph.J graph.J]),1);
    cost=sum(matm.*results.Qjkn.^param.nu,3).^((param.beta+1)/param.nu);
    PQ=PCj.*cost;
    grad=param.gamma*graph.delta_tau.*(PQ+PQ').*I0.^-(1+param.gamma)./graph.delta_i;
    grad( graph.adjacency==false )=0;
end

% ========
% ADD LINK: add the most beneficial link
% ========

I2=I1;

[B,id]=sort(grad(tril(graph.adjacency==1)),'descend');
add_link=id(1);

id=1;
for j=1:graph.J
    for k=1:length(graph.nodes{j}.neighbors)
        if id==add_link % if link is in the list to remove
            I2(j,k)=param.K/(2*graph.ndeg);
            I2(k,j)=param.K/(2*graph.ndeg);
        end
        id=id+1;
    end
end

% =======
% RESCALE
% =======

% Make sure graph satisfies symmetry and capacity constraint
I2=(I2+I2')/2; % make sure kappa is symmetric
I2=param.K*I2/sum(reshape(graph.delta_i.*I2,[graph.J^2 1])); % rescale

end % End of function

%% ----------------------------
% RETRIEVE OPTIONS FROM INPUTS

function options = retrieve_options(param,graph,args)
p=inputParser;

% =============
% DEFINE INPUTS

checkSwitch = @(x) any(validatestring(x,{'on','off'}));

addParameter(p,'PerturbationMethod','random rebranching',@(x) any(validatestring(x,{'shake','random','rebranching','random rebranching','hybrid alder'})));
addParameter(p,'PreserveCentralSymmetry','off',checkSwitch);
addParameter(p,'PreserveVerticalSymmetry','off',checkSwitch);
addParameter(p,'PreserveHorizontalSymmetry','off',checkSwitch);
addParameter(p,'SmoothingRadius',0.25,@isnumeric);
addParameter(p,'MuPerturbation',log(.3),@isnumeric);
addParameter(p,'SigmaPerturbation',.05,@isnumeric);
addParameter(p,'Display','off',checkSwitch);
addParameter(p,'TStart',100,@isnumeric);
addParameter(p,'TEnd',1,@isnumeric);
addParameter(p,'TStep',0.9,@isnumeric);
addParameter(p,'NbDeepening',4,@isnumeric);
addParameter(p,'NbRandomPerturbations',1,@isnumeric);
addParameter(p,'Funcs',[]);
addParameter(p,'Iu',inf*ones(graph.J,graph.J), @(x) isequal(size(x),[graph.J graph.J]));
addParameter(p,'Il',zeros(graph.J,graph.J), @(x) isequal(size(x),[graph.J graph.J]));

% Parse inputs
parse(p,args{:});

% ==============
% RETURN OPTIONS

options.perturbation_method = p.Results.PerturbationMethod;
options.preserve_central_symmetry = strcmp(p.Results.PreserveCentralSymmetry,'on');
options.preserve_vertical_symmetry = strcmp(p.Results.PreserveVerticalSymmetry,'on');
options.preserve_horizontal_symmetry = strcmp(p.Results.PreserveHorizontalSymmetry,'on');
options.smoothing_radius = p.Results.SmoothingRadius;
options.mu_perturbation = p.Results.MuPerturbation;
options.sigma_perturbation = p.Results.SigmaPerturbation;
options.display = strcmp(p.Results.Display,'on');
options.t_start = p.Results.TStart;
options.t_end = p.Results.TEnd;
options.t_step = p.Results.TStep;
options.nb_deepening = round(p.Results.NbDeepening);
options.nb_random_perturbations = round(p.Results.NbRandomPerturbations);
options.funcs = p.Results.Funcs;
options.Iu = p.Results.Iu;
options.Il = p.Results.Il;
end % End of function

%% ==============
% kappa_extract()
% Description: auxiliary function that converts kappa_jk into kappa_i

function kappa_ex=kappa_extract(graph,kappa)
kappa_ex=zeros(graph.ndeg,1);
id=1;
for i=1:graph.J
    for j=1:length(graph.nodes{i}.neighbors)
        if graph.nodes{i}.neighbors(j)>i
            kappa_ex(id)=kappa(i,graph.nodes{i}.neighbors(j));
            id=id+1;
        end
    end
end
end


