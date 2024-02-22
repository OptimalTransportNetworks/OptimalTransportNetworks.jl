%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

example02.m: script that computes the optimal network for a geography with
20 random cities around a more productive central node and a unique good.

This example only makes sense without labor mobility.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2018, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

close all;
clear all;

% ==============
% INITIALIZATION
% ==============

param = init_parameters('LaborMobility','off','K',100,'gamma',1,'beta',1,'N',1,'Annealing','off','ADiGator','off');

% ------------
% Init network

w=13; h=13;
[param,graph]=create_graph(param,w,h,'Type','triangle'); % create a triangular network of 21x21

ncities = 20; % number of random cities
param.Lj=zeros(param.J,1); % set population to minimum everywhere
param.Zjn=0.01*ones(param.J,1); % set low productivity everywhere

Ni=find_node(graph,ceil(w/2),ceil(h/2)); % Find the central node
param.Zjn(Ni)=1; % central node is more productive
param.Lj(Ni)=1; % cities are equally populated

% Draw the rest of the cities randomly
rng(5); % reinit random number generator
for i=1:ncities
    newdraw=false;
    while newdraw==false
        j=round(1+rand()*(graph.J-1));
        if param.Lj(j)~=1/(ncities+1) % make sure node j is unpopulated           
            newdraw=true;
            param.Lj(j)=1;
        end
    end
end


% ==========
% RESOLUTION
% ==========

tic;

% first, compute the optimal network in the convex case (beta>=gamma)
res(1)=optimal_network(param,graph);

% second, in the nonconvex case (gamma>beta)
param = init_parameters('param',param,'gamma',2); % change only gamma, keep other parameters
res(2)=optimal_network(param,graph); % optimize by iterating on FOCs
res(3)=annealing(param,graph,res(2).Ijk); % improve with annealing, starting from previous result

toc;

%% % Plot them 

close all;
figure('Position',[200 200 1500 400]);
labels={'Convex','Nonconvex (FOC)','Nonconvex (annealing)'};

for i=1:3
    subplot(1,3,i);
    results=res(i);
    sizes=4*results.Cj/max(results.Cj);
    shades=results.Cj/max(results.Cj);
    plot_graph(param,graph,results.Ijk,'Sizes',sizes,'Shades',results.Cj/max(results.Cj),...
        'NodeFgColor',[1 .9 .4],'NodeBgColor',[.8 .1 .0],'NodeOuterColor',[0 .5 .6],...
        'EdgeColor',[0 .4 .8],'MaxEdgeThickness',4);
    
    text(0.5,0.05,labels{i},'HorizontalAlignment','center');
end
