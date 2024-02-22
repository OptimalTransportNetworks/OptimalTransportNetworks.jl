%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

example01.m: script that computes the optimal network for a regular
geometry (11x11 nodes, square) with equally distributed population and
productivity, except for the more productive central node

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2018, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

close all;

% ==============
% INITIALIZATION
% ==============

% Set parameters: try with LaborMobility: on/off, convex: beta>=gamma,
% nonconvex: gamma>beta, CrossGoodCongestion: on/off, or ADiGator: on/off

param = init_parameters('LaborMobility','on','K',10,'gamma',1,'beta',1,...
    'N',1,'TolKappa',1e-5,'ADiGator','on','CrossGoodCongestion','off','nu',1);

% TolKappa is the tolerance in distance b/w iterations for road capacity
% kappa=I^gamma/delta_i, default is 1e-7 but we relax it a bit here

% ------------
% Init network

[param,graph]=create_graph(param,11,11,'Type','map'); % create a map network of 11x11 nodes located in [0,10]x[0,10]
% note: by default, productivity and population are equalized everywhere

% Customize graph
param.Zjn=0.1*ones(param.J,1); % set most places to low productivity
Ni=find_node(graph,6,6); % Find index of the central node at (6,6)
param.Zjn(Ni)=1; % central node more productive

% ==========
% RESOLUTION
% ==========

tic;
results=optimal_network(param,graph);
toc;

%% % Plot them 

sizes=1.5*(results.Cj-min(results.Cj))/(max(results.Cj)-min(results.Cj)); % size of each node
shades=(results.Cj-min(results.Cj))/(max(results.Cj)-min(results.Cj)); % shading for each node in [0,1] between NodeBgColor and NodeFgColor

plot_graph(param,graph,results.Ijk,'Sizes',sizes,'Shades',shades,...
    'NodeFgColor',[1 .9 .4],'NodeBgColor',[.8 .1 .0],'NodeOuterColor',[0 .5 .6]);

% the size/shade of the nodes reflects the total consumption at each node


