%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

example03.m: script that computes the optimal network for a geography with
5+1 goods, 5 of which are produced in unique random locations, the last ('agricultural') which 
is produced everywhere else.

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
% nonconvex: gamma>betam, CrossGoodCongestion: on/off, or ADiGator: on/off

param = init_parameters('LaborMobility','on','K',100,'gamma',2,'beta',1,'N',6,'TolKappa',1e-4,...
                        'CrossGoodCongestion','off','ADiGator','off');

% ------------
% Init network

[param,graph]=create_graph(param,7,7,'Type','triangle'); % create a triangular network of 21x21

param.Zjn(:,1:param.N-1)=0; % default locations cannot produce goods 1-10
param.Zjn(:,param.N)=1; % but they can all produce good 11 (agricultural)

% Draw the cities randomly
% rng(5); % reinit random number generator
for i=1:param.N-1
    newdraw=false;
    while newdraw==false
        j=round(1+rand()*(graph.J-1));
        if any(param.Zjn(j,1:param.N-1)>0)==0 % make sure node j does not produce any differentiated good
            newdraw=true;
            param.Zjn(j,1:param.N)=0;
            param.Zjn(j,i)=1;
        end
    end
end


% ==========
% RESOLUTION
% ==========

tic;
res=optimal_network(param,graph);
toc;

%% % Plot them 

close all;

rows=ceil((param.N+1)/4);
cols=min(4,param.N+1);

figure('Position',[200 200 cols*400 rows*300]);

subplot(rows,cols,1);
results=res;
plot_graph(param,graph,results.Ijk);
title('(a) Network I');

for i=1:param.N
    subplot(rows,cols,i+1);
    results=res;
    plot_graph(param,graph,results.Qjkn(:,:,i),'Arrows','on','ArrowScale',1.5);
    title(sprintf('(%c) Flows good %i',97+i,i));
end

