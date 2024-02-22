%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

paper_example01.m: script that generate illustrative example 01 from the
paper.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2018, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

clear;
close all;

%% 4.1.1 One Good on a Regular Geometry
%. COMPARATIVE STATICS OVER K IN A SYMMETRIC NETWORK

% ==============
% INITIALIZATION
% ==============

K=[1,100];
print_dummy=[1,1];
param = init_parameters();
[param,g]=create_graph(param,9,9,'Type','map');

% Define fundamentals
param.N = 1;
param.Zjn = 0.1*ones(g.J,1); % matrix of productivity
Ni=find_node(g,5,5); % center
param.Zjn(Ni)=1; % more productive node

%% Plot the mesh
s='simple_geography_mesh_population';
fig=figure('Units','inches','Position',[0,0,7.5,7.5],'Name',s);
set(gcf,'PaperUnits',get(gcf,'Units'),'PaperPosition',get(gcf,'Position')+[-.25 2.5 0 0]);
plot_graph(param,g,[],'Edges','off','Mesh','on','MeshColor',[.2 .4 .6]);
box off;
axis off;

s='simple_geography_mesh_productivity';
fig=figure('Units','inches','Position',[0,0,7.5,7.5],'Name',s);
set(gcf,'PaperUnits',get(gcf,'Units'),'PaperPosition',get(gcf,'Position')+[-.25 2.5 0 0]);
sizes=ones(g.J,1);
sizes(find_node(g,5,5))=2.5;
plot_graph(param,g,[],'Edges','off','Shades',(param.Zjn-.1)/.9,'Sizes',sizes,'NodeFgColor',[.6 .8 1],'Mesh','on','MeshColor',[.2 .4 .6]);
box off;
axis off;

%% Compute networks
for i=1:length(K) % Solve for the optimal network for each K
    param.K=K(i);
    results(i,1)=optimal_network(param,g);
end

%% Plot networks

for k=1:length(K)
    if print_dummy(k)==true
        s=sprintf('Comparative_statics_K=%d',K(k));

        fig=figure('Units','inches','Position',[0,0,7.5,6],'Name',s);
        set(gcf,'PaperUnits',get(gcf,'Units'),'PaperPosition',get(gcf,'Position')+[-.25 2.5 0 0]);
        set(gca,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',9,'FontName','Times');

        hor_scaling=1.1;
        ver_scaling=1.05;

        h=subplot(2,2,1);
        size=.25*ones(g.J,1);
        plot_graph( param,g,results(k,1).Ijk,'Sizes',size );
        pos=get(h,'Position');
        set(h,'Position',[pos(1) pos(2) hor_scaling*pos(3) ver_scaling*pos(4)]);
        title('(a) Transport Network (I_{jk})','FontWeight','normal','FontName','Times','FontSize',9);

        h=subplot(2,2,2);
        plot_graph( param,g,results(k,1).Qjkn,'Nodes','off','Arrows','on','ArrowScale',1 );
        pos=get(h,'Position');
        set(h,'Position',[pos(1) pos(2) hor_scaling*pos(3) ver_scaling*pos(4)]);
        title('(b) Shipping (Q_{jk})','FontWeight','normal','FontName','Times','FontSize',9);

        h=subplot(2,2,3);
        plot_graph( param,g,results(k,1).Ijk,'Nodes','off','Map', results(k,1).Pjn/max(results(k,1).Pjn) );
        pos=get(h,'Position');
        set(h,'Position',[pos(1) pos(2) hor_scaling*pos(3) ver_scaling*pos(4)]);
        colorbar();
        title('(c) Prices (P_{j})','FontWeight','normal','FontName','Times','FontSize',9);

        h=subplot(2,2,4);
        plot_graph( param,g,results(k,1).Ijk,'Nodes','off','Map',results(k,1).cj/max(results(k,1).cj) );
        hold off;
        pos=get(h,'Position');
        set(h,'Position',[pos(1) pos(2) hor_scaling*pos(3) ver_scaling*pos(4)]);
        colorbar();
        title('(d) Consumption (c_{j})','FontWeight','normal','FontName','Times','FontSize',9);
        
    end
end



