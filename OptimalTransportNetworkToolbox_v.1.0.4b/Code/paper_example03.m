%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

paper_example03.m: script that generate illustrative example 03 from the
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

%% ------------------------------------
% 4.2 Random Cities with Multiple Goods
% -------------------------------------

ngoods=10;
width=9; height=9;

% Init and Solve network

param = init_parameters('TolKappa',1e-5,'K',10000,'LaborMobility','on','Annealing','off');
[param,g]=create_graph(param,width,height,'Type','triangle');

% set fundamentals

param.N = 1+ngoods;
param.Zjn = 0*ones(g.J,param.N); % matrix of productivity

param.Zjn(:,1)=1; % the entire countryside produces the homogenenous good 1

if ngoods>0
    Ni=find_node(g,5,5); % center
    param.Zjn(Ni,2)=1; % central node
    param.Zjn(Ni,1)=0; % central node
    
    rng(5);
    for i=2:ngoods
        newdraw=false;
        while newdraw==false
            j=round(1+rand()*(g.J-1));
            if param.Zjn(j,1)>0
                newdraw=true;
                param.Zjn(j,i+1)=1;
                param.Zjn(j,1)=0;
            end
        end
    end
end

% Convex case
results(1)=optimal_network(param,g);

% Nonconvex
param.gamma=2;
results(2)=optimal_network(param,g,results(1).Ijk);


%% Plot results
close all;

cols = 3; %4 % number of columns
rows = ceil((1+param.N)/cols);

s={'random_cities_multigoods_convex','random_cities_multigoods_nonconvex'};

for j=1:2
    fig=figure('Units','inches','Position',[0,0,7.5,11],'Name',char(s(j)));

    % Plot network
    subplot(rows,cols,1);
    plot_graph( param,g,results(j).Ijk, 'Shades', (results(j).Lj-min(results(j).Lj))./(max(results(j).Lj)-min(results(j).Lj)), 'Sizes', 1+16*(results(j).Lj./mean(results(j).Lj)-1), 'NodeFgColor',[.6 .8 1],'Transparency','off' );
    h=title('(a) Transport Network (I_{jk})','FontWeight','normal','FontName','Times','FontSize',9);

    for i=1:param.N
        subplot(rows,cols,i+1);
        sizes=3*results(j).Yjn(:,i)./sum(results(j).Yjn(:,i));
        shades=param.Zjn(:,i)./max(param.Zjn(:,i));
        plot_graph( param,g,results(j).Qjkn(:,:,i),'Arrows','on','ArrowScale',1,'ArrowStyle','thin','Nodes','on','Sizes',sizes,'Shades',shades, 'NodeFgColor',[.6 .8 1],'Transparency','off' );
        title(sprintf('(%c) Shipping (Q^{%d}_{jk})',97+i,i),'FontWeight','normal','FontName','Times','FontSize',9);
    end

    % Save
%     print('-depsc2',[graph_path,char(s(j)),'.eps'],'-r400');
%     saveas(fig,[graph_path,char(s(j)),'.jpg']);
end

