%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

paper_example02.m: script that generate illustrative example 02 from the
paper.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

clear;
close all;


% 4.1.2 RANDOM CITIES
%
% Init and Solve network

width=9; height=9;
nb_cities=20;

param = init_parameters('TolKappa',1e-5,'K',100,'Annealing','off');
[param,g]=create_graph(param,width,height,'Type','triangle'); % case with random cities, one good

% set fundamentals

rng(5);
param.N = 1;
param.Zjn = param.minpop*ones(g.J,1); % matrix of productivity
param.Hj = ones(g.J,1); % matrix of housing
param.Lj = 0*ones(g.J,1); % matrix of population

Ni=find_node(g,ceil(width/2),ceil(height/2)); % find center
param.Zjn(Ni)=1; % more productive node
param.Lj(Ni)=1; % more productive node

for i=1:nb_cities-1
    newdraw=false;
    while newdraw==false
        j=nearest(1+rand()*(g.J-1));
        if param.Lj(j)<=param.minpop
            newdraw=true;
            param.Lj(j)=1;
        end
    end
end

param.hj=param.Hj./param.Lj;
param.hj(param.Lj==0)=1;

% Convex case
results(1)=optimal_network(param,g);

% Nonconvex - no annealing
param.gamma=2;
results(2)=optimal_network(param,g);

% Nonconvex - annealing
results(3)=annealing(param,g,results(2).Ijk,'PerturbationMethod','rebranching');

welfare_increase=(results(3).welfare/results(2).welfare)^(1/(param.alpha*(1-param.rho))); % compute welfare increase in consumption equivalent


% plot
close all;

s={'random_cities_convex','random_cities_nonconvex','random_cities_annealing'};
titles = [ {'Transport Network (I_{jk})','Shipping (Q_{jk})'};
           {'Transport Network (I_{jk})','Shipping (Q_{jk})'};
           {'Before annealing','After annealing'} ];

plots = [ {'results(i).Ijk','results(i).Qjkn'};
          {'results(i).Ijk','results(i).Qjkn'};
          {'results(2).Ijk','results(3).Ijk'}];

texts = [ {'',''};
         {'',''};
         {'Welfare = 1',sprintf('Welfare = %1.3f (cons. eq.)',welfare_increase)}];

arrows = [ {'off','on'};
           {'off','on'};
           {'off','off'} ];


for i=1:3
    fig=figure('Units','inches','Position',[0,0,7.5,3],'Name',char(s(i)));

    subplot(1,2,1);
    plot_graph(param,g,eval(char(plots(i,1))),'Sizes',1.2*param.Lj,'Arrows',char(arrows(i,1)));
    title(sprintf('(%c) %s',97+2*(i-1),char(titles(i,1))),'FontWeight','normal','FontName','Times','FontSize',9);
    text(.5,-.05,char(texts(i,1)),'HorizontalAlignment','center','Fontsize',8);


    subplot(1,2,2);
    plot_graph(param,g,eval(char(plots(i,2))),'Sizes',1.2*param.Lj,'Arrows',char(arrows(i,2)),'ArrowScale',1);
    title(sprintf('(%c) %s',97+2*(i-1)+1,char(titles(i,2))),'FontWeight','normal','FontName','Times','FontSize',9);
    text(.5,-.05,char(texts(i,2)),'HorizontalAlignment','center','Fontsize',8);


%     print('-depsc2',[graph_path,char(s(i)),'.eps'],'-r400');
%     saveas(fig,[graph_path,char(s(i)),'.jpg']);

end


