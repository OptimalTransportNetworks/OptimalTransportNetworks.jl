%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

apply_geography (graph,geography,varargin): applies a geography structure to a graph by updating
its network building costs and removing edges where geographical barriers (obstacles) are.

The geography structure should contain the fields:
- z: a Jx1 vector containing the z coordinate of each node
- obstacles: a Nobs*2 matrix of (i,j) pairs of nodes that are linked by the
obstacles (Nobs is an arbitrary number of obstacles)

The function apply_geography takes the geography structure as an input and
uses it to calibrate the building cost delta_i. Aversion to altitude
changes rescales building costs by (see documentation):
    1+alpha_up*max(0,Z2-Z1)^beta_up+alpha_down*max(0,Z1-Z2)^beta_down.

Optional arguments:
- 'AcrossObstacleDelta_i': rescaling parameter for building cost that cross
an obstacle (default inf)
- 'AlongObstacleDelta_i': rescaling parameter for building cost that goes along
an obstacle (default 1)
- 'AcrossObstacleDelta_tau': rescaling parameter for transport cost that cross
an obstacle (default inf)
- 'AlongObstacleDelta_tau': rescaling parameter for transport cost that goes along nn obstacle (default 1)
- 'AlphaUp_i': building cost parameter (scale) for roads that go up in elevation
(default 0)
- 'BetaUp_i': building cost parameter (elasticity) for roads that go up in elevation
(default 1)
- 'AlphaUp_tau': transport cost parameter (scale) for roads that go up in elevation
(default 0)
- 'BetaUp_tau': transport cost parameter (elasticity) for roads that go up in elevation
(default 1)
- 'AlphaDown_i': building cost parameter (scale) for roads that go down in elevation
(default 0)
- 'BetaDown_i': building cost parameter (elasticity) for roads that go down in elevation
(default 1)
- 'AlphaDown_tau': transport cost parameter (scale) for roads that go down in elevation
(default 0)
- 'BetaDown_tau': transport cost parameter (elasticity) for roads that go down in elevation
(default 1)

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}


function graph = apply_geography (graph, geography, varargin)
options = retrieve_options(varargin);

% Embed geographical barriers into network building costs

for i=1:graph.J
    for j=1:length(graph.nodes{i}.neighbors)
        distance=sqrt( (graph.x(i)-graph.x(graph.nodes{i}.neighbors(j)))^2 + (graph.y(i)-graph.y(graph.nodes{i}.neighbors(j)))^2 );
        Delta_z=geography.z(graph.nodes{i}.neighbors(j))-geography.z(i);
        graph.delta_i(i,graph.nodes{i}.neighbors(j)) = distance * (1 + options.alpha_up_i * max(0, Delta_z )^options.beta_up_i + options.alpha_down_i * max(0, -Delta_z )^options.beta_down_i);
        graph.delta_tau(i,graph.nodes{i}.neighbors(j)) = distance * (1 + options.alpha_up_tau * max(0, Delta_z )^options.beta_up_tau + options.alpha_down_tau * max(0, -Delta_z )^options.beta_down_tau);
    end
end

% Remove edges where geographical barriers are (rivers)

sz=size(geography.obstacles);
graph.across_obstacle=zeros(graph.J,graph.J);
graph.along_obstacle=zeros(graph.J,graph.J);

% Store initial delta matrics (avoid double counting)
delta_i = graph.delta_i;
delta_tau = graph.delta_tau;

for i=1:graph.J
    neighbors=graph.nodes{i}.neighbors;
    for j=1:length(neighbors)
        % check if edge (i,j) intersects any of the obstacles
        k=1;
        has_been_destroyed=false;
        while ~has_been_destroyed && k<=sz(1)
            %             if i~=geography.obstacles(k,1) && neighbors(j)~=geography.obstacles(k,2)
            if ~isempty(setdiff([i neighbors(j)],geography.obstacles(k,:))) % if the link is not the obstacle itself
                                Z2_obj=[graph.x(geography.obstacles(k,2));graph.y(geography.obstacles(k,2))];
                                Z1_obj=[graph.x(geography.obstacles(k,1));graph.y(geography.obstacles(k,1))];
                                Z2=[graph.x(neighbors(j));graph.y(neighbors(j))];
                                Z1=[graph.x(i);graph.y(i)];
                                
                vec=Z2_obj-Z1_obj;
                normvec=[-vec(2);vec(1)];
                val1 = (normvec'*(Z1-Z1_obj))*(normvec'*(Z2-Z1_obj));
                
                vec=Z2-Z1;
                normvec=[-vec(2);vec(1)];
                val2 = (normvec'*(Z1_obj-Z1))*(normvec'*(Z2_obj-Z1));
                
                if (val1<=0 && val2<=0)  && normvec'*(Z2_obj-Z1_obj)~=0 % if the two edges intersect and are not colinears
                    if isinf(options.across_obstacle_delta_i) || isinf(options.across_obstacle_delta_tau) % remove the edge
                        graph.nodes{neighbors(j)}.neighbors=setdiff(graph.nodes{neighbors(j)}.neighbors,i);
                        graph.adjacency(i,neighbors(j))=0;
                        graph.adjacency(neighbors(j),i)=0;
                        graph.nodes{i}.neighbors=setdiff(graph.nodes{i}.neighbors,neighbors(j));
                        has_been_destroyed=true;
                    else
                        % or make it costly to cross
                        graph.delta_i(i,neighbors(j))=options.across_obstacle_delta_i*delta_i(i,neighbors(j));
                        graph.delta_i(neighbors(j),i)=options.across_obstacle_delta_i*delta_i(neighbors(j),i);
                        graph.delta_tau(i,neighbors(j))=options.across_obstacle_delta_tau*delta_tau(i,neighbors(j));
                        graph.delta_tau(neighbors(j),i)=options.across_obstacle_delta_tau*delta_tau(neighbors(j),i);
                        graph.across_obstacle(i,neighbors(j))=true;
                        graph.across_obstacle(neighbors(j),i)=true;
                    end
                end
            end
            k=k+1;
        end
    end
end

% allow for different delta while traveling along obstacles (e.g., water transport)

for i=1:sz(1)
    if ~isinf(options.along_obstacle_delta_i) && ~isinf(options.along_obstacle_delta_tau)
        graph.delta_i(geography.obstacles(i,1),geography.obstacles(i,2))=options.along_obstacle_delta_i*graph.delta_i(geography.obstacles(i,1),geography.obstacles(i,2));
        graph.delta_i(geography.obstacles(i,2),geography.obstacles(i,1))=options.along_obstacle_delta_i*graph.delta_i(geography.obstacles(i,2),geography.obstacles(i,1));
        graph.delta_tau(geography.obstacles(i,1),geography.obstacles(i,2))=options.along_obstacle_delta_tau*graph.delta_tau(geography.obstacles(i,1),geography.obstacles(i,2));
        graph.delta_i(geography.obstacles(i,2),geography.obstacles(i,1))=options.along_obstacle_delta_tau*graph.delta_tau(geography.obstacles(i,2),geography.obstacles(i,1));
        graph.along_obstacle(geography.obstacles(i,1),geography.obstacles(i,2))=true;
        graph.along_obstacle(geography.obstacles(i,2),geography.obstacles(i,1))=true;
        graph.across_obstacle(geography.obstacles(i,1),geography.obstacles(i,2))=false;
        graph.across_obstacle(geography.obstacles(i,2),geography.obstacles(i,1))=false;
    else % if infinite, remove edge
        graph.nodes{geography.obstacles(i,1)}.neighbors=setdiff(graph.nodes{geography.obstacles(i,1)}.neighbors,geography.obstacles(i,2));
        graph.nodes{geography.obstacles(i,2)}.neighbors=setdiff(graph.nodes{geography.obstacles(i,2)}.neighbors,geography.obstacles(i,1));
        graph.adjacency(geography.obstacles(i,1),geography.obstacles(i,2))=0;
        graph.adjacency(geography.obstacles(i,2),geography.obstacles(i,1))=0;
    end
end

% make sure that the degrees of freedom of the updated graph match the # of
% links
graph.ndeg = sum( reshape(tril(graph.adjacency),[graph.J*graph.J 1]) );



% ----------------------------
% RETRIEVE OPTIONS FROM INPUTS

function options = retrieve_options(args)
p=inputParser;

% =============
% DEFINE INPUTS

addParameter(p,'AcrossObstacleDelta_i',1,@(x) (isnumeric(x) | isinf(x)));
addParameter(p,'AlongObstacleDelta_i',1,@(x) (isnumeric(x) | isinf(x)));
addParameter(p,'AcrossObstacleDelta_tau',1,@(x) (isnumeric(x) | isinf(x)));
addParameter(p,'AlongObstacleDelta_tau',1,@(x) (isnumeric(x) | isinf(x)));

addParameter(p,'AlphaUp_i',0,@isnumeric);
addParameter(p,'BetaUp_i',1,@isnumeric);
addParameter(p,'AlphaUp_tau',0,@isnumeric);
addParameter(p,'BetaUp_tau',1,@isnumeric);

addParameter(p,'AlphaDown_i',0,@isnumeric);
addParameter(p,'BetaDown_i',1,@isnumeric);
addParameter(p,'AlphaDown_tau',0,@isnumeric);
addParameter(p,'BetaDown_tau',1,@isnumeric);

% Parse inputs
parse(p,args{:});

% ==============
% RETURN OPTIONS

options.across_obstacle_delta_i = p.Results.AcrossObstacleDelta_i;
options.along_obstacle_delta_i = p.Results.AlongObstacleDelta_i;
options.across_obstacle_delta_tau = p.Results.AcrossObstacleDelta_tau;
options.along_obstacle_delta_tau = p.Results.AlongObstacleDelta_tau;

options.alpha_up_i = p.Results.AlphaUp_i;
options.beta_up_i = p.Results.BetaUp_i;
options.alpha_up_tau = p.Results.AlphaUp_tau;
options.beta_up_tau = p.Results.BetaUp_tau;
options.alpha_down_i = p.Results.AlphaDown_i;
options.beta_down_i = p.Results.BetaDown_i;
options.alpha_down_tau = p.Results.AlphaDown_tau;
options.beta_down_tau = p.Results.BetaDown_tau;
