%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[param,graph]=remove_node(param,graph,i): removes node i from the graph

Arguments:
- param: structure that contains the model's parameters
- graph: structure that contains the underlying graph (created by
create_graph)
- i: index of the mode to be removed (integer between 1 and graph.J)

Returns the updated graph and param structure (param is affected too
because the variable Zjn, Lj, Hj and others are changed).


-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function [param,graph]=remove_node(param,graph,i)

if i<1 || i>graph.J || (i-floor(i)~=0)
    fprintf('%s.m: node i should be an integer between 1 and %d.\n',mfilename,graph.J);
    return;
end

% create new nodes structure
nodes{graph.J-1}=struct('neighbors',[]);
x=zeros(graph.J-1,1);
y=zeros(graph.J-1,1);

for k=1:graph.J-1
    cursor=k + double(k>=i);
    
    nodes{k}.neighbors=setdiff(graph.nodes{cursor}.neighbors,i); % remove this nove from others' neighbors.
    nodes{k}.neighbors(nodes{k}.neighbors>i)=nodes{k}.neighbors(nodes{k}.neighbors>i)-1; % reindex nodes k > i to k-1
    
    x(k)=graph.x(cursor);
    y(k)=graph.y(cursor);
end

% rebuild adjacency matrix, delta_i and delta_tau
adjacency = [graph.adjacency(1:i-1,1:i-1) graph.adjacency(1:i-1,i+1:end);
          graph.adjacency(i+1:end,1:i-1) graph.adjacency(i+1:end,i+1:end)];

delta_i = [graph.delta_i(1:i-1,1:i-1) graph.delta_i(1:i-1,i+1:end);
          graph.delta_i(i+1:end,1:i-1) graph.delta_i(i+1:end,i+1:end)];
      
delta_tau = [graph.delta_i(1:i-1,1:i-1) graph.delta_i(1:i-1,i+1:end);
          graph.delta_i(i+1:end,1:i-1) graph.delta_i(i+1:end,i+1:end)];
      
ndeg = sum(reshape(tril(adjacency),[(graph.J-1)^2,1]));

% return new graph structure
graph = struct('J',graph.J-1,'x',x,'y',y,'nodes',{nodes},'adjacency',adjacency,...
               'delta_i',delta_i,'delta_tau',delta_tau,'ndeg',ndeg);

% now, update the param structure
param.J = graph.J;
param.Lj = [param.Lj(1:i-1);param.Lj(i+1:end)];
param.Hj = [param.Hj(1:i-1);param.Hj(i+1:end)];
param.hj = [param.hj(1:i-1);param.hj(i+1:end)];
param.omegaj = [param.omegaj(1:i-1);param.omegaj(i+1:end)];
param.Zjn = [param.Zjn(1:i-1,:);param.Zjn(i+1:end,:)];

end % End of function
