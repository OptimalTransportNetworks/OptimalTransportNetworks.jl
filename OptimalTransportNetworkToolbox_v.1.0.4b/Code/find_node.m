%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

[id]=find_node(graph,x,y): returns the index of the node closest to the 
coordinates (x,y) on the graph

Arguments:
- graph: structure that contains the underlying graph (created by
create_map, create_rectangle or create_triangle functions)
- x: x coordinate on the graph between 1 and w
- y: y coordinate on the graph between 1 and h

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function id=find_node(graph,x,y)

distance=(graph.x-x).^2 + (graph.y-y).^2;
[~,id]=min(distance);

end % End of function
