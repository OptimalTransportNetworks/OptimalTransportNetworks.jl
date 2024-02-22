%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

create_auxdata.m: creates the auxdata structure that contains all the
auxiliary parameters for ADiGator and solve_allocation_...()

Arguments:
- param: structure that contains the model's parameters
- graph: structure that contains the underlying graph (created by
create_graph function)
- I: provides the current JxJ symmetric matrix of infrastructure investment

Output:
- auxdata: structure auxdata to be used by the IPOPT/ADiGator bundle.

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}


function auxdata = create_auxdata(param,graph,I)

% ----------------
% Initialize kappa

kappa=I.^param.gamma./graph.delta_tau;
kappa=max(kappa,param.MIN_KAPPA);
kappa(~graph.adjacency)=0;
kappa_ex=kappa_extract(graph,kappa); % extract the ndeg free values of matrix kappa (due to symmetry)


% ---------------
% Create matrix A

% Note: matrix A is of dimension J*ndeg and takes value 1 when node J is
% the starting point of edge ndeg, and -1 when it is the ending point. It 
% is used to make hessian and jacobian calculations using matrix algebra

A=zeros(graph.J,graph.ndeg);

id=1;
for j=1:graph.J
    for k=1:length(graph.nodes{j}.neighbors)
        if graph.nodes{j}.neighbors(k)>j
            A(j,id)=1;
            A(graph.nodes{j}.neighbors(k),id)=-1;
            id=id+1;
        end
    end
end


% ----------------
% Store in auxdata

auxdata.param=param;
auxdata.graph=graph;
auxdata.kappa=kappa;
auxdata.kappa_ex=kappa_ex;
auxdata.Iex=kappa_extract(graph,I);
auxdata.delta_tau_ex=kappa_extract(graph,graph.delta_tau);
auxdata.A=A;
auxdata.Apos=max(A,0);
auxdata.Aneg=max(-A,0);

end


% ---------------
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

