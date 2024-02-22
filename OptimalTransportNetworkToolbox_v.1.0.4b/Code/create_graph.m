%{
 ==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

create_graph.m: initializes the underlying graph, population and
productivity parameters

Arguments:
- param: structure that contains the model's paramters
- w: number of nodes along the width of the underlying graph (should be an integer)
- h: number of nodes along the height of the underlying graph (should be an integer, odd if triangle)

Optional arguments:
- 'Type': either 'map', 'square', 'triangle', or 'custom'
- 'ParetoWeights': Jx1 vector of Pareto weights for each node (default ones(J,1))
- 'Adjacency': JxJ adjacency matrix (only used for custom network)
- 'NRegions': number of regions (only for partial mobility)
- 'Regions': Jx1 vector indicating the region of location j (only for
partial mobility)

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}


function [param,graph] = create_graph(param,w,h,varargin)

% ----------------
% Retrieve options
options = retrieve_options(param,w,h,varargin);

% --------------
% Create network

switch options.type
    case 'map'
        graph = create_map(w,h);
    case 'triangle'
        graph = create_triangle(w,h);
    case 'square'
        graph = create_square(w,h);
    case 'custom'
        graph = create_custom(options.adjacency,options.x,options.y);
end

param.J = graph.J;

% ------------------------------------
% Default distribution of fundamentals

% by default: population equally distributed and equal productivity

if param.mobility == false
    param.Lj=ones(param.J,1)/param.J; % only used if labor mobility is off
    param.Zjn=ones(param.J,param.N);
    param.Hj=ones(param.J,1);
    param.hj=param.Hj./param.Lj;
    param.hj(param.Lj==0)=1; % catch potential errors when unpopulated
    param.omegaj = options.omega;
elseif param.mobility == true
    param.Zjn=ones(param.J,param.N);
    param.Hj=ones(param.J,1);
elseif param.mobility==0.5
    param.Zjn=ones(param.J,param.N);
    param.Hj=ones(param.J,1);
    graph.region=options.region;
    param.nregions=options.nregions;
    param.omegar = options.omega;
    param.Lr=ones(options.nregions,1)/options.nregions;
end

end % End of function

function options = retrieve_options(param,w,h,args)
p=inputParser;

% ====================================
% DEFINE STANDARD VALUES OF PARAMETERS

default_type='map';
default_paretoweights=ones(w*h,1);
default_adj=[];
default_x=[];
default_y=[];


% =============
% DEFINE INPUTS

addParameter(p,'Type',default_type,@(x) any(validatestring(x,{'map','triangle','square','custom'})));
addParameter(p,'ParetoWeights',default_paretoweights,@isvector);
addParameter(p,'X',[],@isvector);
addParameter(p,'Y',[],@isvector);
addParameter(p,'Adjacency',[],@isadjacency);
addParameter(p,'NRegions',1,@isnumeric);
addParameter(p,'Region',[],@isvector);

% Parse inputs
parse(p,args{:});

% ==============
% RETURN OPTIONS

options.type = p.Results.Type;
options.omega = p.Results.ParetoWeights;
options.adjacency = p.Results.Adjacency;
options.x = p.Results.X;
options.y = p.Results.Y;
options.nregions = p.Results.NRegions;
options.region = p.Results.Region;

options.J=w*h;

% if triange
if strcmp(options.type,'triangle')
    options.J=w*ceil(h/2)+(w-1)*(ceil(h/2)-1);        
end

% if custom
if strcmp(options.type,'custom') % if using custom type
    if strmatch('Adjacency', p.UsingDefaults) % check that adjacency matrix is provided
        error('%s.m: Custom network requires an adjacency matrix to be provided.\n',mfilename);
    end
    
    if strmatch('X', p.UsingDefaults) | strmatch('Y', p.UsingDefaults) % check that X and Y coordinates are provided
        error('%s.m: X and Y coordinates of locations must be provided.\n',mfilename);
    end
    
    if ~isequal(size(options.x),size(options.y)) % check if X and Y have same size
        error('%s.m: the provided X and Y do not have the same size.\n',mfilename);
    end
    
    if size(options.adjacency,1)~=length(options.x) % check if X and adjacency have same nb of locations
        error('%s.m: the adjacency matrix and X should have the same number of locations.\n',mfilename);
    end
    
    options.J=length(options.x);
end

if param.mobility==0.5 && any(strcmp(p.UsingDefaults,'Region')) % if no vector of region is specified
	options.region = ones(options.J,1); % all locations belong to region 1 by default
end

if param.mobility==0.5 && any(strcmp(p.UsingDefaults,'NRegions')) && ~any(strcmp(p.UsingDefaults,'Region')) % if no nb of regions is specified
	options.nregions = length(unique(options.region)); % count the number of regions
end

if param.mobility==0.5 && ~any(strcmp(p.UsingDefaults,'NRegions')) && ~any(strcmp(p.UsingDefaults,'Region')) % if no nb of regions is specified
	if length(unique(options.region)) > options.nregions % count the number of regions
        error('%s.m: NRegions does not match the provided Region vector.\n',mfilename);
    end
end

% Pareto weights
if strmatch('ParetoWeights',p.UsingDefaults) && param.mobility==0 % if Pareto weights are not provided
    options.omega=ones(options.J,1);
end

if strmatch('ParetoWeights',p.UsingDefaults) && param.mobility==0.5% if Pareto weights are not provided
    options.omega=ones(options.nregions,1);
end

if length(options.omega)~=options.J && param.mobility==0
    fprintf('%s.m: Pareto weights should be a vector of size %d. Using default instead.\n',mfilename,options.J);
    options.omega = ones(options.J,1);
end

if length(options.omega)~=options.nregions && param.mobility==0.5
    fprintf('%s.m: Pareto weights should be a vector of size %d. Using default instead.\n',mfilename,options.nregions);
    options.omega = ones(options.nregions,1);
end

end % End of function

%{
res=isadjacency(M): checks if matrix M is an adjacency matrix

Arguments:
- M: some matrix

%}

function res=isadjacency (M)
res = true;

% check is matrix is square
sz=size(M);
if sz(1)~=sz(2)
    fprintf('%s.m: adjacency matrix is not square.\n',mfilename);
    res = false;
end

% check is matrix is symmetric
if any(M-M'~=0)
    fprintf('%s.m: adjacency matrix is not symmetric.\n',mfilename);
    res = false;
end

% check if matrix has only 0's and 1's
if any(M~=0 & M~=1)
    fprintf('%s.m: adjacency matrix should have only 0s and 1s.\n',mfilename);
    res = false;
end

end

%{
[graph]=create_map(w,h): creates a square graph structure with
width w and height h (nodes have 8 neighbors in total, along
horizontal and vertical dimensions and diagonals)

Arguments:
- w: width of graph (ie. the number of nodes along horizontal
dimension), must be an integer
- h: height of graph (ie. the number of nodes along vertical
dimension), must be an integer

%}

function graph = create_map(w,h)
J=w*h;
nodes{J}=struct('neighbors',[]);

delta=zeros(J,J); % matrix of building costs
x=zeros(J,1);
y=zeros(J,1);
for i=1:J
    neighbors=[];
    
    y(i)=floor((i-1)/w)+1;
    x(i)=i-w*(y(i)-1)+0;
    
    if x(i)<w
        neighbors = [neighbors,x(i)+1+w*(y(i)-1)];
        delta(x(i)+w*(y(i)-1),x(i)+1+w*(y(i)-1))=1;
    end
    if x(i)>1
        neighbors = [neighbors,x(i)-1+w*(y(i)-1)];
        delta(x(i)+w*(y(i)-1),x(i)-1+w*(y(i)-1))=1;
    end
    if y(i)<h
        neighbors = [neighbors,x(i)+w*(y(i)+1-1)];
        delta(x(i)+w*(y(i)-1),x(i)+w*(y(i)+1-1))=1;
    end
    if y(i)>1
        neighbors = [neighbors,x(i)+w*(y(i)-1-1)];
        delta(x(i)+w*(y(i)-1),x(i)+w*(y(i)-1-1))=1;
    end
    if x(i)<w && y(i)<h
        neighbors = [neighbors,x(i)+1+w*(y(i)+1-1)];
        delta(x(i)+w*(y(i)-1),x(i)+1+w*(y(i)+1-1))=sqrt(2);
    end
    if x(i)<w && y(i)>1
        neighbors = [neighbors,x(i)+1+w*(y(i)-1-1)];
        delta(x(i)+w*(y(i)-1),x(i)+1+w*(y(i)-1-1))=sqrt(2);
    end
    if x(i)>1 && y(i)<h
        neighbors = [neighbors,x(i)-1+w*(y(i)+1-1)];
        delta(x(i)+w*(y(i)-1),x(i)-1+w*(y(i)+1-1))=sqrt(2);
    end
    if x(i)>1 && y(i)>1
        neighbors = [neighbors,x(i)-1+w*(y(i)-1-1)];
        delta(x(i)+w*(y(i)-1),x(i)-1+w*(y(i)-1-1))=sqrt(2);
    end
    
    nodes{i}=struct('neighbors',neighbors);
end

% construct adjacency matrix
adjacency=zeros(J,J);
for i=1:J
    for j=1:length(nodes{i}.neighbors)
        adjacency(i,nodes{i}.neighbors(j))=1;
    end
end

ndeg = sum(reshape(tril(adjacency),[J^2,1])); % nb of degrees of liberty in adjacency matrix
graph = struct('J',J,'x',x,'y',y,'nodes',{nodes},'adjacency',adjacency,'delta_i',delta,'delta_tau',delta,'ndeg',ndeg);

end % End of function

%{
[graph]=create_triangle(w,h): creates a triangular graph structure
with width w and height h (each node is the center of an hexagon and
each node has 6 neighbors, horizontal and along the two diagonals)

Arguments:
- w: width of graph (ie. the max number of nodes along horizontal
dimension), must be an integer
- h: height of graph (ie. the max number of nodes along vertical
dimension), must be an odd integer

%}

function graph = create_triangle(w,h)

if mod(h,2)==0
    error('create_triangle(w,h): argument h must be an odd number.\n');
end

rows_outer = ceil(h/2);
rows_inner = ceil(h/2)-1;
J=w*rows_outer+(w-1)*rows_inner;

% Construct the network
nodes{J}=struct('neighbors',[]);
delta=zeros(J,J); % matrix of building costs
x=zeros(J,1);
y=zeros(J,1);
for j=1:rows_outer
    for i=1:w
        neighbors=[];
        
        id = i+w*(j-1)+(w-1)*(j-1);
        x(id) = i;
        y(id) = 1+(j-1)*2;
        
        if i<w
            neighbors = [neighbors,id+1];
            delta(id,id+1)=1;
            
            if j<rows_outer
                neighbors = [neighbors,id+w];
                delta(id,id+w)=1;
            end
            if j>1
                neighbors = [neighbors,id-(w-1)];
                delta(id,id-(w-1))=1;
            end
        end
        if i>1
            neighbors = [neighbors,id-1];
            delta(id,id-1)=1;
            if j<rows_outer
                neighbors = [neighbors,id+(w-1)];
                delta(id,id+(w-1))=1;
            end
            if j>1
                neighbors = [neighbors,id-w];
                delta(id,id-w)=1;
            end
        end
        
        nodes{id}=struct('neighbors',neighbors);
    end
end

for j=1:rows_inner
    for i=1:w-1
        neighbors=[];
        
        id = i+w*j+(w-1)*(j-1);
        x(id) = i+0.5;
        y(id) = 2+(j-1)*2;
        
        if i<w-1
            neighbors = [neighbors,id+1];
            delta(id,id+1)=1;
        end
        if i>1
            neighbors = [neighbors,id-1];
            delta(id,id-1)=1;
        end
        neighbors = [neighbors,id+w];
        delta(id,id+w)=1;
        neighbors = [neighbors,id-(w-1)];
        delta(id,id-(w-1))=1;
        neighbors = [neighbors,id+(w-1)];
        delta(id,id+w-1)=1;
        neighbors = [neighbors,id-w];
        delta(id,id-w)=1;
        
        nodes{id}=struct('neighbors',neighbors);
    end
end

% construct adjacency matrix
adjacency=zeros(J,J);
for i=1:J
    for j=1:length(nodes{i}.neighbors)
        adjacency(i,nodes{i}.neighbors(j))=1;
    end
end
ndeg = sum(reshape(tril(adjacency),[J^2,1]));

graph = struct('J',J,'x',x,'y',y,'nodes',{nodes},'adjacency',adjacency,'delta_i',delta,'delta_tau',delta,'ndeg',ndeg);

end % End of function

%{

[graph]=create_square(w,h): creates a square graph structure
with width w and height h (nodes have 4 neighbors in total, along
horizontal and vertical dimensions, NOT diagonals)

Arguments:
- w: width of graph (ie. the number of nodes along horizontal
dimension), must be an integer
- h: height of graph (ie. the number of nodes along vertical
dimension), must be an integer

%}

function graph = create_square(w,h)
J=w*h;
nodes{J}=struct('neighbors',[]);
delta=zeros(J,J); % matrix of building costs
x=zeros(J,1);
y=zeros(J,1);
for i=1:J
    neighbors=[];
    
    y(i)=floor((i-1)/w)+1;
    x(i)=i-w*(y(i)-1);
    
    
    if x(i)<w
        neighbors = [neighbors,x(i)+1+w*(y(i)-1)];
        delta(x(i)+w*(y(i)-1),x(i)+1+w*(y(i)-1))=1;
    end
    if x(i)>1
        neighbors = [neighbors,x(i)-1+w*(y(i)-1)];
        delta(x(i)+w*(y(i)-1),x(i)-1+w*(y(i)-1))=1;
    end
    if y(i)<h
        neighbors = [neighbors,x(i)+w*(y(i)+1-1)];
        delta(x(i)+w*(y(i)-1),x(i)+w*(y(i)+1-1))=1;
    end
    if y(i)>1
        neighbors = [neighbors,x(i)+w*(y(i)-1-1)];
        delta(x(i)+w*(y(i)-1),x(i)+w*(y(i)-1-1))=1;
    end
    
    nodes{i}=struct('neighbors',neighbors);
end

% construct adjacency matrix
adjacency=zeros(J,J);
for i=1:J
    for j=1:length(nodes{i}.neighbors)
        adjacency(i,nodes{i}.neighbors(j))=1;
    end
end

ndeg = sum(reshape(tril(adjacency),[J^2,1]));

graph = struct('J',J,'x',x,'y',y,'nodes',{nodes},'adjacency',adjacency,'delta_i',delta,'delta_tau',delta,'ndeg',ndeg);

end % End of function

%{

[graph]=create_custom(adjacency,x,y): creates a custom graph structure
with given adjacency matrix, x and y vectors of coordinates.

Arguments:
- adjacency: adjacency matrix
- x: vector of x coordinates of locations
- y: vector of y coordinates of locations

%}

function graph = create_custom(adjacency,x,y)

J=length(x);
nodes{J}=struct('neighbors',[]);

for i=1:J
    neighbors=find(adjacency(i,:)==1);
    
    nodes{i}=struct('neighbors',neighbors);
end

% compute degrees of freedom
ndeg = sum(reshape(tril(adjacency),[J^2,1]));

% compute matrix of bilateral distances for delta_tau and delta_i
delta=zeros(J,J); % matrix of distances
xx=x(:,ones(J,1));
yy=y(:,ones(J,1));
delta=sqrt((xx-xx').^2+(yy-yy').^2);
delta(~adjacency)=0;

graph = struct('J',J,'x',x,'y',y,'nodes',{nodes},'adjacency',adjacency,'delta_i',delta,'delta_tau',delta,'ndeg',ndeg);

end % End of function

