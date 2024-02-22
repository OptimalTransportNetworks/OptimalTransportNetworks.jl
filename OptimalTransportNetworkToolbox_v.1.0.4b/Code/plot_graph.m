%{
==============================================================
 OPTIMAL TRANSPORT NETWORKS IN SPATIAL EQUILIBRIUM
 by P. Fajgelbaum, E. Schaal, D. Henricot, C. Mantovani 2017-19
 ================================================ version 1.0.4

plot_graph ( param, graph, edges, varargin ): 
Plots the network with edge intensity given by edges (kappa,Q...).

Arguments:
- param: structure that contains the model's paramters
- graph: structure that contains the underlying graph (created by
create_map, create_rectangle or create_triangle functions)
- edges: a JxJ matrix containing the intensity/capacity of the edges to be plot
- varargin: (optional) various optional arguments as follows

'Mesh" = {'on','off'} if the underlying mesh is to be plotted, default is off
'Edges" = {'on','off'} if the edges are to be plotted, default is on
'Arrows" = {'on','off'} if arrows on the edges are to be plotted, default is off
'Nodes" = {'on','off'} if the cities are to be plotted, default is on
'Map' = [] or a vector of size J to plot a colormap with intensity given by the map. Default is empty (no map).
'Obstacles' = {'on','off'} if the geographical obstacles (rivers,...) are to be plotted
'Geography' = {'on','off'} if a geography is to be plotted, default is off,
on if GeographyStruct is specified

'MinEdge' = min value of the edge to be plotted
'MaxEdge' = max value of the edge to be plotted
'MaxEdgeThickness' = max thickness of edges (default 2)
'MinEdgeThickness' = min thickness of edges (default 0.1)
'Shades' = intensity of color for the node
'Sizes' = size of the node, 1 is normal size
'NodeFgColor' = foreground color for nodes with shade=1
'NodeBgColor' = background color for nodes with shade=0
'NodeOuterColor' = edge color for nodes
'NodeColorMap' = colormap for nodes (overrides Fg and Bg color)
'EdgeColor' = color for edges
'MeshColor' = color for underlying mesh
'MeshStyle' = {'-','--'...}
'MeshTransparency' = alpha value between 0 and 1
'ObstacleColor' = color for obstacles
'CMax' = color for the max of the heat map
'CMin' = color for the min of the heat map
'Margin' = margin around the graph.
'ArrowScale' = multiplier on the size of the arrow 1=same
'ArrowStyle' = {'long','thin'}
'GeographyStruct' = geography structure or [] if none
'View' = [AZ,EL] to choose the view in 3d with horizontal rotation AZ and vertical EL (in degree)
'Transparency' = {'on','off'} % whether transparency is to be used
'EdgeScaling' = {'on','off'} % whether the edges should be rescaled or just
use the absolute value, default on for rescaling
edges

-----------------------------------------------------------------------------------
REFERENCE: "Optimal Transport Networks in Spatial Equilibrium" (2019) by Pablo D.
Fajgelbaum and Edouard Schaal.

Copyright (c) 2017-2019, Pablo D. Fajgelbaum, Edouard Schaal
pfajgelbaum@ucla.edu, eschaal@crei.cat

This code is distributed under BSD-3 License. See LICENSE.txt for more information.
-----------------------------------------------------------------------------------
%}

function plot_graph( param,graph,edges,varargin )


% -------------------------
% Retrieve plotting options
options = retrieve_options(param,graph,edges,varargin);

% ----
% INIT
 
cla reset;
set(gca,'Unit','normalized');
hold on;

% Set margins

if options.geography==false
    margin = options.margin;
else
    margin=0;
end


xl = margin;
xu = 1-margin;
yl = margin;
yu = 1-margin;

% Resize graph to fit window

% vec_x=zeros(graph.J,1);
% vec_y=zeros(graph.J,1);
% for i=1:graph.J
%     vec_x(i)=graph.nodes{i}.x;
%     vec_y(i)=graph.nodes{i}.y;
% end

graph_w = max(graph.x)-min(graph.x);
graph_h = max(graph.y)-min(graph.y);

vec_x = xl + (graph.x-min(graph.x))*(xu-xl)/graph_w;
vec_y = yl + (graph.y-min(graph.y))*(yu-yl)/graph_h;

% Define arrow
if strcmp(options.arrow_style,'long')
    arrow_vertices = [-0.75,0.5;
        0,0;
        -0.75,-0.5;
        1.25,0];
end

if strcmp(options.arrow_style,'thin')
    arrow_vertices = [-0.5,1;
        .5,0;
        -0.5,-1;
        1,0];
end
              
arrow_z=complex(arrow_vertices(:,1),arrow_vertices(:,2));
arrow_scale=(1-margin)/graph_w * 0.15 * options.arrow_scale; % adjust size of the arrow to the average length of an edge

% Define colormap for geography

cres=8;
theta=linspace(0,1,cres)';
cmap=theta*options.cmax+(1-theta)*options.cmin;
colormap(gca,cmap);
if(max(options.map)-min(options.map)==0)
    caxis([.99 1]);
end

% Define colormap for nodes

node_color_red = griddedInterpolant(linspace(0,1,size(options.node_colormap,1)),options.node_colormap(:,1));
node_color_green = griddedInterpolant(linspace(0,1,size(options.node_colormap,1)),options.node_colormap(:,2));
node_color_blue = griddedInterpolant(linspace(0,1,size(options.node_colormap,1)),options.node_colormap(:,3));
node_color = @(x) [node_color_red(x) node_color_green(x) node_color_blue(x)];

% Define colormap for edges

edge_color_red = griddedInterpolant(linspace(0,1,size(options.edge_colormap,1)),options.edge_colormap(:,1));
edge_color_green = griddedInterpolant(linspace(0,1,size(options.edge_colormap,1)),options.edge_colormap(:,2));
edge_color_blue = griddedInterpolant(linspace(0,1,size(options.edge_colormap,1)),options.edge_colormap(:,3));
edge_color = @(x) [edge_color_red(x) edge_color_green(x) edge_color_blue(x)];



% =============
% PLOT COLORMAP

if ~isempty(options.map) || options.geography
    
    if ~isempty(options.geography_struct)
        map = options.geography_struct.z;
    end
    if ~isempty(options.map)
        map = options.map;
    end
    
    mapfunc=scatteredInterpolant(vec_x,vec_y,map,'natural','linear');
    [xmap,ymap]=ndgrid( linspace(min(vec_x),max(vec_x),2*graph_w), linspace(min(vec_y),max(vec_y),2*graph_h) );   
    fmap = mapfunc( xmap, ymap);      
    
%     contourf( xmap, ymap, fmap , 'LineStyle', 'none'); 
    h=pcolor(xmap, ymap, fmap);
    set(h,'EdgeColor','none');
    shading interp;

end


% ==============
% PLOT OBSTACLES
% ==============

if options.obstacles==true
    sz=size(options.geography_struct.obstacles);
    for i=1:sz(1)
        x1=vec_x(options.geography_struct.obstacles(i,1));
        y1=vec_y(options.geography_struct.obstacles(i,1));
        x2=vec_x(options.geography_struct.obstacles(i,2));
        y2=vec_y(options.geography_struct.obstacles(i,2));
%         y1=yl+(graph.nodes{options.geography_struct.obstacles(i,1)}.y-min(vec_y))*(yu-yl)/graph_h;
%         x2=xl+(graph.nodes{options.geography_struct.obstacles(i,2)}.x-min(vec_x))*(xu-xl)/graph_w;
%         y2=yl+(graph.nodes{options.geography_struct.obstacles(i,2)}.y-min(vec_y))*(yu-yl)/graph_h;
        
        patchline([x1;x2],[y1;y2],...
            'EdgeColor',options.obstacle_color,...
            'LineWidth',3,...
            'EdgeAlpha',1);
    end
end


% =========
% PLOT MESH

if options.mesh==true % Display the underlying mesh
    
    for i=1:graph.J
        xi=vec_x(i);
        yi=vec_y(i);
                
        for j=1:length(graph.nodes{i}.neighbors)
            xj=vec_x(graph.nodes{i}.neighbors(j));
            yj=vec_y(graph.nodes{i}.neighbors(j));
            
            patchline([xi;xj],[yi;yj],'EdgeColor',options.mesh_color,'LineStyle',options.mesh_style,'EdgeAlpha',options.mesh_transparency);
%             line([xi;xj],[yi;yj],'Color',options.mesh_color);
        end
    end

end

% ==========
% PLOT EDGES

if options.edges==true

    for i=1:graph.J
        xi=vec_x(i);
        yi=vec_y(i);
        
        for j=1:length(graph.nodes{i}.neighbors)
            xj=vec_x(graph.nodes{i}.neighbors(j));
            yj=vec_y(graph.nodes{i}.neighbors(j));
            
            if edges(i,graph.nodes{i}.neighbors(j)) >= options.min_edge % only display the main edges (due to numerical errors, some edges are <1e-16 and some important edges can be < min(C)
                                                
                % adjust width and color
                if strcmp(options.edge_scaling,'off')
                    q=edges( i,graph.nodes{i}.neighbors(j) );
                else
                    q=min((edges( i,graph.nodes{i}.neighbors(j) )-options.min_edge)/(options.max_edge - options.min_edge ),1);
                    if options.max_edge==options.min_edge
                        q=1;
                    end
                    
                    if q<.001 % don't even plot if less 
                        q=0;
                    else
                        q=.1+.9*q; % just to make sure the lines aren't too small
                    end
                end
%                 width=ceil( (1-2*margin)/graph_w*10*( q-0.1 ) )+1;
%                 width=(options.max_edge_thickness-0.1)*max(q-0.1,0)+0.1;
                width=options.min_edge_thickness + q*(options.max_edge_thickness-options.min_edge_thickness);
                                
                if strcmp(options.transparency,'on')
                    alpha=q;
                    color=edge_color(1);
                else
                    alpha=1;
                    color=edge_color(q);
                end
                
                % plot
                if q>0
                    patchline([xi;xj],[yi;yj],...
                        'EdgeColor',color,...
                        'LineWidth',width,...
                        'EdgeAlpha',alpha);
                end
                
                if options.arrows==true %&& q>0.1 % only plot arrows for large edges, otherwise mess
                    p = [xj-xi yj-yi];
                    p = p/norm(p);
                    rot=complex(p(1),p(2));
                    h=fill((xi+xj)/2+width*arrow_scale*real(rot*arrow_z),(yi+yj)/2+width*arrow_scale*imag(rot*arrow_z),color);
                    set(h,'EdgeColor','none','FaceAlpha',alpha);
                end
            end
        end
    end

end


% ==========
% PLOT NODES

if options.nodes==true    
    
    for i=1:graph.J
        xi=vec_x(i);
        yi=vec_y(i);
                        
        % sizes
        r=options.sizes(i)*0.075*(1-2*margin)/graph_w; % adjust size to average length of edge
        
        if options.sizes(i)>param.minpop
            rectangle('Position',[xi-r,yi-r,2*r,2*r],...
                'Curvature',[1,1],...
                'FaceColor',node_color(options.shades(i)),...
                'EdgeColor',options.node_outercolor);
        end        
    end

end

% ====
% PLOT

if ~options.geography
    % ------------------------------------------
    % PLOT REGULAR GRAPH IN 2D (then plot in 3D)
    
    axis([0 1 0 1]);
    set(gca,'XTick',[],'XTickLabel',[],'YTick',[],'YTickLabel',[],'Box','on');
    hold off;
    drawnow;
else
    % --------------
    % PLOT GEOGRAPHY
    
    % Format and save
    
    axis([0 1 0 1]);
    if(max(fmap)-min(fmap)==0)
        caxis([.99 1]);
    end
    set(gca,'XTick',[],'XTickLabel',[],'YTick',[],'YTickLabel',[],'Units','normalized','Position',[0 0 1 1]);
    F = getframe(gcf);
    texturemap=frame2im(F);
    
    % Plot 3d graph
    
    cla reset;
    h=surf(xmap,ymap,fmap);
    set(h,'CData',flipud(texturemap),'FaceColor','texturemap','EdgeColor','none');
    xx=get(h,'XData');
    yy=get(h,'YData');
    set(h,'XData',yy,'YData',xx);
    view(options.view);
    zmin=min(fmap(:));
    zmax=2*max(fmap(:));
    if zmin==zmax
        zmax=zmin+1e-2;
    end
    axis([min(xmap(:)),max(xmap(:)),min(ymap(:)),max(ymap(:)),zmin,zmax]);
    axis off;
    set(gca,'position',[0,0,1,1.4],'Units','normalized'); % scale up along the vertical axis for better visibility
    
end

% ----------------------------
% RETRIEVE OPTIONS FROM INPUTS

function options = retrieve_options(param,graph,edges,args)
p=inputParser;

% ====================================
% DEFINE STANDARD VALUES OF PARAMETERS

default_mesh='off';
default_arrows='off';
default_edges='on';
default_nodes='on';
default_min_edge=min(edges(edges>0));
default_max_edge=max(edges(:));
default_min_edge_thickness=0.1;
default_max_edge_thickness=2;
default_map=[];
default_obstacles='off';
default_geography='off';

default_sizes=ones(graph.J,1);
default_shade=zeros(graph.J,1);
default_nodefgcolor=[1 0 0];
default_nodebgcolor=[1 1 1];
default_nodeoutercolor=[0 0 0];
default_nodecolormap=[];
default_edgecolor=[0,.2,.5];
default_edgecolormap=[];
default_meshcolor=[.9,.9,.9];
default_meshstyle='-';
default_meshtransparency=1;
default_obstaclecolor=[.4,.7,1];
default_cmax=[.9 .95 1];
default_cmin=[0.4 0.65 0.6];
default_margin=0.1;
default_arrowscale=1;
default_arrowstyle='long';
default_geographystruct=[];
default_view=[30,45];
default_transparency='on';
default_edgescaling='on';

validSwitch = {'on','off'};
checkSwitch = @(x) any(validatestring(x,validSwitch));

% =============
% DEFINE INPUTS

addParameter(p,'Mesh',default_mesh,checkSwitch);
addParameter(p,'Arrows',default_arrows,checkSwitch);
addParameter(p,'Edges',default_edges,checkSwitch);
addParameter(p,'Nodes',default_nodes,checkSwitch);
addParameter(p,'Map',default_map,@(x) (isempty(x) | isequal(size(x),[graph.J 1])) );
addParameter(p,'Geography',default_geography,checkSwitch);
addParameter(p,'Obstacles',default_obstacles,checkSwitch);
addParameter(p,'MinEdge',default_min_edge,@isnumeric);
addParameter(p,'MaxEdge',default_max_edge,@isnumeric);
addParameter(p,'MinEdgeThickness',default_min_edge_thickness,@isnumeric);
addParameter(p,'MaxEdgeThickness',default_max_edge_thickness,@isnumeric);
addParameter(p,'Sizes',default_sizes,@(x) isequal(size(x),[graph.J 1]));
addParameter(p,'Shades',default_shade,@(x) isequal(size(x),[graph.J 1]));
addParameter(p,'NodeFgColor',default_nodefgcolor,@(x) isequal(size(x),[1 3]));
addParameter(p,'NodeBgColor',default_nodebgcolor,@(x) isequal(size(x),[1 3]));
addParameter(p,'NodeOuterColor',default_nodeoutercolor,@(x) isequal(size(x),[1 3]));
addParameter(p,'NodeColormap',default_nodecolormap,@(x) (isequal(size(x,2),3) | isempty(x)));
addParameter(p,'EdgeColor',default_edgecolor,@(x) isequal(size(x),[1 3]));
addParameter(p,'EdgeColormap',default_edgecolormap,@(x) (isequal(size(x,2),3) | isempty(x)));
addParameter(p,'MeshColor',default_meshcolor,@(x) isequal(size(x),[1 3]));
addParameter(p,'MeshStyle',default_meshstyle,@(x) any(validatestring(x,{'-','--',':'})));
addParameter(p,'MeshTransparency',default_meshtransparency,@isnumeric);
addParameter(p,'ObstacleColor',default_obstaclecolor,@(x) isequal(size(x),[1 3]));
addParameter(p,'CMax',default_cmax,@(x) isequal(size(x),[1 3]));
addParameter(p,'CMin',default_cmin,@(x) isequal(size(x),[1 3]));
addParameter(p,'Margin',default_margin,@isnumeric);
addParameter(p,'ArrowScale',default_arrowscale,@isnumeric);
addParameter(p,'ArrowStyle',default_arrowstyle,@(x) any(validatestring(x,{'long','thin'})));
addParameter(p,'View',default_view,@(x) isequal(size(x),[1 2]));
addParameter(p,'GeographyStruct',default_geographystruct);
addParameter(p,'Transparency',default_transparency,checkSwitch);
addParameter(p,'EdgeScaling',default_edgescaling,checkSwitch);

% Parse inputs
parse(p,args{:});

% ==============
% RETURN OPTIONS

options.mesh = strcmp(p.Results.Mesh,'on');
options.arrows = strcmp(p.Results.Arrows,'on');
options.edges = strcmp(p.Results.Edges,'on');
options.nodes = strcmp(p.Results.Nodes,'on');
options.map = p.Results.Map;
if any(strcmp('Geography',p.UsingDefaults))
    options.geography = ~isempty(p.Results.GeographyStruct); % geography is on if GeographyStruct specified
else
    options.geography = strcmp(p.Results.Geography,'on'); % but force decision if specified
end

if any(strcmp('Obstacles',p.UsingDefaults)) 
    options.obstacles = options.geography; % obstacles are on if a geography is specified
else
    options.obstacles = strcmp(p.Results.Obstacles,'on'); % but force decision if specified
end

options.min_edge = p.Results.MinEdge;
options.max_edge = p.Results.MaxEdge;
if options.max_edge<options.min_edge
    error('%s.m: MaxEdge less than MinEdge.',mfilename);
end

options.max_edge_thickness = p.Results.MaxEdgeThickness;
options.min_edge_thickness = p.Results.MinEdgeThickness;
options.sizes = p.Results.Sizes;
options.shades = p.Results.Shades;
options.node_fgcolor = p.Results.NodeFgColor;
options.node_bgcolor = p.Results.NodeBgColor;
options.node_outercolor = p.Results.NodeOuterColor;
options.node_colormap = p.Results.NodeColormap;
if isempty(options.node_colormap)
    vec=linspace(0,1,100)';
    options.node_colormap = vec*options.node_fgcolor + (1-vec)*options.node_bgcolor;
end

options.edge_color = p.Results.EdgeColor;
options.edge_colormap = p.Results.EdgeColormap;
if isempty(options.edge_colormap)
    vec=linspace(0,1,100)';
    options.edge_colormap = vec*options.edge_color + (1-vec)*[1 1 1];
end

options.mesh_color = p.Results.MeshColor;
options.mesh_style = p.Results.MeshStyle;
options.mesh_transparency = max(min(p.Results.MeshTransparency,1),0);
options.obstacle_color = p.Results.ObstacleColor;
options.cmax = p.Results.CMax;
options.cmin = p.Results.CMin;
options.margin = p.Results.Margin;
options.arrow_scale = p.Results.ArrowScale;
options.arrow_style = p.Results.ArrowStyle;
options.view = p.Results.View;
options.geography_struct = p.Results.GeographyStruct;
options.transparency = p.Results.Transparency;
options.edge_scaling = p.Results.EdgeScaling;

