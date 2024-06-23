
# using Plots
# using Dierckx
# using Colors

"""
    plot_graph(graph, edges = nothing; kwargs...) -> Plots.Plot

Plot a graph visualization with various styling options.

# Arguments
- `graph::Dict`: The network graph (created with `create_graph()`)
- `edges::Matrix{Float64}=nothing`: Matrix of edge weights (J x J)

# Keyword Arguments
- `grid::Bool=false`: Show gridlines 
- `axis::Tuple=([], false)`: Axis ticks and labels (see Plots.jl docs, default disable axis)
- `margin::Real=0mm`: Margin around plot 
- `aspect_ratio::Symbol=:equal`: Plot aspect ratio (set to a real number (h/w) if not :equal)
- `height::Int=600`: Plot height in pixels (width is proportional to height / aspect_ratio, but also depends on the relative ranges of the x and y coordinates of the graph)
- `map::Vector=nothing`: Values mapped to graph for background heatmap
- `map_color::Symbol=:YlOrBr_4`: Colorscale for background heatmap
- `mesh::Bool=false`: Show mesh lines between nodes
- `mesh_color::Symbol=:grey90`: Color for mesh lines 
- `mesh_style::Symbol=:dash`: Linestyle for mesh lines
- `mesh_transparency::Real=1`: Opacity for mesh lines
- `edges::Bool=true`: Show edges between nodes
- `edge_color::Symbol=:blue`: Edge color or color gradient
- `edge_scaling::Bool=false`: Size edges based on raw values
- `edge_transparency::Union{Bool,Real}=true`: Transparency for edges
- `edge_min::Real`: Minimum edge value for scaling
- `edge_max::Real`: Maximum edge value for scaling  
- `edge_min_thickness::Real=0.1`: Minimum thickness for edges
- `edge_max_thickness::Real=2`: Maximum thickness for edges
- `arrows::Bool=false`: Show arrowheads on edges 
- `arrow_scale::Real=1`: Scaling factor for arrowheads
- `arrow_style::String="long"`: Style of arrowheads ("long" or "thin")
- `nodes::Bool=true`: Show nodes
- `node_sizes::Vector=ones(J)`: Sizes for nodes
- `node_sizes_scale::Real=75`: Overall scaling for node sizes
- `node_shades::Vector=nothing`: Shades mapped to nodes
- `node_color::Symbol=:purple`: Node color or color gradient 
- `node_stroke_width::Real=0`: Stroke width for node outlines
- `node_stroke_color::Symbol=nothing`: Stroke color for node outlines 
- `geography=nothing`: Dict or NamedTuple with geography data, see also `apply_geography()`
- `obstacles::Bool=false`: Show obstacles from geography
- `obstacle_color::Symbol=:black`: Color for obstacles
- `obstacle_thickness::Symbol=3`: Thickness for obstacles

# Examples
```julia
param = init_parameters(K = 10)
param, graph = create_graph(param)
param[:Zjn][51] = 10.0
results = optimal_network(param, graph)
plot_graph(graph, results[:Ijk])
```
"""
function plot_graph(graph, edges = nothing; kwargs...)

    graph = dict_to_namedtuple(graph)

    op = retrieve_options_plot_graph(graph, edges; kwargs...)

    # Resize graph to fit window
    graph_w = maximum(graph.x) - minimum(graph.x)
    graph_h = maximum(graph.y) - minimum(graph.y)
    graph_ext = max(graph_w, graph_h)
    
    plot_width = graph_w / graph_h * op.height
    if !(op.aspect_ratio isa Symbol)
        plot_width /= op.aspect_ratio
    end

    # Empty plot
    pl = plot(grid = op.grid, axis = op.axis, margin = op.margin, 
              aspect_ratio = op.aspect_ratio, size = (plot_width, op.height)) 

    # # Set margins
    # margin = op.margin
    # lb = margin
    # ub = 1 - margin
    # vec_x = lb .+ (graph.x .- minimum(graph.x)) .* (ub - lb) ./ graph_ext
    # vec_y = lb .+ (graph.y .- minimum(graph.y)) .* (ub - lb) ./ graph_ext

    vec_x = (graph.x .- minimum(graph.x)) ./ graph_ext
    vec_y = (graph.y .- minimum(graph.y)) ./ graph_ext

    # PLOT COLORMAP
    if op.map !== nothing || op.geography !== nothing
        if op.geography !== nothing
            vec_map = op.geography[:z]
        else
            vec_map = vec(op.map)
        end
        # Interpolate map onto grid
        # itp = interpolate((vec_x, vec_y), vec_map, Gridded(Linear()))
        spl = Spline2D(vec_x, vec_y, vec_map, s = 0.1)
        xmap = range(minimum(vec_x), stop=maximum(vec_x), length=2*length(vec_x))
        ymap = range(minimum(vec_y), stop=maximum(vec_y), length=2*length(vec_y))
        Xmap, Ymap = xmap' .* ones(length(ymap)), ymap .* ones(length(xmap))'
        Xmap, Ymap = Xmap[:], Ymap[:]
        fmap = evaluate(spl, Xmap, Ymap)
        # make fmap a matrix with same size as xmap and ymap
        fmap = reshape(fmap, length(xmap), length(ymap))

        # Plot heatmap 
        heatmap!(pl, xmap, ymap, fmap,
                 color = op.map_color,
                 colorbar = false)
                 # clim=(minimum(fmap), maximum(fmap)))
    end

    # PLOT OBSTACLES
    if op.obstacles && length(op.geography[:obstacles]) > 0
        obstacles = op.geography[:obstacles]
        for i in 1:size(obstacles, 1)
            x1 = vec_x[obstacles[i, 1]]
            y1 = vec_y[obstacles[i, 1]]
            x2 = vec_x[obstacles[i, 2]]
            y2 = vec_y[obstacles[i, 2]]

            plot!(pl, [x1, x2], [y1, y2], 
                  linecolor = op.obstacle_color, 
                  linewidth = op.obstacle_thickness, 
                  linealpha = 1, label = nothing)
        end
    end

    # PLOT MESH
    if op.mesh
        for i in 1:graph.J
            xi = vec_x[i]
            yi = vec_y[i]

            for j in graph.nodes[i]
                xj = vec_x[j]
                yj = vec_y[j]

                plot!(pl, [xi, xj], [yi, yj], 
                      linecolor = op.mesh_color, 
                      linestyle = op.mesh_style, 
                      linealpha = op.mesh_transparency, 
                      label = nothing)
            end
        end
    end

    # PLOT EDGES
    if op.edges && edges !== nothing
        if op.arrows
            # Define arrow
            if op.arrow_style == "long"
                arrow_vertices = [-0.75 0.5; 0 0; -0.75 -0.5; 1.25 0]
            elseif op.arrow_style == "thin"
                arrow_vertices = [-0.5 1; 0.5 0; -0.5 -1; 1 0]
            end

            arrow_z = complex.(arrow_vertices[:, 1], arrow_vertices[:, 2])
            arrow_scale = 1 / graph_ext * 0.15 * op.arrow_scale # (1 - margin) / graph_ext
        end

        edge_color_is_color = is_color(op.edge_color)
        edge_color = edge_color_is_color ? op.edge_color : cgrad(op.edge_color, [0.0, 1.0])
        var_transparency = op.edge_transparency === true
        alt_transparency = op.edge_transparency === false ? 1 : op.edge_transparency


        for i in 1:graph.J
            xi = vec_x[i]
            yi = vec_y[i]

            for j in graph.nodes[i]
                xj = vec_x[j]
                yj = vec_y[j]

                if edges[i, j] >= op.edge_min
                    if op.edge_scaling
                        q = edges[i, j]
                    else
                        q = min((edges[i, j] - op.edge_min) / (op.edge_max - op.edge_min), 1)
                        if op.edge_max == op.edge_min
                            q = 1
                        end

                        if q < 0.001
                            q = 0
                        else
                            q = 0.1 + 0.9 * q
                        end
                    end

                    width = op.edge_min_thickness + q * (op.edge_max_thickness - op.edge_min_thickness)
                    alpha = var_transparency ? q : alt_transparency
                    color = edge_color_is_color ? edge_color : edge_color[q]

                    if q > 0
                        plot!(pl, [xi, xj], [yi, yj], 
                              linecolor = color, 
                              linewidth = width, 
                              linealpha = alpha, 
                              label = nothing)
                        # xlims!(plot_min, plot_max)
                        # ylims!(plot_min, plot_max)
                    end

                    if op.arrows && q > 0.1
                        p = [xj - xi, yj - yi]
                        p = p ./ sqrt(sum(abs2, p))
                        rot = complex(p[1], p[2])
                        ar_x = (xi + xj) / 2 .+ width * arrow_scale * real(rot * arrow_z)
                        ar_y = (yi + yj) / 2 .+ width * arrow_scale * imag(rot * arrow_z)

                        plot!(pl, Shape(ar_x, ar_y), 
                              fill = color, 
                              linewidth = 0, 
                              opacity = alpha, 
                              label = nothing)
                    end
                end
            end
        end
    end

    # PLOT NODES
    if op.nodes

        has_sizes = false
        if op.node_sizes !== nothing
            has_sizes = true
            sizes = vec(op.node_sizes)
            if length(sizes) != length(vec_x)
                error("length(node_sizes) = $(length(sizes)) does not match number of nodes = $(length(vec_x))")
            end
            diff_sizes = maximum(sizes) - 0.0 # minimum(sizes)
            if diff_sizes > 0.0
                sizes = (sizes .- 0.0) ./ diff_sizes # minimum(sizes)
            end
            r = sizes .* (op.node_sizes_scale / graph_ext) # * 0.075 * (1 - 2 * margin) / graph_ext
        end

        color_grad = false
        has_shades = false
        node_color = op.node_color
        if op.node_shades !== nothing
            has_shades = true
            shades = vec(op.node_shades)
            if length(shades) != length(vec_x)
                error("length(node_shades) = $(length(shades)) does not match number of nodes = $(length(vec_x))")
            end
            diff_shades = maximum(shades) - minimum(shades)
            if diff_shades > 0.0
                shades = 0.05 .+ 0.95 .* (shades .- minimum(shades)) ./ diff_shades
            end
            if !is_color(node_color)
                node_color = cgrad(node_color, [0.0, 1.0])
                color_grad = true
            end
        end


        scatter!(pl, vec_x, vec_y, 
                 markercolor = color_grad ? node_color[shades] : node_color, 
                 markeralpha = has_shades && !color_grad ? shades : 1,
                 markerstrokewidth = op.node_stroke_width,
                 markerstrokecolor = op.node_stroke_color,
                 markersize = r, label = nothing)
    end

    return pl
end

function retrieve_options_plot_graph(graph, edges; kwargs...)

    options = (
        grid = get(kwargs, :grid, false),
        axis = get(kwargs, :axis, ([], false)),
        margin = get(kwargs, :margin, 0Plots.mm),
        aspect_ratio = get(kwargs, :aspect_ratio, :equal),
        height = get(kwargs, :height, 600),

        map = get(kwargs, :map, nothing),
        map_color = get(kwargs, :map_color, :YlOrBr_4),

        mesh = get(kwargs, :mesh, false),
        mesh_color = get(kwargs, :mesh_color, :grey90), 
        mesh_style = get(kwargs, :mesh_style, :dash),
        mesh_transparency = get(kwargs, :mesh_transparency, 1),

        edges = edges !== nothing,
        edge_color = get(kwargs, :edge_color, :blue), 
        edge_scaling = get(kwargs, :edge_scaling, false),
        edge_transparency = get(kwargs, :edge_transparency, true),
        edge_min = get(kwargs, :edge_min, edges !== nothing ? minimum(edges[edges .> 0]) : nothing),
        edge_max = get(kwargs, :edge_max, edges !== nothing ? maximum(edges) : nothing),
        edge_min_thickness = get(kwargs, :edge_min_thickness, 0.1),
        edge_max_thickness = get(kwargs, :edge_max_thickness, 2),

        arrows = get(kwargs, :arrows, false),
        arrow_scale = get(kwargs, :arrow_scale, 1),
        arrow_style = get(kwargs, :arrow_style, "long"),

        nodes = get(kwargs, :nodes, true),
        node_sizes = get(kwargs, :node_sizes, ones(graph.J)),
        node_sizes_scale = get(kwargs, :node_sizes_scale, 75),
        node_shades = get(kwargs, :node_shades, nothing),
        node_color = get(kwargs, :node_color, :purple),
        node_stroke_width = get(kwargs, :node_stroke_width, 0),
        node_stroke_color = get(kwargs, :node_stroke_color, nothing),

        geography = get(kwargs, :geography, nothing),
        obstacles = get(kwargs, :obstacles, false),
        obstacle_color = get(kwargs, :obstacle_color, :black), 
        obstacle_thickness = get(kwargs, :obstacle_thickness, 3)
    )

    unmatched_keys = setdiff(keys(kwargs), keys(options))
    # Check if non-supported keys
    if !isempty(unmatched_keys)
        # Print the error message indicating the unmatched keys
        @warn "Unsupported styling parameters:  $unmatched_keys"
    end

    return options
end
