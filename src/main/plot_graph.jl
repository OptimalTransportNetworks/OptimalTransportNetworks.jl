
# using Plots
# using Interpolations
# using Colors

function plot_graph(param, graph, edges; kwargs...)

    options = retrieve_options_plot_graph(param, graph, edges; kwargs...)

    # Empty plot
    pl = plot(grid=false, axis=([], false))

    # Set margins
    if !options[:geography]
        margin = options[:margin]
    else
        margin = 0
    end

    lb = margin
    ub = 1 - margin

    # Resize graph to fit window
    graph_w = maximum(graph.x) - minimum(graph.x)
    graph_h = maximum(graph.y) - minimum(graph.y)
    graph_ext = max(graph_w, graph_h)

    vec_x = lb .+ (graph.x .- minimum(graph.x)) .* (ub - lb) ./ graph_ext
    vec_y = lb .+ (graph.y .- minimum(graph.y)) .* (ub - lb) ./ graph_ext

    # PLOT COLORMAP
    if !options[:map] === nothing || options[:geography]
        if !isempty(options[:geography_struct])
            vec_map = options[:geography_struct][:z]
        end
        if !isempty(options[:map])
            vec_map = options[:map]
        end
        # Interpolate map onto grid
        # Maxbe need dierickx Spline2D
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
                 color = options[:map_colormap],
                 colorbar = false)
                 # clim=(minimum(fmap), maximum(fmap)))
    end

    # PLOT OBSTACLES
    if options[:obstacles]
        obstacles = options[:geography_struct][:obstacles]
        for i in 1:size(obstacles, 1)
            x1 = vec_x[obstacles[i, 1]]
            y1 = vec_y[obstacles[i, 1]]
            x2 = vec_x[obstacles[i, 2]]
            y2 = vec_y[obstacles[i, 2]]

            plot!(pl, [x1, x2], [y1, y2], 
                  linecolor = options[:obstacle_color], 
                  linewidth = options[:obstacle_thickness], 
                  linealpha = 1, label = nothing)
        end
    end

    # PLOT MESH
    if options[:mesh]
        for i in 1:graph.J
            xi = vec_x[i]
            yi = vec_y[i]

            for j in graph.nodes[i]
                xj = vec_x[j]
                yj = vec_y[j]

                plot!(pl, [xi, xj], [yi, yj], 
                      linecolor = options[:mesh_color], 
                      linestyle = options[:mesh_style], 
                      linealpha = options[:mesh_transparency], 
                      label = nothing)
            end
        end
    end

    # PLOT EDGES
    if options[:edges]
        if options[:arrows]
            # Define arrow
            if options[:arrow_style] == "long"
                arrow_vertices = [-0.75 0.5; 0 0; -0.75 -0.5; 1.25 0]
            elseif options[:arrow_style] == "thin"
                arrow_vertices = [-0.5 1; 0.5 0; -0.5 -1; 1 0]
            end

            arrow_z = complex.(arrow_vertices[:, 1], arrow_vertices[:, 2])
            arrow_scale = (1 - margin) / graph_ext * 0.15 * options[:arrow_scale]
        end

        edge_color = is_color(options[:edge_color]) ? options[:edge_color] : cgrad(options[:edge_color], [0.0, 1.0])

        for i in 1:graph.J
            xi = vec_x[i]
            yi = vec_y[i]

            for j in graph.nodes[i]
                xj = vec_x[j]
                yj = vec_y[j]

                if edges[i, j] >= options[:min_edge]
                    if options[:edge_scaling] == "off"
                        q = edges[i, j]
                    else
                        q = min((edges[i, j] - options[:min_edge]) / (options[:max_edge] - options[:min_edge]), 1)
                        if options[:max_edge] == options[:min_edge]
                            q = 1
                        end

                        if q < 0.001
                            q = 0
                        else
                            q = 0.1 + 0.9 * q
                        end
                    end

                    width = options[:min_edge_thickness] + q * (options[:max_edge_thickness] - options[:min_edge_thickness])

                    if options[:transparency] == "on" && is_color(options[:edge_color])
                        alpha = q
                        color = edge_color
                    else
                        alpha = 1
                        color = edge_color[q]
                    end

                    if q > 0
                        plot!(pl, [xi, xj], [yi, yj], 
                              linecolor = color, 
                              linewidth = width, 
                              linealpha = alpha, 
                              label = nothing)
                        # xlims!(plot_min, plot_max)
                        # ylims!(plot_min, plot_max)
                    end

                    if options[:arrows] && q > 0.1
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
    if options[:nodes]

        sizes = options[:sizes]
        sizes = (sizes .- minimum(sizes)) ./ (maximum(sizes) - minimum(sizes))

        has_shades = false
        color_grad = false
        node_color = options[:node_color]
        if options[:shades] !== nothing
            has_shades = true
            shades = options[:shades]
            shades = (shades .- minimum(shades)) ./ (maximum(shades) - minimum(shades))
            if !is_color(node_color)
                node_color = cgrad(node_color, [0.0, 1.0])
                color_grad = true
                has_shades = false
            end
        end

        r = sizes .* (options[:sizes_scale] * (1 - 2 * margin) / graph_ext) # * 0.075

        scatter!(pl, vec_x, vec_y, 
                 markercolor = color_grad ? node_color[shades] : node_color, 
                 markeralpha = has_shades ? shades : 1,
                 markerstrokewidth = options[:node_stroke_width],
                 markerstrokecolor = options[:node_stroke_color],
                 markersize = r, label = nothing)
    end

    return pl
end

function retrieve_options_plot_graph(param, graph, edges; kwargs...)
    options = Dict(
        :mesh => get(kwargs, :mesh, "off") == "on",
        :arrows => get(kwargs, :arrows, "off") == "on",
        :edges => get(kwargs, :edges, "on") == "on",
        :nodes => get(kwargs, :nodes, "on") == "on",
        :map => get(kwargs, :map, nothing),
        :map_color => get(kwargs, :map_color, :YlOrBr_4),

        :min_edge => get(kwargs, :min_edge, minimum(edges[edges .> 0])),
        :max_edge => get(kwargs, :max_edge, maximum(edges)),
        :min_edge_thickness => get(kwargs, :min_edge_thickness, 0.1),
        :max_edge_thickness => get(kwargs, :max_edge_thickness, 2),
        :sizes => get(kwargs, :sizes, ones(graph.J)),
        :sizes_scale => get(kwargs, :sizes_scale, 75),
        :shades => get(kwargs, :shades, nothing),

        :mesh_color => get(kwargs, :mesh_color, :grey90), 
        :mesh_style => get(kwargs, :mesh_style, :dash),
        :mesh_transparency => get(kwargs, :mesh_transparency, 1),

        :edge_color => get(kwargs, :edge_color, :blue), 
        :edge_scaling => get(kwargs, :edge_scaling, "on"),

        :geography => get(kwargs, :geography, "off") == "on",
        :geography_struct => get(kwargs, :geography_struct, nothing),
        :obstacles => get(kwargs, :obstacles, "off") == "on",
        :obstacle_color => get(kwargs, :obstacle_color, :black), 
        :obstackle_thickness => get(kwargs, :obstackle_thickness, 3),

        :node_color => get(kwargs, :node_color, :purple),
        :node_stroke_width => get(kwargs, :node_stroke_width, 0),
        :node_stroke_color => get(kwargs, :node_stroke_color, nothing),
   
        :margin => get(kwargs, :margin, 0.1),
        :arrow_scale => get(kwargs, :arrow_scale, 1),
        :arrow_style => get(kwargs, :arrow_style, "long"),
        :transparency => get(kwargs, :transparency, "on")
    )
    return options
end


# Please note that this is a direct translation and might not work as expected due to differences in how Matlab and Julia handle certain operations. You might need to adjust the code to fit your specific needs.

