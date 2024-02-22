
using Plots
using Interpolations
using Colors

function plot_graph(param, graph, edges; kwargs...)
    options = retrieve_options_plot_graph(param, graph, edges; kwargs...)

    # INIT
    cla()
    set(gca(), "Unit", "normalized")
    hold(true)

    # Set margins
    if !options[:geography]
        margin = options[:margin]
    else
        margin = 0
    end

    xl = margin
    xu = 1 - margin
    yl = margin
    yu = 1 - margin

    # Resize graph to fit window
    graph_w = maximum(graph.x) - minimum(graph.x)
    graph_h = maximum(graph.y) - minimum(graph.y)

    vec_x = xl .+ (graph.x .- minimum(graph.x)) .* (xu - xl) ./ graph_w
    vec_y = yl .+ (graph.y .- minimum(graph.y)) .* (yu - yl) ./ graph_h

    # Define arrow
    if options[:arrow_style] == "long"
        arrow_vertices = [-0.75 0.5; 0 0; -0.75 -0.5; 1.25 0]
    elseif options[:arrow_style] == "thin"
        arrow_vertices = [-0.5 1; 0.5 0; -0.5 -1; 1 0]
    end

    arrow_z = complex.(arrow_vertices[:, 1], arrow_vertices[:, 2])
    arrow_scale = (1 - margin) / graph_w * 0.15 * options[:arrow_scale]

    # Define colormap for geography
    cres = 8
    theta = range(0, stop=1, length=cres)
    cmap = theta * options[:cmax] .+ (1 .- theta) * options[:cmin]
    colormap(cmap)

    # Define colormap for nodes
    node_color_red = LinearInterpolation(linspace(0, 1, size(options[:node_colormap], 1)), options[:node_colormap][:, 1])
    node_color_green = LinearInterpolation(linspace(0, 1, size(options[:node_colormap], 1)), options[:node_colormap][:, 2])
    node_color_blue = LinearInterpolation(linspace(0, 1, size(options[:node_colormap], 1)), options[:node_colormap][:, 3])
    node_color = x -> [node_color_red(x) node_color_green(x) node_color_blue(x)]

    # Define colormap for edges
    edge_color_red = LinearInterpolation(linspace(0, 1, size(options[:edge_colormap], 1)), options[:edge_colormap][:, 1])
    edge_color_green = LinearInterpolation(linspace(0, 1, size(options[:edge_colormap], 1)), options[:edge_colormap][:, 2])
    edge_color_blue = LinearInterpolation(linspace(0, 1, size(options[:edge_colormap], 1)), options[:edge_colormap][:, 3])
    edge_color = x -> [edge_color_red(x) edge_color_green(x) edge_color_blue(x)]

    # PLOT COLORMAP
    if !isempty(options[:map]) || options[:geography]
        if !isempty(options[:geography_struct])
            map = options[:geography_struct][:z]
        end
        if !isempty(options[:map])
            map = options[:map]
        end

        mapfunc = LinearInterpolation((vec_x, vec_y), map, extrapolation_bc=Line())
        xmap, ymap = ndgrid(range(minimum(vec_x), stop=maximum(vec_x), length=2*graph_w), range(minimum(vec_y), stop=maximum(vec_y), length=2*graph_h))
        fmap = mapfunc(xmap, ymap)

        contourf(xmap, ymap, fmap, linestyle="none")
    end

    # PLOT OBSTACLES
    if options[:obstacles]
        for i in 1:size(options[:geography_struct][:obstacles], 1)
            x1 = vec_x[options[:geography_struct][:obstacles][i, 1]]
            y1 = vec_y[options[:geography_struct][:obstacles][i, 1]]
            x2 = vec_x[options[:geography_struct][:obstacles][i, 2]]
            y2 = vec_y[options[:geography_struct][:obstacles][i, 2]]

            patchline([x1, x2], [y1, y2], edgecolor=options[:obstacle_color], linewidth=3, edgealpha=1)
        end
    end

    # PLOT MESH
    if options[:mesh]
        for i in 1:graph.J
            xi = vec_x[i]
            yi = vec_y[i]

            for j in graph.nodes[i].neighbors
                xj = vec_x[j]
                yj = vec_y[j]

                patchline([xi, xj], [yi, yj], edgecolor=options[:mesh_color], linestyle=options[:mesh_style], edgealpha=options[:mesh_transparency])
            end
        end
    end

    # PLOT EDGES
    if options[:edges]
        for i in 1:graph.J
            xi = vec_x[i]
            yi = vec_y[i]

            for j in graph.nodes[i].neighbors
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

                    if options[:transparency] == "on"
                        alpha = q
                        color = edge_color(1)
                    else
                        alpha = 1
                        color = edge_color(q)
                    end

                    if q > 0
                        patchline([xi, xj], [yi, yj], edgecolor=color, linewidth=width, edgealpha=alpha)
                    end

                    if options[:arrows] && q > 0.1
                        p = [xj - xi, yj - yi]
                        p = p / norm(p)
                        rot = complex(p[1], p[2])
                        h = fill((xi + xj) / 2 + width * arrow_scale * real(rot * arrow_z), (yi + yj) / 2 + width * arrow_scale * imag(rot * arrow_z), color)
                        set(h, edgecolor="none", facealpha=alpha)
                    end
                end
            end
        end
    end

    # PLOT NODES
    if options[:nodes]
        for i in 1:graph.J
            xi = vec_x[i]
            yi = vec_y[i]

            r = options[:sizes][i] * 0.075 * (1 - 2 * margin) / graph_w

            if options[:sizes][i] > param.minpop
                rectangle("Position" => [xi - r, yi - r, 2 * r, 2 * r], "Curvature" => [1, 1], "FaceColor" => node_color(options[:shades][i]), "EdgeColor" => options[:node_outercolor])
            end
        end
    end

    # PLOT
    if !options[:geography]
        axis([0, 1, 0, 1])
        set(gca(), "XTick" => [], "XTickLabel" => [], "YTick" => [], "YTickLabel" => [], "Box" => "on")
        hold(false)
        drawnow()
    else
        # PLOT GEOGRAPHY
        axis([0, 1, 0, 1])
        if maximum(fmap) - minimum(fmap) == 0
            caxis([0.99, 1])
        end
        set(gca(), "XTick" => [], "XTickLabel" => [], "YTick" => [], "YTickLabel" => [], "Units" => "normalized", "Position" => [0, 0, 1, 1])
        F = getframe(gcf())
        texturemap = frame2im(F)

        # Plot 3d graph
        cla()
        h = surf(xmap, ymap, fmap)
        set(h, "CData" => flipud(texturemap), "FaceColor" => "texturemap", "EdgeColor" => "none")
        xx = get(h, "XData")
        yy = get(h, "YData")
        set(h, "XData" => yy, "YData" => xx)
        view(options[:view])
        zmin = minimum(fmap)
        zmax = 2 * maximum(fmap)
        if zmin == zmax
            zmax = zmin + 1e-2
        end
        axis([minimum(xmap), maximum(xmap), minimum(ymap), maximum(ymap), zmin, zmax])
        axis("off")
        set(gca(), "position" => [0, 0, 1, 1.4], "Units" => "normalized")
    end
end

function retrieve_options_plot_graph(param, graph, edges; kwargs...)
    options = Dict(
        :mesh => get(kwargs, :mesh, "off") == "on",
        :arrows => get(kwargs, :arrows, "off") == "on",
        :edges => get(kwargs, :edges, "on") == "on",
        :nodes => get(kwargs, :nodes, "on") == "on",
        :map => get(kwargs, :map, []),
        :geography => get(kwargs, :geography, "off") == "on",
        :obstacles => get(kwargs, :obstacles, "off") == "on",
        :min_edge => get(kwargs, :min_edge, minimum(edges[edges .> 0])),
        :max_edge => get(kwargs, :max_edge, maximum(edges)),
        :min_edge_thickness => get(kwargs, :min_edge_thickness, 0.1),
        :max_edge_thickness => get(kwargs, :max_edge_thickness, 2),
        :sizes => get(kwargs, :sizes, ones(graph.J)),
        :shades => get(kwargs, :shades, zeros(graph.J)),
        :node_fgcolor => get(kwargs, :node_fgcolor, [1, 0, 0]),
        :node_bgcolor => get(kwargs, :node_bgcolor, [1, 1, 1]),
        :node_outercolor => get(kwargs, :node_outercolor, [0, 0, 0]),
        :node_colormap => get(kwargs, :node_colormap, []),
        :edge_color => get(kwargs, :edge_color, [0, 0.2, 0.5]),
        :edge_colormap => get(kwargs, :edge_colormap, []),
        :mesh_color => get(kwargs, :mesh_color, [0.9, 0.9, 0.9]),
        :mesh_style => get(kwargs, :mesh_style, "-"),
        :mesh_transparency => get(kwargs, :mesh_transparency, 1),
        :obstacle_color => get(kwargs, :obstacle_color, [0.4, 0.7, 1]),
        :cmax => get(kwargs, :cmax, [0.9, 0.95, 1]),
        :cmin => get(kwargs, :cmin, [0.4, 0.65, 0.6]),
        :margin => get(kwargs, :margin, 0.1),
        :arrow_scale => get(kwargs, :arrow_scale, 1),
        :arrow_style => get(kwargs, :arrow_style, "long"),
        :view => get(kwargs, :view, [30, 45]),
        :geography_struct => get(kwargs, :geography_struct, []),
        :transparency => get(kwargs, :transparency, "on"),
        :edge_scaling => get(kwargs, :edge_scaling, "on")
    )

    if isempty(options[:node_colormap])
        vec = range(0, stop=1, length=100)
        options[:node_colormap] = vec * options[:node_fgcolor] .+ (1 .- vec) * options[:node_bgcolor]
    end

    if isempty(options[:edge_colormap])
        vec = range(0, stop=1, length=100)
        options[:edge_colormap] = vec * options[:edge_color] .+ (1 .- vec) * [1, 1, 1]
    end

    return options
end


# Please note that this is a direct translation and might not work as expected due to differences in how Matlab and Julia handle certain operations. You might need to adjust the code to fit your specific needs.