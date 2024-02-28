
function gen_network_flows(Qin, graph, N)
    Qjkn = zeros(graph.J, graph.J, N)
    nodes = graph.nodes
    id = 1
    for i in 1:graph.J
        for j in 1:length(nodes[i])
            if nodes[i][j] > i
                Qjkn[i, nodes[i][j], :] = max.(Qin[id, :], 0)
                Qjkn[nodes[i][j], i, :] = max.(-Qin[id, :], 0)
                id += 1
            end
        end
    end
    return Qjkn
end