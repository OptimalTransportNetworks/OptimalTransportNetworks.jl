
"""
    find_node(graph, x, y)

Returns the index of the node closest to the coordinates (x,y) on the graph.

# Arguments
- `graph`: structure that contains the underlying graph (created by
create_map, create_rectangle or create_triangle functions)
- `x`: x coordinate on the graph between 1 and w
- `y`: y coordinate on the graph between 1 and h
"""
function find_node(graph, x, y)
    distance = (graph.x .- x).^2 + (graph.y .- y).^2
    _, id = findmin(distance)
    return id
end


# In this translation, I assumed that `graph` is a structure (or in Julia, a composite type) with fields `x` and `y` that are arrays of the same size. The `.-` operator is used for element-wise subtraction. The `findmin` function is used to find the minimum value in an array and its index, which is equivalent to the `min` function in Matlab. The underscore `_` is used to ignore the minimum value, as we are only interested in the index.