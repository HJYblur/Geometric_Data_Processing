import bpy
import bmesh




# !!! This function will be used for automatic grading, don't edit the signature !!!
def mesh_boundary_loops(mesh: bmesh.types.BMesh) -> list[set[bmesh.types.BMEdge]]:
    """
    Finds the boundary loops of a BMesh.

    Each boundary loop is represented by a `set()` of the edges which make it up.
    Each edge appears in at most one boundary loop.
    Non-boundary edges will not appear in any boundary loops
    HINT: how do you check if an edge is on a boundary?

    :param mesh: The mesh to find the boundary loops of.
    :return: A list of boundary loops, each of which is a set of `BMEdge`s.
    """
    # TODO: Find the boundary loops of the mesh
    visit = {}
    def dfs(edge: bmesh.types.BMEdge) -> set[bmesh.types.BMEdge]:
        visit[edge.index] = True
        cur_boundary = set([edge])
        for v in edge.verts:
            for e in v.link_edges:
                if e.index not in visit and e.is_boundary:
                    cur_boundary.update(dfs(e))
        return cur_boundary
    res = []
    for edge in mesh.edges:
        if edge.index not in visit and edge.is_boundary:
            res.append(dfs(edge))
    return res
