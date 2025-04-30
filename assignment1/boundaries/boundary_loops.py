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

    return list(set())
