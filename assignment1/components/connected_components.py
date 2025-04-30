import bmesh
from typing import Optional, List, Set




# !!! This function will be used for automatic grading, don't edit the signature !!!
def mesh_connected_components(mesh: bmesh.types.BMesh) -> List[Set[bmesh.types.BMVert]]:
    """
    Finds the connected components of the mesh.

    Each connected component is represented by a `set()` of vertices.
    Each vertex of the mesh should appear in exactly one connected component.

    HINT: the union of the sets should be empty (no shared vertices),
          and the total number of vertices in all sets should add up to the number of vertices in the mesh.
          
    NOTE: the use of dedicated functions like those in `scipy` is not permitted inside this function!
          This task is intended to be completed manually, using methods discussed in class.

    :param mesh: The mesh to find the connected components of.
    :return: A list of connected components, each of which is a set of `BMVert`s.
    """
    # TODO: Find the connected components of the mesh

    return list(set())
