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
    res_dict = {}
    num_comp = 0

    def bfs(init_v: bmesh.types.BMVert):
        vqueue = [init_v]
        while vqueue:
            v = vqueue.pop(0)
            for e in v.link_edges:
                for vert in e.verts:
                    if vert.index not in res_dict:
                        res_dict[vert.index] = num_comp
                        vqueue.append(vert)

    for v in mesh.verts:
        if v.index not in res_dict:
            res_dict[v.index] = num_comp
            bfs(v)
            num_comp += 1

    res = [set() for _ in range(num_comp)]
    for v in mesh.verts:
        res[res_dict[v.index]].add(v)

    return res
