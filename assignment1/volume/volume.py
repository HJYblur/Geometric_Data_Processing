import bmesh
import numpy as np


def signed_volume_tetrahedron(p1, p2, p3):
    return np.dot(p1, np.cross(p2, p3)) / 6.0


# !!! This function will be used for automatic grading, don't edit the signature !!!
def mesh_volume(mesh: bmesh.types.BMesh) -> float:
    """
    Finds the volume of the mesh.

    This function only needs to work correctly for meshes with one connected component and no holes.

    NOTE: the use of `mesh.calc_volume()` is not permitted inside this function!
          This task is intended to be completed manually, using methods discussed in class.

    :param mesh: The mesh to find the volume of.
    :return: The volume of the mesh as a float.
    """

    # (The math is a lot simpler if you know you only need to work with triangular faces)
    bmesh.ops.triangulate(mesh, faces=mesh.faces)

    # TODO: Return the volume of the mesh (without using Blender's built-in functionality)
    sum_volume = 0
    for tri in mesh.faces:
        sum_volume += signed_volume_tetrahedron(
            tri.verts[0].co, tri.verts[1].co, tri.verts[2].co
        )
    return sum_volume
