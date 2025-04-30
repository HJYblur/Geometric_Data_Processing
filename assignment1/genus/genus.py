import bmesh
from ..boundaries import mesh_boundary_loops


# !!! This function will be used for automatic grading, don't edit the signature !!!
def mesh_genus(mesh: bmesh.types.BMesh) -> int:
    """
    Finds the genus of a mesh.

    Assuming the mesh represents a manifold surface (no boundary loops),
    this function should return the number of "holes" in the surface.
    e.g. A sphere or a cube has genus 0, a toroid has genus 1, a double toroid has genus 2.

    BONUS: Account for non-manifold meshes by treating each boundary loop (hole) as one face.

    :param mesh: The mesh to find the genus of.
    :return: The genus of the mesh, as an integer.
    """
    # TODO: This should return the genus of the mesh
    return 0
