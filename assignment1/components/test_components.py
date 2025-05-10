import pytest
from .connected_components import mesh_connected_components
from data import primitives, meshes


# HINT: Add unit tests for simple primitives (one connected component)

# HINT: Add unit tests for multiple connected components
#       (You can create examples in the Blender UI by selecting multiple objects and pressing CTRl-J)

# HINT: Add automated testing of larger component counts, automatically generate examples by combining primitives

def verify_components_properties(mesh, components):
    all_vertices = set()
    total_vertices = 0

    for component in components:
        if any(vertex in all_vertices for vertex in component):
            return False
        all_vertices.update(component)
        total_vertices += len(component)

    return total_vertices == len(mesh.verts)

def test_cube():
    components = mesh_connected_components(primitives.CUBE)
    assert len(components) == 1, "connected components CUBE failed"
    assert verify_components_properties(primitives.CUBE, components), "connected components CUBE failed"

def test_two_tori():
    components = mesh_connected_components(meshes.TWO_TORI)
    assert len(components) == 2, "connected components TWO_TORI failed"
    assert verify_components_properties(meshes.TWO_TORI, components), "connected components TWO_TORI failed"

def test_double_torus():
    components = mesh_connected_components(meshes.DOUBLE_TORUS)
    assert len(components) == 1, "connected components DOUBLE_TORUS failed"
    assert verify_components_properties(meshes.DOUBLE_TORUS, components), "connected components DOUBLE_TORUS failed"


