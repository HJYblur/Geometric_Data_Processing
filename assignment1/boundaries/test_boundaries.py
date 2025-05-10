import bpy
import bmesh
from .boundary_loops import mesh_boundary_loops
from data import primitives, meshes


def verify_boundary_loops(mesh, boundary_loops):

    all_edges = set()
    for loop in boundary_loops:
        for edge in loop:
            if edge in all_edges:
                return False
            all_edges.add(edge)


    for edge in all_edges:
        if not edge.is_boundary:
            return False

    for edge in mesh.edges:
        if edge.is_boundary and edge not in all_edges:
            return False

    return True


def test_cube():
    boundary_loops = mesh_boundary_loops(primitives.CUBE)
    assert len(boundary_loops) == 0, "CUBE boundary loops FAILED"
    assert verify_boundary_loops(primitives.CUBE, boundary_loops), "CUBE boundary loops FAILED"


def test_half_torus():
    boundary_loops = mesh_boundary_loops(meshes.HALF_TORUS)
    assert len(boundary_loops) == 2, "half_torus boundary loops FAILED"
    assert verify_boundary_loops(meshes.HALF_TORUS, boundary_loops), "half_torus boundary loops FAILED"


def test_fish():
    boundary_loops = mesh_boundary_loops(meshes.FISH)
    assert verify_boundary_loops(meshes.FISH, boundary_loops), "FISH boundary loops FAILED"
