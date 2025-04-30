import bpy
import bmesh


def cube(**kwargs) -> bmesh.types.BMesh:
    bpy.ops.mesh.primitive_cube_add(**kwargs)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    return bm


def torus(**kwargs) -> bmesh.types.BMesh:
    bpy.ops.mesh.primitive_torus_add(**kwargs)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    return bm


def uv_sphere(**kwargs) -> bmesh.types.BMesh:
    bpy.ops.mesh.primitive_uv_sphere_add(**kwargs)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    return bm


def tetrahedron(**kwargs):
    bpy.ops.mesh.primitive_cone_add(vertices=1, **kwargs)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    return bm


CUBE = cube()
TORUS = torus()
UV_SPHERE = uv_sphere()
TETRAHEDRON = tetrahedron()

ALL_PRIMITIVES = [
    CUBE,
    TORUS,
    UV_SPHERE,
    TETRAHEDRON
]
