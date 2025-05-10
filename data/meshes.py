import os
import bpy, bmesh

MESH_DIR = os.path.join(os.path.dirname(__file__), 'meshes')


def load(mesh_name: str) -> bmesh.types.BMesh:
    # todo: maybe this should append .obj automatically? It only loads obj files anyway
    mesh_path = os.path.join(MESH_DIR, mesh_name)
    bpy.ops.wm.obj_import(filepath=mesh_path)
    obj_object = bpy.context.selected_objects[-1]
    bm = bmesh.new()
    bm.from_mesh(obj_object.data)
    return bm


BAGEL_CUT_TORUS = load('bagel-cut-torus.obj')
DOUBLE_TORUS = load('double-torus.obj')
HALF_BAGEL_CUT_TORUS = load('half-bagel-cut-torus.obj')
HALF_TORUS = load('half-torus.obj')
TWO_TORI = load('two-tori.obj')
DOUBLE_TORUS_WITH_HOLE = load('double-torus-with-hole.obj')
FISH = load('fish.obj')