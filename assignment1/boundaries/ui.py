import bpy
import bmesh
from .boundary_loops import mesh_boundary_loops


class MeshBoundaryLoops(bpy.types.Panel):
    # TODO: Implement a panel which shows the number of boundary loops in the active mesh
    bl_idname = "VIEW3D_PT_MeshBoundaryLoops"
    bl_label = "Mesh Boundary Loops"

    bl_category = "Assignment 1"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        if context.active_object is None:
            self.layout.label(text="Select a mesh")
            return

        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)

        num_comp = len(mesh_boundary_loops(bm))
        self.layout.label(text=f"boundary loop: {num_comp}")
    pass
