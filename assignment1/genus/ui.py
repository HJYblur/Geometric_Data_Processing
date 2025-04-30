import bpy
import bmesh
from .genus import mesh_genus


class MeshGenus(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_MeshGenus"
    bl_label = "Mesh Genus"

    bl_category = "Assignment 1"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        if context.active_object is None:
            self.layout.label(text="Select a mesh")
            return

        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)

        self.layout.label(text=f'Genus: {mesh_genus(bm)}')
