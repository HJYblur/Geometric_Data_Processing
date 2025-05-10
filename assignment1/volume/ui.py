import bpy
import bmesh

from .volume import mesh_volume


class MeshVolume(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_MeshVolume"
    bl_label = "Mesh Volume"

    bl_category = "Assignment 1"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        # TODO: Check that the user has selected a mesh
        if context.active_object is None:
            self.layout.label(text="Select a mesh")
            return

        # TODO: Obtain a BMesh from the selected mesh
        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)

        # TODO: Apply the world transformation to the BMesh (so that scaling with hotkey 's' affects volume)
        transformed_bm = bm.copy()

        matrix_world = context.active_object.matrix_world
        for v in transformed_bm.verts:
            v.co = matrix_world @ v.co

        # TODO: Show the computed volume using a label
        volume = mesh_volume(transformed_bm)
        self.layout.label(text=f"Volume: {volume:.3f}")
