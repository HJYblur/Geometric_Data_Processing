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

        # TODO: Obtain a BMesh from the selected mesh

        # TODO: Apply the world transformation to the BMesh (so that scaling with hotkey 's' affects volume)

        # TODO: Show the computed volume using a label

        pass
