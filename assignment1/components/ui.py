import bpy
import bmesh

from .connected_components import mesh_connected_components


class MeshConnectedComponents(bpy.types.Panel):
    # TODO: Add bpy boilerplate (ID name, label, category, etc.)
    bl_idname = "VIEW3D_PT_MeshConnectedComponents"
    bl_label = "Mesh Connected Components"

    bl_category = "Assignment 1"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    # TODO: Add a draw method which uses the function in connected_components.py to show the number of components

    def draw(self, context):
        if context.active_object is None:
            self.layout.label(text="Select a mesh")
            return

        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)

        num_comp = len(mesh_connected_components(bm))
        self.layout.label(text=f"connected components: {num_comp}")
    # BONUS: Show the number of points in each connected component
    # BONUS: Include a dropdown which allows the user to select the vertices of each connected component
