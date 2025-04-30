import bpy
import bmesh

from .connected_components import mesh_connected_components


class MeshConnectedComponents(bpy.types.Panel):
    # TODO: Add bpy boilerplate (ID name, label, category, etc.)
    # TODO: Add a draw method which uses the function in connected_components.py to show the number of components
    # BONUS: Show the number of points in each connected component
    # BONUS: Include a dropdown which allows the user to select the vertices of each connected component
    pass
