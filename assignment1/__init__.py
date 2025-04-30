import bpy

import inspect
import sys
from pathlib import Path

from . import boundaries, genus, volume, components, registration

ADDON_DIR = Path(__file__).resolve().parent.parent
if str(ADDON_DIR) not in sys.path:
    sys.path.append(str(ADDON_DIR))

classes = [
    genus.MeshGenus,
    boundaries.MeshBoundaryLoops,
    volume.MeshVolume,
    components.MeshConnectedComponents,
    registration.ObjectICPRegistration,
]


def register():
    for c in classes:
        try:
            bpy.utils.register_class(c)
        except AttributeError as e:
            print(
                f"Encountered an error while loading your {c.__name__} module, maybe you're missing some boilerplate?\n"
                f"\tError: '{e}'\n"
                f"\t(Take a look in '{inspect.getfile(c)}' to find out what's missing)"
            )


def unregister():
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            # If unregistering fails, it's probably just because registering failed!
            pass
