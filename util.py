import os
import shutil
import subprocess
import sys
from functools import wraps


def running_in_blender():
    try:
        import bpy.app

        return bool(bpy.app.binary_path)
    except:
        return False


def run_in_blender(func):
    """
    Decorator that ensures the function runs inside Blender.
    If not currently in Blender, re-invokes the script with blender --background.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Short-circuit if we're already running in Blender
        if running_in_blender():
            return func(*args, **kwargs)

        print("Not running in Blender. Restarting with Blender...")

        # Determine the blender executable
        blender = shutil.which("/Applications/Blender.app/Contents/MacOS/Blender")
        if not blender:
            raise FileNotFoundError(
                "Unable to find the appropriate command to run blender! "
                "If blender requires a different command from 'blender' on your computer, try modifying this script."
            )

        script_path = os.path.abspath(sys.argv[0])
        script_args = sys.argv[1:] if len(sys.argv) > 1 else []
        result = subprocess.run(
            [
                blender,
                "--background",  # Launch without a UI
                "--python",
                script_path,  # Run the current script with blender's interpreter
                "--",  # None of the following arguments should be interpreted as blender args
                *script_args,  # Forward any args passed to this script
            ],
            check=True,
        )
        return result.returncode

    return wrapper
