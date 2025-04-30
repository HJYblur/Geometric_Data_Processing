import shutil
import subprocess

if __name__ == '__main__':
    blender_bin = shutil.which("blender")
    if blender_bin:
        print("Found:", blender_bin)
    else:
        print("Unable to find blender!")

    result = subprocess.run(
        [
            blender_bin,
        ],
        check=True,
    )
