import contextlib
import os
import sys
import warnings

import bpy
from pathlib import Path

# Make sure local modules are available when running in blender
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

import util

# fixme: only needed because of the grading directory
warnings.filterwarnings('ignore', 'add-on missing manifest')


@util.run_in_blender
def main():
    # Get currently loaded repositories
    repo_dirs = set(Path(repo.directory) for repo in bpy.context.preferences.extensions.repos)

    # Add a new repo if necessary
    if PROJECT_DIR not in repo_dirs:
        print(f"Adding the directory '{PROJECT_DIR}' as an extension repository in blender!")
        bpy.ops.preferences.extension_repo_add(
            name='gdp',
            use_custom_directory=True,
            custom_directory=str(PROJECT_DIR),
            type='LOCAL',
        )

    # Retrieve the relevant repo
    relevant_repos = [repo for repo in bpy.context.preferences.extensions.repos if Path(repo.directory) == PROJECT_DIR]
    if len(relevant_repos) == 0:
        raise RuntimeError("Failed to add this directory as an extension repo!")
    if len(relevant_repos) > 1:
        raise RuntimeError(
            "This directory has been added as an extension repo more than once! "
            "This might not cause problems, but if it does you can delete the extra repo(s) in the blender UI."
        )
    gdp_repo = relevant_repos[0]
    assert gdp_repo.enabled

    # Finding all relevant addons
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        import addon_utils  # only works inside Blender
        gdp_addons = [addon for addon in addon_utils.modules() if PROJECT_DIR in Path(addon.__file__).parents]
    if len(gdp_addons) == 0:
        raise RuntimeError("The loaded extension repo contains no addons!")

    # Enable new addons
    print("Enabling addons:")
    for addon in gdp_addons:
        print(f" + '{addon.__name__}'")
        bpy.ops.preferences.addon_enable(module=addon.__name__)
        # todo: install dependencies

    # Save changes
    bpy.ops.wm.save_userpref()


if __name__ == '__main__':
    main()
