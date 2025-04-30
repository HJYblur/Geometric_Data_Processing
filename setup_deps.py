import os
import subprocess
import tomlkit

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
WHEELS_DIR = PROJECT_DIR / '.wheels'
BLENDER_PYTHON_VERSION = '3.11'  # Latest blender uses 3.11!
BLENDER_MANIFESTS = list(PROJECT_DIR.glob('**/blender_manifest.toml'))


def main():
    print("Downloading dependencies:")
    with open(PROJECT_DIR / 'pyproject.toml', 'rb') as pyproject:
        pyproject = tomlkit.load(pyproject)
        for dep in pyproject['project']['dependencies']:
            # Install the dependency
            print(f" + {dep}")
            subprocess.run(
                [
                    'pip', 'download', dep,
                    '--dest', str(WHEELS_DIR),
                    '--only-binary=:all:',
                    f'--python-version={BLENDER_PYTHON_VERSION}',
                ],
                check=True,
                capture_output=True,
            )

    print("Writing dependencies to 'blender_manifest.toml'")
    wheels = list(f'../.wheels/{whl.name}' for whl in WHEELS_DIR.iterdir())
    for manifest_file in BLENDER_MANIFESTS:
        manifest = tomlkit.load(open(manifest_file, 'rb'))
        manifest['wheels'] = wheels
        manifest['wheels'].multiline(True)
        open(manifest_file, 'wb').write(tomlkit.dumps(manifest).encode('utf-8'))


if __name__ == '__main__':
    main()
