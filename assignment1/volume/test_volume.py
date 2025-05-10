import pytest

import pytest

from .volume import mesh_volume
from data import primitives, meshes


# HINT: Add your own unit tests here
def test_fish():
    assert abs(mesh_volume(meshes.FISH) - meshes.FISH.calc_volume()) < 1e-6, "Fish FAILED"


def test_cube():
    assert abs(mesh_volume(primitives.CUBE) - primitives.CUBE.calc_volume()) < 1e-6, "Cube FAILED"


def test_torus():
    assert abs(mesh_volume(primitives.TORUS) - primitives.TORUS.calc_volume()) < 1e-6, "Torus FAILED"