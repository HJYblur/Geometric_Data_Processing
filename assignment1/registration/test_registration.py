import random
import pytest
from .iterative_closest_point import *
from data import primitives, meshes
import mathutils

NUM_TESTS = 10


def assert_similar_transformations(a: mathutils.Matrix, b: mathutils.Matrix):
    translation_error = a.to_translation() - b.to_translation()
    assert translation_error.magnitude == pytest.approx(
        0, abs=1e-3
    ), "Translation error should be low"

    rotation_error = a.to_quaternion() - b.to_quaternion()
    assert rotation_error.magnitude == pytest.approx(
        0, abs=1e-3
    ), "Rotation error should be low"

    scaling_error = a.to_scale() - b.to_scale()
    assert scaling_error.magnitude == pytest.approx(
        0, abs=1e-3
    ), "Scaling should be zero"


def test_cube():
    translation = mathutils.Matrix.Translation(
        [
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
        ]
    )
    rotation = (
        mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "X")
        @ mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "Y")
        @ mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "Z")
    )
    transformation = translation @ rotation

    source, destination = primitives.CUBE, primitives.CUBE.copy()
    destination.transform(
        transformation
    )  # Move the destination, so we don't need to invert the transform

    registration_transformations = iterative_closest_point_registration(
        source,
        destination,
        k=2.5,
        num_points=4096,
        iterations=100,
        epsilon=0.0005,
        distance_metric="POINT_TO_PLANE",
        match_method="kdtree",
        p_norm="1",  # 1, 2, inf
    )
    print(len(registration_transformations))
    # The function should have converged
    # assert len(registration_transformations) < 100

    # Check that we found the right matrix
    estimated_transformation = net_transformation(registration_transformations)
    print(transformation - estimated_transformation)
    assert_similar_transformations(transformation, estimated_transformation)


# TODO: Add unit tests for ICP


# HINT: You can generate test-cases by applying a random transformation to a mesh
#       and checking if ICP can 'undo' the transformation

# HINT: Start by testing (small) random translations of different primitives

# HINT: Once your method works for translations, test (small) random rotations

# HINT: Finally, make sure your method works for transformations with both a translation and a rotation component


def test_tetrahedron():
    translation = mathutils.Matrix.Translation(
        [
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
        ]
    )
    rotation = (
        mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "X")
        @ mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "Y")
        @ mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "Z")
    )
    transformation = translation @ rotation

    source, destination = primitives.TETRAHEDRON, primitives.TETRAHEDRON.copy()
    destination.transform(
        transformation
    )  # Move the destination, so we don't need to invert the transform

    registration_transformations = iterative_closest_point_registration(
        source,
        destination,
        k=2.5,
        num_points=4096,
        iterations=100,
        epsilon=0.0005,
        distance_metric="POINT_TO_PLANE",
        match_method="kdtree",
        p_norm="inf",  # 1, 2, inf
    )

    # The function should have converged
    assert len(registration_transformations) < 100

    # Check that we found the right matrix
    estimated_transformation = net_transformation(registration_transformations)
    assert_similar_transformations(transformation, estimated_transformation)


def test_mesh():
    translation = mathutils.Matrix.Translation(
        [
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
        ]
    )
    rotation = (
        mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "X")
        @ mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "Y")
        @ mathutils.Matrix.Rotation(random.uniform(-0.01, 0.01), 4, "Z")
    )
    transformation = translation @ rotation

    source, destination = (
        meshes.DOUBLE_TORUS,
        meshes.DOUBLE_TORUS.copy(),
    )
    destination.transform(
        transformation
    )  # Move the destination, so we don't need to invert the transform

    registration_transformations = iterative_closest_point_registration(
        source,
        destination,
        k=2,
        num_points=1000,
        iterations=100,
        epsilon=0.0005,
        distance_metric="POINT_TO_PLANE",
        match_method="kdtree",
        p_norm="inf",  # 1, 2, inf
    )

    # The function should have converged
    assert len(registration_transformations) < 100

    # Check that we found the right matrix
    estimated_transformation = net_transformation(registration_transformations)
    assert_similar_transformations(transformation, estimated_transformation)
