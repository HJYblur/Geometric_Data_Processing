import random
import sys

import bpy
import bmesh
import mathutils
import numpy as np
import numpy.random
import scipy


def numpy_verts(mesh: bmesh.types.BMesh) -> np.ndarray:
    """
    Extracts a numpy array of (x, y, z) vertices from a blender mesh

    :param mesh: The BMesh to extract the vertices of.
    :return: A numpy array of shape [n, 3], where array[i, :] is the x, y, z coordinate of vertex i.
    """
    data = bpy.data.meshes.new('tmp')
    mesh.to_mesh(data)
    # Explained here:
    # https://blog.michelanders.nl/2016/02/copying-vertices-to-numpy-arrays-in_4.html
    vertices = np.zeros(len(mesh.verts) * 3, dtype=np.float64)
    data.vertices.foreach_get('co', vertices)
    return vertices.reshape([len(mesh.verts), 3])


def numpy_normals(mesh: bmesh.types.BMesh) -> np.ndarray:
    """
    Extracts a numpy array of (x, y, z) normals from a blender mesh

    :param mesh: The BMesh to extract the normals of.
    :return: A numpy array of shape [n, 3], where array[i, :] is the x, y, z normal of vertex i.
    """
    data = bpy.data.meshes.new('tmp')
    mesh.to_mesh(data)
    normals = np.zeros(len(mesh.verts) * 3, dtype=np.float64)
    data.vertices.foreach_get('normal', normals)
    return normals.reshape([len(mesh.verts), 3])


# !!! This function will be used for automatic grading, don't edit the signature !!!
def point_to_point_transformation(
        source_points: np.ndarray,
        destination_points: np.ndarray,
        **kwargs
) -> mathutils.Matrix:
    """
    Given a set of point-pairs, finds an approximate transformation to register source points to destination points.

    Point pairs are passed as two separate matrices of the same shape, associations are made index-wise:
        source_points[i, :] --> destination_points[i, :]

    :param source_points: Collection of n points to move, represented by an [n, 3] numpy matrix.
    :param destination_points: Collection of n points to move toward, represented by an [n, 3] numpy matrix.
    :return: A transformation matrix which, applied to every point in source_points,
             would bring them closer to being registered with destination_points.
             The transformation should contain only translation and rotation components;
             this version of rigid registration should not re-scale the source mesh.
    """
    if len(source_points) == 0 or source_points.shape != destination_points.shape:
        return mathutils.Matrix.Identity(4)

    # TODO: Move both point clouds to the origin by finding their centroids

    # TODO: Find the covariance between the source and destination coordinates

    # TODO: Find a rotation matrix using SVD

    # TODO: Find a translation based on the rotated centroid (and not the original)

    # TODO: Return the combined matrix
    return mathutils.Matrix(4)


# !!! This function will be used for automatic grading, don't edit the signature !!!
def point_to_plane_transformation(
        source_points: np.ndarray,
        destination_points: np.ndarray,
        destination_normals: np.ndarray,
        **kwargs
) -> mathutils.Matrix:
    """
    Given a set of point-pairs, finds an approximate transformation to register source points to destination points.

    Here, the destination point are supplemented with normals.
    Instead of moving the source points toward each destination point,
    we can instead move them toward the planes defined by each destination point and its normal.

    Point pairs and normals are passed as three separate matrices of the same shape, associations are made index-wise:
        source_points[i, :] --> destination_points[i, :], destination_normals[i, :]

    :param source_points: Collection of n points to move, represented by an [n, 3] numpy matrix.
    :param destination_points: Collection of n points to move toward, represented by an [n, 3] numpy matrix.
    :param destination_normals: Collection of n normals of the destination points, represented by an [n, 3] numpy matrix.
    :return: A transformation matrix which, applied to every point in source_points,
             would bring them closer to being registered with destination_points.
             The transformation should contain only translation and rotation components;
             this version of rigid registration should not re-scale the source mesh.
    """

    # TODO: Implement a version of the ICP algorithm based on point-to-plane distances
    return mathutils.Matrix(4)


# !!! This function will be used for automatic grading, don't edit the signature !!!
def closest_point_registration(
        source: bmesh.types.BMesh, destination: bmesh.types.BMesh,
        k: float, num_points: int,
        distance_metric: str = "POINT_TO_POINT",
        **kwargs
) -> mathutils.Matrix:
    """
    Given a pair of meshes, finds an approximate transformation to register the source mesh to the destination.
    (This is one iteration of the rigid registration process)

    First, we randomly select some points from both meshes (determined by num_points).
    Next, we find the nearest point in the destination point selection for each point in the source selection.
    From these pairings, we find the median distance between source points and their associated destinations.

    When registering identical meshes, each source point will have a "true" counterpart in the destination mesh.
    At the end of iterative registration, all points should have ~0 distance from their counterpart.
    Because we've selected points randomly some counterparts of the source points won't be available to register with.
    We can exclude these outliers by rejecting any point pairs with distance greater than k * the median distance.
    When the meshes are fully registered, many point-pairs will have zero distance, so the median will be near-zero.

    Finally, we can find an approximate transformation by minimizing point-to-point or point-to-plane distances.
    The approach to use is determined by the distance_metric argument.

    NOTE: If you add additional arguments for your experiments, you can pass them as keyword arguments.
          Call the function like `iterative_closest_point_registration(..., my_custom_arg=42)`,
          And then use the value of the argument like `kwargs.get_default('my_custom_arg', 0)`.
          The default value is important, because our testing harness won't know about your custom args!

    :param source: Source mesh (mesh to move)
    :param destination: Destination mesh (mesh to move the source mesh toward)
    :param k: Point rejection coefficient, points further than k * (median distance) apart are not included.
    :param num_points: The maximum number of points to include for registration.
    :param distance_metric: Determines which approach to use for registration, "POINT_TO_POINT" or "POINT_TO_PLANE".
    :return: A transformation matrix which, applied to the source mesh,
             would bring it closer to being registered with the destination mesh.
             The transformation should contain only translation and rotation components;
             this version of rigid registration should not re-scale the source mesh.
    """
    # TODO: Find a transformation matrix which moves the source mesh closer to the destination mesh


    # HINT: Make sure not to select more points than are in the mesh or fewer than one point

    # TODO: Select some points from both meshes

    # TODO: Get the nearest destination point for each source point
    # HINT: scipy.spatial.KDTree makes this much faster!

    # TODO: Reject outlier point-pairs

    # Estimate a transformation based on the selected point-pairs
    if distance_metric == "POINT_TO_POINT":
        # TODO: call point_to_point_transformation() on your selected source and destination points
        return mathutils.Matrix.LocRotScale(
            mathutils.Vector(numpy.random.uniform(low=-0.1, high=0.1, size=[3])),
            mathutils.Matrix(numpy.random.uniform(low=-1, high=1, size=[4, 4])).to_quaternion(),
            mathutils.Vector([1, 1, 1]),
        )

    elif distance_metric == "POINT_TO_PLANE":
        raise NotImplementedError("Implement point-to-plant estimation")
    else:
        raise Exception(f"Unrecognized distance metric '{distance_metric}'")


# !!! This function will be used for automatic grading, don't edit the signature !!!
def iterative_closest_point_registration(
        source: bmesh.types.BMesh, destination: bmesh.types.BMesh,
        k: float, num_points: int,
        iterations: int, epsilon: float,
        distance_metric: str = "POINT_TO_POINT",
        **kwargs
) -> list[mathutils.Matrix]:
    """
    Iterative Closest Point (ICP) registration algorithm.
    Attempts to find a transformation which registers the source mesh to the destination.

    Applies closest-point registration until convergence or `iterations` is reached.
    Convergence is determined by epsilon:
    If the next transformation is approximately the same as the identity matrix (as determined by epsilon),
    then the registration is recommending no change to the source mesh's position, so we must have converged.

    :param source: Source mesh (mesh to move)
    :param destination: Destination mesh (mesh to move the source mesh toward)
    :param k: Point rejection coefficient, points further than k * (median distance) apart are not included.
    :param num_points: The maximum number of points to include for registration.
    :param iterations: The maximum number of iterations to use for registration.
    :param epsilon: Magnitude of allowable error in the final result.
    :param distance_metric: Determines which approach to use for registration, "POINT_TO_POINT" or "POINT_TO_PLANE".
    :return: A sequence of transformations which, applied to the source mesh in sequence,
             would move it so that it matches the destination mesh (registered).
             The transformation should contain only translation and rotation components;
             this version of rigid registration should not re-scale the source mesh.
             For some cases (such as non-identical meshes, or meshes with very different orientations)
             ICP may fail to converge, the transformations representing an attempted registration are still returned.
    """
    transformations = []
    for i in range(iterations):

        # Find a transformation which moves the source mesh closer to the target mesh
        transformation = closest_point_registration(
            source, destination,
            k, num_points,
            distance_metric,
            **kwargs
        )

        # Check for early-stopping (transformation is very similar to identity)
        deviation = np.asarray(transformation - mathutils.Matrix.Identity(4))
        if np.linalg.norm(deviation) < epsilon and np.max(deviation) < epsilon:
            break

        # Apply the transformation to the source mesh
        source.transform(transformation)
        transformations.append(transformation)

    return transformations


def net_transformation(transformations: list[mathutils.Matrix]) -> mathutils.Matrix:
    """
    Combines a sequence of transformations into a single transformation matrix with equivalent results.

    :param transformations: A list of transformation matrices.
    :return: A transformation matrix with equivalent results to applying all in sequence.
    """
    m = mathutils.Matrix.Identity(4)
    for t in transformations:
        m = t @ m
    return m
