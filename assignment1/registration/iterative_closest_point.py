import random
import sys

import bpy
import bmesh
import mathutils
import numpy as np
import scipy


def numpy_verts(mesh: bmesh.types.BMesh) -> np.ndarray:
    """
    Extracts a numpy array of (x, y, z) vertices from a blender mesh

    :param mesh: The BMesh to extract the vertices of.
    :return: A numpy array of shape [n, 3], where array[i, :] is the x, y, z coordinate of vertex i.
    """
    data = bpy.data.meshes.new("tmp")
    mesh.to_mesh(data)
    # Explained here:
    # https://blog.michelanders.nl/2016/02/copying-vertices-to-numpy-arrays-in_4.html
    vertices = np.zeros(len(mesh.verts) * 3, dtype=np.float64)
    data.vertices.foreach_get("co", vertices)
    return vertices.reshape([len(mesh.verts), 3])


def numpy_normals(mesh: bmesh.types.BMesh) -> np.ndarray:
    """
    Extracts a numpy array of (x, y, z) normals from a blender mesh

    :param mesh: The BMesh to extract the normals of.
    :return: A numpy array of shape [n, 3], where array[i, :] is the x, y, z normal of vertex i.
    """
    data = bpy.data.meshes.new("tmp")
    mesh.to_mesh(data)
    normals = np.zeros(len(mesh.verts) * 3, dtype=np.float64)
    data.vertices.foreach_get("normal", normals)
    return normals.reshape([len(mesh.verts), 3])


def farthest_point_sampling(points, k):
    centroids = []
    centroids_indices = []

    # Randomly select the first centroid
    first_index = np.random.randint(len(points))
    centroids_indices.append(first_index)
    centroids.append(points[first_index])

    # Fill the centroid list by selecting the farthest point
    for _ in range(k - 1):
        distances = []
        for x in points:
            # Find the min distance from point to current centroids
            min_dis_to_centroids = min(np.linalg.norm(x - centroids, axis=1))
            distances.append(min_dis_to_centroids)
        farthest_index = np.argmax(distances)
        centroids.append(points[farthest_index])
        centroids_indices.append(farthest_index)
    return np.array(centroids_indices)


# !!! This function will be used for automatic grading, don't edit the signature !!!
def point_to_point_transformation(
    source_points: np.ndarray, destination_points: np.ndarray, **kwargs
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

    # DONE: Move both point clouds to the origin by finding their centroids
    source_centroid = np.mean(source_points, axis=0)
    destination_centroid = np.mean(destination_points, axis=0)
    source_points -= source_centroid
    destination_points -= destination_centroid
    # print("Source points shape: ", source_points.shape)
    # print("Destination points shape: ", destination_points.shape)

    # DONE: Find the covariance between the source and destination coordinates
    covariance = source_points.T @ destination_points
    # print("Covariance matrix: \n", covariance.shape)

    # DONE: Find a rotation matrix using SVD
    # M = U D V^h
    # R = V^h (det(VU^h)) U^h
    u, _, vh = np.linalg.svd(covariance)
    v = vh.T
    middle = np.eye(3)
    middle[2, 2] = np.linalg.det(v @ u.T)
    rotation_matrix = v @ middle @ u.T

    # DONE: Find a translation based on the rotated centroid (and not the original)
    translation = destination_centroid - rotation_matrix @ source_centroid

    # DONE: Return the combined matrix
    transformation = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            transformation[i][j] = rotation_matrix[i][j]
        transformation[i][3] = translation[i]
    print("Transformation matrix: \n", transformation)
    return transformation
    # return mathutils.Matrix.Identity(4)


# !!! This function will be used for automatic grading, don't edit the signature !!!
def point_to_plane_transformation(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    destination_normals: np.ndarray,
    **kwargs,
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

    # DONE: Implement a version of the ICP algorithm based on point-to-plane distances

    # p -> source pts
    # n -> normals
    # q -> dst pts

    vertices_size = len(source_points)
    a = np.zeros((vertices_size, 6))  # (8, 6)
    a[:, 0:3] = np.cross(source_points, destination_normals)
    a[:, 3:6] = destination_normals
    A = a.T @ a
    print("A: ", A.shape)

    c = np.zeros(vertices_size)  # (8, )
    for i in range(vertices_size):
        c[i] = np.dot(
            (source_points[i] - destination_points[i]), destination_normals[i].T
        )
    print("c: ", c)

    b = a.T @ c  # (6, )
    print("b: ", b)

    lambda_reg = 1e-6  # Small factor to avoid singularity
    A_reg = A + lambda_reg * np.eye(6)
    try:
        answer = np.linalg.solve(A_reg, -b)
    except np.linalg.LinAlgError:
        print("Singular matrix, using least squares solver")
        answer = np.linalg.lstsq(A, -b, rcond=None)[0]

    r = answer[0:3]
    t = answer[3:6]
    print(f"r: {r}, t:{t}")

    # R = (source_points + np.cross(r, source_points)).T @ source_points
    R = np.array([[1, -r[2], r[1]], [r[2], 1, -r[0]], [-r[1], r[0], 1]])
    print("R: ", R)

    u, _, vh = np.linalg.svd(R)
    v = vh.T
    middle = np.eye(3)
    middle[2, 2] = np.linalg.det(v @ u.T)
    rotation_matrix = u @ middle @ vh

    transformation = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            transformation[i][j] = rotation_matrix[i][j]
        transformation[i][3] = t[i]
    print("Transformation matrix: \n", transformation)
    return transformation


def get_corresponding_indices_euclid(src_points, dst_points):
    """
    Finds the indices of the closest points in the destination point set for each point in the source point set
    using Euclidean distance.

    Args:
        src_points (numpy.ndarray): A 2D array of shape (N, D) representing the source points
        dst_points (numpy.ndarray): A 2D array of shape (M, D) representing the destination points.

    Returns:
        numpy.ndarray: An array containing the indices of the closest points in `dst_points` 
                       for each point in `src_points`.
    """
    distances_index = np.zeros(len(src_points))
    for i in range(len(src_points)):
        distances = np.linalg.norm(
            src_points[i] - dst_points, axis=1
        )
        distances_index[i] = np.argmin(distances)
    return distances_index.astype(int)


def get_corresponding_points_normals(src_points, src_normals, dst_points, dst_normals, depth, angle_weight):
    tree = scipy.spatial.KDTree(dst_points)

    # Query k closest neighbors for each point in src_points
    _, all_indices = tree.query(src_points, k=depth)

    # Ensure all_indices array is 2 dimensional
    if depth == 1:
        all_indices = all_indices[:, np.newaxis]  # (N,) → (N, 1)

    # dst_candidates shape: (N,k,3), normals shape: (N,k,3)
    dst_candidates = dst_points[all_indices]
    dst_normal_candidates = dst_normals[all_indices]

    # calculate the distance between the candidate pts and the source pts
    dist_costs = np.linalg.norm(
        dst_candidates - src_points[:, np.newaxis, :], 
        axis=2
    )
    # calculate the custom angle cost between the candidate normals and the source normals
    angle_costs = 1 - np.sum(
        src_normals[:, np.newaxis, :] * dst_normal_candidates, 
        axis=2
    )

    total_costs = dist_costs + angle_weight * angle_costs

    best_k_indices = np.argmin(total_costs, axis=1)  # shape (N,)
    best_indices = all_indices[np.arange(len(src_points)), best_k_indices]
    print(depth)
    return best_indices


# !!! This function will be used for automatic grading, don't edit the signature !!!
def closest_point_registration(
    source: bmesh.types.BMesh,
    destination: bmesh.types.BMesh,
    k: float,
    num_points: int,
    distance_metric: str = "POINT_TO_POINT",
    **kwargs,
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
    # DONE: Find a transformation matrix which moves the source mesh closer to the destination mesh
    # DONE: Read the parameters from the command line

    # DONE: Select some points from both meshes
    src_points = numpy_verts(source)
    src_normals = numpy_normals(source)
    dst_points = numpy_verts(destination)
    dst_normals = numpy_normals(destination)
    assert len(src_points) == len(
        dst_points
    ), "Source and destination meshes must have the same number of vertices"
    assert len(src_points) == len(
        dst_normals
    ), "Source vertices and destination normals must have the same number"

    # HINT: Make sure not to select more points than are in the mesh or fewer than one point
    num_points = np.clip(num_points, 1, len(src_points))

    # TODO: Use farthest-point sampling / normal-space sampling for sampling
    sampling_method = kwargs.get(
        "sampling_method", "random"
    )  #  or "farthest_point" or "normal_space"
    if sampling_method == "random":
        src_indices = random.sample(range(len(src_points)), num_points)
        dst_indices = random.sample(range(len(dst_points)), num_points)
    elif sampling_method == "farthest_point":
        src_indices = farthest_point_sampling(src_points, num_points)
        dst_indices = farthest_point_sampling(dst_points, num_points)
    elif sampling_method == "normal_space":
        pass
    else:
        raise (ValueError(f"Unsupported sampling method: {sampling_method}."))

    selected_src_points = src_points[src_indices]
    selected_src_normals = src_normals[src_indices]
    selected_dst_points = dst_points[dst_indices]
    selected_dst_normals = dst_normals[dst_indices]

    # DONE: Get the nearest destination point for each source point
    # DONE: HINT: scipy.spatial.KDTree makes this much faster!

    # "brute_force" or "kdtree"
    
    matching_metric = kwargs.get("matching_metric", "euclid")

    if matching_metric == "euclid":
        matching_method = kwargs.get("matching_method", "kdtree")

        if matching_method == "brute_force":
            new_indices = get_corresponding_indices_euclid(selected_src_points ,selected_dst_points)
        elif matching_method == "kdtree":
            k_neighbors = 1
            tree = scipy.spatial.KDTree(selected_dst_points)
            _, new_indices = tree.query(selected_src_points, k=k_neighbors)
        else:
            raise (ValueError(f"Unsupported matching method: {matching_method}."))
        
    elif matching_metric == "normals":
        new_indices = get_corresponding_points_normals(selected_src_points, selected_src_normals, selected_dst_points, selected_dst_normals, 10, 1)
    else:
        raise (ValueError(f"Unsupported matching method: {matching_metric}."))

    # Re-order destination points
    selected_dst_points = selected_dst_points[new_indices]
    selected_dst_normals = selected_dst_normals[new_indices]

    # DONE: Reject outlier point-pairs
    # DONE: use different p-norms for culling
    p_norms = kwargs.get("p_norms", "2")  # 1, 2, inf
    if p_norms == "1":
        distances = np.abs(selected_src_points - selected_dst_points).sum(
            axis=1
        )
    elif p_norms == "2":
        distances = np.linalg.norm(
            selected_src_points - selected_dst_points, axis=1
        )
    else:
        distances = np.linalg.norm(
            selected_src_points - selected_dst_points, axis=1, ord=np.inf
        )
    median_distance = np.median(distances)
    threshold = k * median_distance
    print(
        f"Median distance: {median_distance}, Threshold for rejecting outlier point-pairs: {threshold}"
    )
    boolean_list = distances < threshold
    selected_src_points = selected_src_points[boolean_list]
    selected_dst_points = selected_dst_points[boolean_list]
    selected_dst_normals = selected_dst_normals[boolean_list]

    # Estimate a transformation based on the selected point-pairs
    if distance_metric == "POINT_TO_POINT":
        return point_to_point_transformation(
            selected_src_points, selected_dst_points
        )
    elif distance_metric == "POINT_TO_PLANE":
        return point_to_plane_transformation(
            selected_src_points,
            selected_dst_points,
            selected_dst_normals,
        )
        # raise NotImplementedError("Implement point-to-plant estimation")
    else:
        raise Exception(f"Unrecognized distance metric '{distance_metric}'")


# !!! This function will be used for automatic grading, don't edit the signature !!!
def iterative_closest_point_registration(
    source: bmesh.types.BMesh,
    destination: bmesh.types.BMesh,
    k: float,
    num_points: int,
    iterations: int,
    epsilon: float,
    distance_metric: str = "POINT_TO_POINT",
    **kwargs,
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
        print(f"Iteration{i}\n")

        # Find a transformation which moves the source mesh closer to the target mesh
        transformation = closest_point_registration(
            source, destination, k, num_points, distance_metric, **kwargs
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
