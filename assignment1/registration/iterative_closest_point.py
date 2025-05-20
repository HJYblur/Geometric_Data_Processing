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
    n = len(points)
    centroids_indices = []
    # Start with a random point
    first_index = np.random.randint(n)
    centroids_indices.append(first_index)
    # Initialize distances to inf
    min_distances = np.full(n, np.inf)
    # Compute distances from all points to the first centroid
    diff = points - points[first_index]
    min_distances = np.linalg.norm(diff, axis=1)
    for _ in range(1, k):
        # Select the point with the maximum distance to any centroid so far
        farthest_index = np.argmax(min_distances)
        centroids_indices.append(farthest_index)
        # Update the min_distances array
        diff = points - points[farthest_index]
        distances = np.linalg.norm(diff, axis=1)
        min_distances = np.minimum(min_distances, distances)
    return np.array(centroids_indices)


def normal_space_sampling(normals: np.ndarray, n: int):
    # Ensure normals are normalized
    unit_normals = normals / np.linalg.norm(normals)

    # Generate n "evenly spaced" points on a sphere, adapted from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    bins = np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )

    # Place normals to bins (the points on the sphere) using kdtree
    tree = scipy.spatial.KDTree(bins)
    bin_indices = tree.query(unit_normals)[1]
    bin_counts = np.bincount(bin_indices, minlength=n)

    valid_bins = np.where(bin_counts > 0)[0]

    selected_indices = np.zeros(n, dtype=np.int64)

    cum_counts = np.cumsum(bin_counts)  # This is the "end index" of each bin
    start_indices = np.zeros(n, dtype=np.int64)
    start_indices[1:] = cum_counts[:-1] # Start of each bin is the end of the previous bin
    # select a point from each bin randomly
    for bin_idx in valid_bins:
        start = start_indices[bin_idx]
        end = cum_counts[bin_idx]
        selected_indices[bin_idx] = np.random.randint(start, end)

    # Handle bins with no points (simply get nearest neighbor)
    empty_bins = np.where(bin_counts == 0)[0]
    if len(empty_bins) > 0:
        _, nearest = tree.query(bins[empty_bins])
        selected_indices[empty_bins] = nearest

    return selected_indices.astype(int)


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

    c = np.zeros(vertices_size)  # (8, )
    for i in range(vertices_size):
        c[i] = np.dot(
            (source_points[i] - destination_points[i]), destination_normals[i].T
        )

    b = a.T @ c  # (6, )

    lambda_reg = 1e-6  # Small factor to avoid singularity
    A_reg = A + lambda_reg * np.eye(6)
    try:
        answer = np.linalg.solve(A_reg, -b)
    except np.linalg.LinAlgError:
        print("Singular matrix, using least squares solver")
        answer = np.linalg.lstsq(A, -b, rcond=None)[0]

    r = answer[0:3]
    t = answer[3:6]

    # R = (source_points + np.cross(r, source_points)).T @ source_points
    R = np.array([[1, -r[2], r[1]], [r[2], 1, -r[0]], [-r[1], r[0], 1]])

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

    return transformation


def get_corresponding_indices_euclid(src_points, dst_points):
    """
    Finds the indices of the closest points in the destination point set for each point in the source point set
    using Euclidean distance.

    :param src_points: (numpy.ndarray): A 2D array of shape (N, D) representing the source points
    :param dst_points: (numpy.ndarray): A 2D array of shape (M, D) representing the destination points.
    :return: numpy.ndarray: An array containing the indices of the closest points in `dst_points` 
                       for each point in `src_points`.
    """
    distances_index = np.zeros(len(src_points))
    for i in range(len(src_points)):
        distances = np.linalg.norm(src_points[i] - dst_points, axis=1)
        distances_index[i] = np.argmin(distances)
    return distances_index.astype(int)


def get_corresponding_indices_normals(
    src_points, src_normals, dst_points, dst_normals, k, angle_weight
):
    """
    Given two sets of (src and dest) points and their corresponding normals, matches the src to dst based on distance and normal similarity.

    :param src_points: Collection of n points to move, represented by an [n, 3] numpy matrix.
    :param src_normals: Collection of n source normals, represented by an [n, 3] numpy matrix.
    :param dst_points: Collection of n points to move toward, represented by an [n, 3] numpy matrix.
    :param dst_normals: Collection of n normals of the destination points, represented by an [n, 3] numpy matrix.
    :param k: The number of closest destination points to consider for every source point.
    :param angle_weight: The weight applied to the similarity of angles. Setting it to zero is equivalent to closest-point matching
    :return: The indices of the destination points that match the src points.
    """
    tree = scipy.spatial.KDTree(dst_points)

    # Make sure number of neighbors is not greater than the number of points
    k = min(k, len(src_points))

    # Query k closest neighbors for each point in src_points
    _, all_indices = tree.query(src_points, k=k)

    # Ensure all_indices array is 2 dimensional
    if k == 1:
        all_indices = all_indices[:, np.newaxis]  # (N,) → (N, 1)

    # dst_candidates shape: (N,k,3), normals shape: (N,k,3)
    dst_candidates = dst_points[all_indices]
    dst_normal_candidates = dst_normals[all_indices]

    # calculate the distance between the candidate pts and the source pts
    dist_costs = np.linalg.norm(dst_candidates - src_points[:, np.newaxis, :], axis=2)
    # calculate the custom angle cost between the candidate normals and the source normals
    angle_costs = 1 - np.sum(
        src_normals[:, np.newaxis, :] * dst_normal_candidates, axis=2
    )

    total_costs = dist_costs + angle_weight * angle_costs

    best_k_indices = np.argmin(total_costs, axis=1)  # shape (N,)
    best_indices = all_indices[np.arange(len(src_points)), best_k_indices]
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
        src_indices = normal_space_sampling(src_normals, num_points)
        dst_indices = normal_space_sampling(dst_normals, num_points)
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
            new_indices = get_corresponding_indices_euclid(
                selected_src_points, selected_dst_points
            )
        elif matching_method == "kdtree":
            k_neighbors = 1
            tree = scipy.spatial.KDTree(selected_dst_points)
            _, new_indices = tree.query(selected_src_points, k=k_neighbors)
        else:
            raise (ValueError(f"Unsupported matching method: {matching_method}."))
    elif matching_metric == "normals":
        new_indices = get_corresponding_indices_normals(
            selected_src_points,
            selected_src_normals,
            selected_dst_points,
            selected_dst_normals,
            10,
            1,
        )
    else:
        raise (ValueError(f"Unsupported matching method: {matching_metric}."))

    # Re-order destination points
    selected_dst_points = selected_dst_points[new_indices]
    selected_dst_normals = selected_dst_normals[new_indices]

    # DONE: Reject outlier point-pairs
    # DONE: use different p-norms for culling
    p_norms = kwargs.get("p_norms", "2")  # 1, 2, inf
    if p_norms == "1":
        distances = np.abs(selected_src_points - selected_dst_points).sum(axis=1)
    elif p_norms == "2":
        distances = np.linalg.norm(selected_src_points - selected_dst_points, axis=1)
    else:
        distances = np.linalg.norm(
            selected_src_points - selected_dst_points, axis=1, ord=np.inf
        )
    median_distance = np.median(distances)
    threshold = k * median_distance
    # print(
    #     f"Median distance: {median_distance}, Threshold for rejecting outlier point-pairs: {threshold}"
    # )
    boolean_list = distances < threshold
    selected_src_points = selected_src_points[boolean_list]
    selected_dst_points = selected_dst_points[boolean_list]
    selected_dst_normals = selected_dst_normals[boolean_list]

    #print(len([0 for b in boolean_list if not b]))
    # Estimate a transformation based on the selected point-pairs
    if distance_metric == "POINT_TO_POINT":
        return point_to_point_transformation(selected_src_points, selected_dst_points)
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
        # print(f"Iteration{i}\n")

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


def calculate_mse(source: bmesh.types.BMesh, destination: bmesh.types.BMesh) -> float:
    """
    Calculate Mean Squared Error between source and destination meshes.

    :param source: Source mesh
    :param destination: Destination mesh
    :return: Mean Squared Error value
    """
    src_points = numpy_verts(source)
    dst_points = numpy_verts(destination)
    indices = get_corresponding_indices_euclid(src_points, dst_points)
    closest_dst_points = dst_points[indices]
    squared_distances = np.sum((src_points - closest_dst_points) ** 2, axis=1)
    mse_value = np.mean(squared_distances)
    return mse_value


def calculate_hausdorff_distance(
    source: bmesh.types.BMesh, destination: bmesh.types.BMesh
) -> float:
    """
    Calculate Hausdorff Distance between source and destination meshes.

    :param source: Source mesh
    :param destination: Destination mesh
    :return: Hausdorff Distance value
    """
    src_points = numpy_verts(source)
    dst_points = numpy_verts(destination)

    # Forward Hausdorff distance (source to destination)
    indices_fwd = get_corresponding_indices_euclid(src_points, dst_points)
    closest_dst_points = dst_points[indices_fwd]
    forward_distances = np.sqrt(np.sum((src_points - closest_dst_points) ** 2, axis=1))
    forward_hausdorff = np.max(forward_distances)

    # Backward Hausdorff distance (destination to source)
    indices_back = get_corresponding_indices_euclid(dst_points, src_points)
    closest_src_points = src_points[indices_back]
    backward_distances = np.sqrt(np.sum((dst_points - closest_src_points) ** 2, axis=1))
    backward_hausdorff = np.max(backward_distances)

    # True Hausdorff distance is the maximum of forward and backward
    hausdorff_value = max(forward_hausdorff, backward_hausdorff)
    return hausdorff_value


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
