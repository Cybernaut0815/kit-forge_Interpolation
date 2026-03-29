import numpy as np


def barycentric_interpolation_tri(tri: np.array, coordinate: np.array) -> np.array:
    """
    Compute the barycentric coordinates of points with respect to the triangle.
    
    Args:
        tri (np.array): Shape (3, 2) - Three vertices of the triangle
        coordinate (np.array): Shape (x, 2) - Points for which to compute barycentric coordinates,
                                where x can be any number of points
    
    Returns:
        np.array: Shape (x, 3) - Barycentric coordinates where each row contains [alpha, beta, gamma]:
            - alpha corresponds to vertex tri[0]
            - beta corresponds to vertex tri[1]      
            - gamma corresponds to vertex tri[2]
            The coordinates satisfy alpha + beta + gamma = 1 for each point
    
    Barycentric Coordinate System:
        - Alpha (first coordinate): weight for tri[0] vertex (first triangle vertex)
            alpha=1 places the point at tri[0], alpha=0 places it on the opposite edge
        - Beta (second coordinate): weight for tri[1] vertex (second triangle vertex)
            beta=1 places the point at tri[1], beta=0 places it on the opposite edge
        - Gamma (third coordinate): weight for tri[2] vertex (third triangle vertex)
            gamma=1 places the point at tri[2], gamma=0 places it on the opposite edge
        - Constraint: alpha + beta + gamma = 1 for any point within or on the triangle
        - Points inside the triangle have all coordinates > 0
        - Points on triangle edges have one coordinate = 0
        - Points at triangle vertices have one coordinate = 1 and others = 0
    """
    # Validate triangle shape
    if tri.shape != (3, 2):
        return None
    
    # Handle both single point and batch of points
    coordinate = np.atleast_2d(coordinate)
    if coordinate.shape[1] != 2:
        return None
    
    num_points = coordinate.shape[0]
    
    # Calculate signed area of the reference triangle
    # For 2D cross product: [a, b] × [c, d] = ad - bc
    edge1 = tri[1] - tri[0]  # Shape (2,)
    edge2 = tri[2] - tri[0]  # Shape (2,)
    signed_area_tri = 0.5 * (edge1[0] * edge2[1] - edge1[1] * edge2[0])
    
    if abs(signed_area_tri) < 1e-10:
        # Degenerate triangle
        return None
    
    # Vectorized barycentric coordinate calculation
    # For barycentric coordinates (u, v, w) corresponding to vertices (tri[0], tri[1], tri[2]):
    # u = signed_area(coordinate, tri[1], tri[2]) / signed_area_tri
    # v = signed_area(tri[0], coordinate, tri[2]) / signed_area_tri  
    # w = signed_area(tri[0], tri[1], coordinate) / signed_area_tri
    
    barycentric = np.zeros((num_points, 3))
    
    # Alpha (u): signed area of triangle (coordinate, tri[1], tri[2])
    v1 = tri[1] - coordinate  # Shape (x, 2)
    v2 = tri[2] - coordinate  # Shape (x, 2)
    cross_u = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # Shape (x,)
    barycentric[:, 0] = 0.5 * cross_u / signed_area_tri
    
    # Beta (v): signed area of triangle (tri[0], coordinate, tri[2])
    v1 = coordinate - tri[0]  # Shape (x, 2)
    v2 = tri[2] - tri[0]      # Shape (2,)
    cross_v = v1[:, 0] * v2[1] - v1[:, 1] * v2[0]  # Shape (x,)
    barycentric[:, 1] = 0.5 * cross_v / signed_area_tri
    
    # Gamma (w): Calculate as 1 - u - v to ensure they sum to 1
    barycentric[:, 2] = 1.0 - barycentric[:, 0] - barycentric[:, 1]
    
    return barycentric


def reverse_barycentric_interpolation_tri(tri: np.array, barycentric: np.array) -> np.array:
    """
    Compute the Cartesian coordinates of points given their barycentric coordinates with respect to the triangle.
    
    Args:
        tri (np.array): Shape (3, 2) - Three vertices of the triangle
        barycentric (np.array): Shape (x, 3) - Barycentric coordinates where each row contains
                                [alpha, beta, gamma] for a point, where x can be any number of points
    
    Returns:
        np.array: Shape (x, 2) - Cartesian coordinates of the interpolated points
    
    Barycentric Coordinate System:
        - Alpha (first coordinate): weight for tri[0] vertex (first triangle vertex)
            alpha=1 places the point at tri[0], alpha=0 places it on the opposite edge
        - Beta (second coordinate): weight for tri[1] vertex (second triangle vertex)
            beta=1 places the point at tri[1], beta=0 places it on the opposite edge
        - Gamma (third coordinate): weight for tri[2] vertex (third triangle vertex)
            gamma=1 places the point at tri[2], gamma=0 places it on the opposite edge
        - Constraint: alpha + beta + gamma = 1 for any point within or on the triangle
        - Points inside the triangle have all coordinates > 0
        - Points on triangle edges have one coordinate = 0
        - Points at triangle vertices have one coordinate = 1 and others = 0
    
    Note:
        The barycentric coordinates are applied as weighted sums to interpolate between
        the three triangle vertices, providing natural interpolation within the triangle.
    """
    # Validate triangle shape
    if tri.shape != (3, 2):
        return None
    
    # Handle both single point and batch of points
    barycentric = np.atleast_2d(barycentric)
    if barycentric.shape[1] != 3:
        return None
    
    # Vectorized interpolation using matrix multiplication
    # barycentric @ tri gives us the weighted sum of triangle vertices for each point
    return barycentric @ tri


def reverse_barycentric_interpolation_tri_with_tangents(tri: np.array, barycentric: np.array) -> tuple:
    """
    Compute the Cartesian coordinates and direction vectors to vertices given barycentric coordinates.
    
    This function maps barycentric coordinates to the corresponding Cartesian positions 
    within the triangle, and also computes the direction vectors from each point to the 
    three triangle vertices. These vectors represent the principal directions in the 
    barycentric coordinate system.
    
    Args:
        tri (np.array): Shape (3, 2) - Three vertices of the triangle
        barycentric (np.array): Shape (x, 3) - Barycentric coordinates where each row contains
                                [alpha, beta, gamma] for a point, where x can be any number of points
    
    Returns:
        tuple: (cartesian_coords, direction_to_v0, direction_to_v1, direction_to_v2) where:
            - cartesian_coords: Shape (x, 2) - Cartesian coordinates of interpolated points
            - direction_to_v0: Shape (x, 2) - Direction vectors from each point to vertex 0 (unnormalized)
            - direction_to_v1: Shape (x, 2) - Direction vectors from each point to vertex 1 (unnormalized)
            - direction_to_v2: Shape (x, 2) - Direction vectors from each point to vertex 2 (unnormalized)
            Returns (None, None, None, None) if input validation fails
    
    Barycentric Coordinate System:
        - Alpha (first coordinate): weight for tri[0] vertex (first triangle vertex)
            alpha=1 places the point at tri[0], alpha=0 places it on the opposite edge
        - Beta (second coordinate): weight for tri[1] vertex (second triangle vertex)
            beta=1 places the point at tri[1], beta=0 places it on the opposite edge
        - Gamma (third coordinate): weight for tri[2] vertex (third triangle vertex)
            gamma=1 places the point at tri[2], gamma=0 places it on the opposite edge
        - Constraint: alpha + beta + gamma = 1 for any point within or on the triangle
    
    Note:
        The direction vectors point from the corresponding vertex toward each interpolated point:
        - direction_to_v0 = P - tri[0]: points from vertex 0 toward the point
        - direction_to_v1 = P - tri[1]: points from vertex 1 toward the point
        - direction_to_v2 = P - tri[2]: points from vertex 2 toward the point
        
        These vectors are position-dependent and vary across the triangle. Their magnitudes 
        represent the distance from each vertex to the respective point. The vectors are NOT 
        normalized - their lengths provide information about proximity to each vertex.
        
        Properties:
        - At a vertex, the corresponding direction vector is zero
        - At the centroid, all three vectors have equal magnitude
        - These vectors satisfy: alpha * direction_to_v0 + beta * direction_to_v1 + gamma * direction_to_v2 = 0
        
        Users can normalize them if needed using np.linalg.norm().
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_barycentric_interpolation_tri(tri, barycentric)
    if cartesian_coords is None:
        return None, None, None, None
    
    v0 = tri[0]  # Shape (2,)
    v1 = tri[1]  # Shape (2,)
    v2 = tri[2]  # Shape (2,)
    
    # Compute direction vectors from each vertex to each point
    # These vectors vary with position (unlike edge-based tangents)
    direction_to_v0 = cartesian_coords - v0[np.newaxis, :]  # Shape (x, 2)
    direction_to_v1 = cartesian_coords - v1[np.newaxis, :]  # Shape (x, 2)
    direction_to_v2 = cartesian_coords - v2[np.newaxis, :]  # Shape (x, 2)
    
    return cartesian_coords, direction_to_v0, direction_to_v1, direction_to_v2