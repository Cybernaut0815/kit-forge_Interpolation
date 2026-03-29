import numpy as np


def barycentric_interpolation_tetrahedron(tetrahedron: np.array, coordinate: np.array) -> np.array:
    """
    Compute the barycentric coordinates of points with respect to the tetrahedron.
    
    Args:
        tetrahedron (np.array): Shape (4, 3) - Four vertices of the tetrahedron
        coordinate (np.array): Shape (x, 3) - Points for which to compute barycentric coordinates,
                                where x can be any number of points
    
    Returns:
        np.array: Shape (x, 4) - Barycentric coordinates where each row contains [alpha, beta, gamma, delta]:
            - alpha corresponds to vertex tetrahedron[0]
            - beta corresponds to vertex tetrahedron[1]      
            - gamma corresponds to vertex tetrahedron[2]
            - delta corresponds to vertex tetrahedron[3]
            The coordinates satisfy alpha + beta + gamma + delta = 1 for each point
    
    Barycentric Coordinate System:
        - Alpha (first coordinate): weight for tetrahedron[0] vertex (first tetrahedron vertex)
          alpha=1 places the point at tetrahedron[0], alpha=0 places it on the opposite face
        - Beta (second coordinate): weight for tetrahedron[1] vertex (second tetrahedron vertex)
          beta=1 places the point at tetrahedron[1], beta=0 places it on the opposite face
        - Gamma (third coordinate): weight for tetrahedron[2] vertex (third tetrahedron vertex)
          gamma=1 places the point at tetrahedron[2], gamma=0 places it on the opposite face
        - Delta (fourth coordinate): weight for tetrahedron[3] vertex (fourth tetrahedron vertex)
          delta=1 places the point at tetrahedron[3], delta=0 places it on the opposite face
        - Constraint: alpha + beta + gamma + delta = 1 for any point within or on the tetrahedron
        - Points inside the tetrahedron have all coordinates > 0
        - Points on tetrahedron faces have one coordinate = 0
        - Points at tetrahedron vertices have one coordinate = 1 and others = 0
    """
    # Validate tetrahedron shape
    if tetrahedron.shape != (4, 3):
        return None
    
    # Handle both single point and batch of points
    coordinate = np.atleast_2d(coordinate)
    if coordinate.shape[1] != 3:
        return None
    
    num_points = coordinate.shape[0]
    
    # Calculate signed volume of the reference tetrahedron
    # Signed volume = (1/6) * dot((v1-v0), cross((v2-v0), (v3-v0)))
    edge1 = tetrahedron[1] - tetrahedron[0]  # Shape (3,)
    edge2 = tetrahedron[2] - tetrahedron[0]  # Shape (3,)
    edge3 = tetrahedron[3] - tetrahedron[0]  # Shape (3,)
    
    cross_product = np.cross(edge2, edge3)
    signed_volume_tet = np.dot(edge1, cross_product) / 6.0
    
    if abs(signed_volume_tet) < 1e-10:
        # Degenerate tetrahedron
        return None
    
    # Vectorized barycentric coordinate calculation
    # For barycentric coordinates (u, v, w, t) corresponding to vertices (tet[0], tet[1], tet[2], tet[3]):
    # u = signed_volume(coordinate, tet[1], tet[2], tet[3]) / signed_volume_tet
    # v = signed_volume(tet[0], coordinate, tet[2], tet[3]) / signed_volume_tet  
    # w = signed_volume(tet[0], tet[1], coordinate, tet[3]) / signed_volume_tet
    # t = 1 - u - v - w
    
    barycentric = np.zeros((num_points, 4))
    
    # Alpha (u): signed volume of tetrahedron (coordinate, tet[1], tet[2], tet[3])
    v1 = tetrahedron[1] - coordinate  # Shape (x, 3)
    v2 = tetrahedron[2] - coordinate  # Shape (x, 3)
    v3 = tetrahedron[3] - coordinate  # Shape (x, 3)
    cross_prod = np.cross(v2, v3)  # Shape (x, 3)
    barycentric[:, 0] = np.sum(v1 * cross_prod, axis=1) / (6.0 * signed_volume_tet)
    
    # Beta (v): signed volume of tetrahedron (tet[0], coordinate, tet[2], tet[3])
    v1 = coordinate - tetrahedron[0]  # Shape (x, 3)
    v2 = tetrahedron[2] - tetrahedron[0]  # Shape (3,)
    v3 = tetrahedron[3] - tetrahedron[0]  # Shape (3,)
    cross_prod = np.cross(v2, v3)  # Shape (3,)
    barycentric[:, 1] = np.dot(v1, cross_prod) / (6.0 * signed_volume_tet)
    
    # Gamma (w): signed volume of tetrahedron (tet[0], tet[1], coordinate, tet[3])
    v1 = tetrahedron[1] - tetrahedron[0]  # Shape (3,)
    v2 = coordinate - tetrahedron[0]  # Shape (x, 3)
    v3 = tetrahedron[3] - tetrahedron[0]  # Shape (3,)
    cross_prod = np.cross(v2, v3)  # Shape (x, 3)
    barycentric[:, 2] = np.dot(v1, cross_prod.T) / (6.0 * signed_volume_tet)
    
    # Delta (t): Calculate as 1 - u - v - w to ensure they sum to 1
    barycentric[:, 3] = 1.0 - barycentric[:, 0] - barycentric[:, 1] - barycentric[:, 2]
    
    return barycentric


def reverse_barycentric_interpolation_tetrahedron(tetrahedron: np.array, barycentric: np.array) -> np.array:
    """
    Compute the Cartesian coordinates of points given their barycentric coordinates with respect to the tetrahedron.
    
    Args:
        tetrahedron (np.array): Shape (4, 3) - Four vertices of the tetrahedron
        barycentric (np.array): Shape (x, 4) - Barycentric coordinates where each row contains
                                [alpha, beta, gamma, delta] for a point, where x can be any number of points
    
    Returns:
        np.array: Shape (x, 3) - Cartesian coordinates of the interpolated points
    
    Barycentric Coordinate System:
        - Alpha (first coordinate): weight for tetrahedron[0] vertex (first tetrahedron vertex)
          alpha=1 places the point at tetrahedron[0], alpha=0 places it on the opposite face
        - Beta (second coordinate): weight for tetrahedron[1] vertex (second tetrahedron vertex)
          beta=1 places the point at tetrahedron[1], beta=0 places it on the opposite face
        - Gamma (third coordinate): weight for tetrahedron[2] vertex (third tetrahedron vertex)
          gamma=1 places the point at tetrahedron[2], gamma=0 places it on the opposite face
        - Delta (fourth coordinate): weight for tetrahedron[3] vertex (fourth tetrahedron vertex)
          delta=1 places the point at tetrahedron[3], delta=0 places it on the opposite face
        - Constraint: alpha + beta + gamma + delta = 1 for any point within or on the tetrahedron
        - Points inside the tetrahedron have all coordinates > 0
        - Points on tetrahedron faces have one coordinate = 0
        - Points at tetrahedron vertices have one coordinate = 1 and others = 0
    
    Note:
        The barycentric coordinates are applied as weighted sums to interpolate between
        the four tetrahedron vertices, providing natural interpolation within the tetrahedron.
    """
    # Validate tetrahedron shape
    if tetrahedron.shape != (4, 3):
        return None
    
    # Handle both single point and batch of points
    barycentric = np.atleast_2d(barycentric)
    if barycentric.shape[1] != 4:
        return None
    
    # Vectorized interpolation using matrix multiplication
    # barycentric @ tetrahedron gives us the weighted sum of tetrahedron vertices for each point
    return barycentric @ tetrahedron


def reverse_barycentric_interpolation_tetrahedron_with_tangents(tetrahedron: np.array, barycentric: np.array) -> tuple:
    """
    Compute the Cartesian coordinates and direction vectors to vertices given barycentric coordinates.
    
    This function maps barycentric coordinates to the corresponding Cartesian positions 
    within the tetrahedron, and also computes the direction vectors from each point to the 
    four tetrahedron vertices. These vectors represent the principal directions in the 
    barycentric coordinate system.
    
    Args:
        tetrahedron (np.array): Shape (4, 3) - Four vertices of the tetrahedron
        barycentric (np.array): Shape (x, 4) - Barycentric coordinates where each row contains
                                [alpha, beta, gamma, delta] for a point, where x can be any number of points
    
    Returns:
        tuple: (cartesian_coords, direction_to_v0, direction_to_v1, direction_to_v2, direction_to_v3) where:
            - cartesian_coords: Shape (x, 3) - Cartesian coordinates of interpolated points
            - direction_to_v0: Shape (x, 3) - Direction vectors from each point to vertex 0 (unnormalized)
            - direction_to_v1: Shape (x, 3) - Direction vectors from each point to vertex 1 (unnormalized)
            - direction_to_v2: Shape (x, 3) - Direction vectors from each point to vertex 2 (unnormalized)
            - direction_to_v3: Shape (x, 3) - Direction vectors from each point to vertex 3 (unnormalized)
            Returns (None, None, None, None, None) if input validation fails
    
    Barycentric Coordinate System:
        - Alpha (first coordinate): weight for tetrahedron[0] vertex (first tetrahedron vertex)
          alpha=1 places the point at tetrahedron[0], alpha=0 places it on the opposite face
        - Beta (second coordinate): weight for tetrahedron[1] vertex (second tetrahedron vertex)
          beta=1 places the point at tetrahedron[1], beta=0 places it on the opposite face
        - Gamma (third coordinate): weight for tetrahedron[2] vertex (third tetrahedron vertex)
          gamma=1 places the point at tetrahedron[2], gamma=0 places it on the opposite face
        - Delta (fourth coordinate): weight for tetrahedron[3] vertex (fourth tetrahedron vertex)
          delta=1 places the point at tetrahedron[3], delta=0 places it on the opposite face
        - Constraint: alpha + beta + gamma + delta = 1 for any point within or on the tetrahedron
    
    Note:
        The direction vectors point from the corresponding vertex toward each interpolated point:
        - direction_to_v0 = P - tetrahedron[0]: points from vertex 0 toward the point
        - direction_to_v1 = P - tetrahedron[1]: points from vertex 1 toward the point
        - direction_to_v2 = P - tetrahedron[2]: points from vertex 2 toward the point
        - direction_to_v3 = P - tetrahedron[3]: points from vertex 3 toward the point
        
        These vectors are position-dependent and vary across the tetrahedron. Their magnitudes 
        represent the distance from each vertex to the respective point. The vectors are NOT 
        normalized - their lengths provide information about proximity to each vertex.
        
        Properties:
        - At a vertex, the corresponding direction vector is zero
        - At the centroid, all four vectors have equal magnitude
        - These vectors satisfy: alpha * direction_to_v0 + beta * direction_to_v1 + 
                                gamma * direction_to_v2 + delta * direction_to_v3 = 0
        
        Users can normalize them if needed using np.linalg.norm().
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_barycentric_interpolation_tetrahedron(tetrahedron, barycentric)
    if cartesian_coords is None:
        return None, None, None, None, None
    
    v0 = tetrahedron[0]  # Shape (3,)
    v1 = tetrahedron[1]  # Shape (3,)
    v2 = tetrahedron[2]  # Shape (3,)
    v3 = tetrahedron[3]  # Shape (3,)
    
    # Compute direction vectors from each vertex to each point
    # These vectors vary with position
    direction_to_v0 = cartesian_coords - v0[np.newaxis, :]  # Shape (x, 3)
    direction_to_v1 = cartesian_coords - v1[np.newaxis, :]  # Shape (x, 3)
    direction_to_v2 = cartesian_coords - v2[np.newaxis, :]  # Shape (x, 3)
    direction_to_v3 = cartesian_coords - v3[np.newaxis, :]  # Shape (x, 3)
    
    return cartesian_coords, direction_to_v0, direction_to_v1, direction_to_v2, direction_to_v3