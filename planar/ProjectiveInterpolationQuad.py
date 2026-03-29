import numpy as np


def projective_interpolation_quad(quad: np.array, coordinate: np.array) -> np.array:
    """
    Convert Cartesian coordinates to UV coordinates within a quadrilateral using projective interpolation.
    
    This function finds the normalized UV coordinates (where u,v ∈ [0,1]) that correspond to
    given Cartesian positions within the quadrilateral. Uses perspective transformation to map
    from the quadrilateral to a unit square UV space.
    
    This function should be only used in specific cases, hence it does not equally subdivide the quadrilateral.
    For the general case in geometric settings, use bilinear interpolation.
    
    Args:
        quad (np.array): Shape (4, 2) - Four vertices of the quadrilateral in order:
                        [bottom-left, bottom-right, top-left, top-right]
        coordinate (np.array): Shape (x, 2) - Cartesian coordinates of the points to convert
                                where x can be any number of points
    
    Returns:
        np.array: Shape (x, 2) - UV coordinates where [0,0] corresponds to bottom-left vertex
                        and [1,1] corresponds to top-right vertex, or None if input validation fails
    
    UV Coordinate System:
        - U axis (first coordinate): varies from 0 to 1 across the quadrilateral's horizontal parametric direction
            u=0 corresponds to left edge vertices (bottom-left and top-left)
            u=1 corresponds to right edge vertices (bottom-right and top-right)
        - V axis (second coordinate): varies from 0 to 1 across the quadrilateral's vertical parametric direction
            v=0 corresponds to bottom edge vertices (bottom-left and bottom-right)
            v=1 corresponds to top edge vertices (top-left and top-right)
    
    Note:
        Uses projective transformation for more accurate interpolation in perspective-distorted quads.
        The unit square reference is [(0,0), (1,0), (0,1), (1,1)].
    """
    # Validate quad shape
    if quad.shape != (4, 2):
        return None
    
    # Handle both single point and batch of points
    coordinate = np.atleast_2d(coordinate)
    if coordinate.shape[1] != 2:
        return None
    
    # Define unit square as target UV space
    unit_square = np.array([
        [0, 0],  # bottom-left
        [1, 0],  # bottom-right  
        [0, 1],  # top-left
        [1, 1]   # top-right
    ])
    
    # Get transformation matrix from quad to unit square
    T = getPerspectiveTransform(quad, unit_square)
    
    # Apply transformation to convert coordinates to UV space
    return transform_points(coordinate, T)


def reverse_projective_interpolation_quad(quad: np.array, uv: np.array) -> np.array:
    """
    Convert UV coordinates to Cartesian coordinates within a quadrilateral using projective interpolation.
    
    This function maps normalized UV coordinates (where u,v ∈ [0,1]) to the corresponding 
    Cartesian positions within the quadrilateral defined by four vertices. Uses perspective 
    transformation for accurate mapping in perspective-distorted quads.
    
    This function should be only used in specific cases, hence it does not equally subdivide the quadrilateral.
    For the general case in geometric settings, use bilinear interpolation.
    
    Args:
        quad (np.array): Shape (4, 2) - Four vertices of the quadrilateral in order:
                        [bottom-left, bottom-right, top-left, top-right]
        uv (np.array): Shape (x, 2) - UV coordinates where [0,0] maps to bottom-left vertex
                        and [1,1] maps to top-right vertex, where x can be any number of points
    
    Returns:
        np.array: Shape (x, 2) - Cartesian coordinates of the interpolated points,
                        or None if input validation fails
    
    UV Coordinate System:
        - U axis (first coordinate): varies from 0 to 1 across the quadrilateral's horizontal parametric direction
            u=0 corresponds to left edge vertices (bottom-left and top-left)
            u=1 corresponds to right edge vertices (bottom-right and top-right)
        - V axis (second coordinate): varies from 0 to 1 across the quadrilateral's vertical parametric direction
            v=0 corresponds to bottom edge vertices (bottom-left and bottom-right)
            v=1 corresponds to top edge vertices (top-left and top-right)
    
    Note:
        The UV coordinates are applied using perspective transformation to accurately handle
        perspective-distorted quadrilaterals, providing better results than bilinear interpolation
        for non-planar or strongly perspective-distorted shapes.
        The unit square reference is [(0,0), (1,0), (0,1), (1,1)].
    """
    # Validate quad shape
    if quad.shape != (4, 2):
        return None
    
    # Handle both single point and batch of points
    uv = np.atleast_2d(uv)
    if uv.shape[1] != 2:
        return None
    
    # Define unit square as source UV space
    unit_square = np.array([
        [0, 0],  # bottom-left
        [1, 0],  # bottom-right
        [0, 1],  # top-left
        [1, 1]   # top-right
    ])
    
    # Get transformation matrix from unit square to quad
    T = getPerspectiveTransform(unit_square, quad)
    
    # Apply transformation to convert UV coordinates to Cartesian space
    return transform_points(uv, T)


def getPerspectiveTransform(src_quad: np.array, dst_quad: np.array) -> np.array:
    """Computes the perspective transformation matrix from 4 corresponding points."""
    # Vectorized construction of coefficient matrix A
    # Create row indices for even and odd rows
    src_x = src_quad[:, 0]  # Shape (4,)
    src_y = src_quad[:, 1]  # Shape (4,)
    dst_x = dst_quad[:, 0]  # Shape (4,)
    dst_y = dst_quad[:, 1]  # Shape (4,)
    
    # Build matrix A (8x8) using vectorized operations
    A = np.zeros((8, 8))
    # Even rows (0, 2, 4, 6) for x-coordinates
    A[0::2, 0] = src_x
    A[0::2, 1] = src_y
    A[0::2, 2] = 1
    A[0::2, 6] = -dst_x * src_x
    A[0::2, 7] = -dst_x * src_y
    
    # Odd rows (1, 3, 5, 7) for y-coordinates
    A[1::2, 3] = src_x
    A[1::2, 4] = src_y
    A[1::2, 5] = 1
    A[1::2, 6] = -dst_y * src_x
    A[1::2, 7] = -dst_y * src_y

    # Vectorized construction of target vector b
    b = np.zeros(8)
    b[0::2] = dst_x  # Even indices get x-coordinates
    b[1::2] = dst_y  # Odd indices get y-coordinates

    # Solve the linear system to find the transformation matrix
    T = np.linalg.solve(A, b)

    # Reshape the transformation matrix to a 3x3 matrix
    T = np.append(T, 1).reshape((3, 3))

    return T


def transform_points(points: np.array, T: np.array) -> np.array:
    """
    Apply perspective transformation to points.
    
    Args:
        points (np.array): Shape (x, 2) or (2,) - Point coordinates where x can be any number of points
        T (np.array): Shape (3, 3) - Perspective transformation matrix
    
    Returns:
        np.array: Shape (x, 2) or (2,) - Transformed points, output shape matches input
    
    Examples:
        # Single point
        result = transform_points(np.array([1.0, 2.0]), T)
        
        # Multiple points  
        result = transform_points(np.array([[1,2], [3,4]]), T)
    """
    original_points = np.array(points)
    was_single_point = (original_points.ndim == 1)
    
    points = np.atleast_2d(points)
    if points.shape[1] != 2:
        raise ValueError("Points must have shape (x, 2)")
    
    # Apply perspective transformation
    num_points = points.shape[0]
    
    # Convert to homogeneous coordinates
    homogeneous = np.ones((num_points, 3))
    homogeneous[:, :2] = points
    
    # Apply transformation: T @ points^T
    transformed = (T @ homogeneous.T).T  # Shape (x, 3)
    
    # Convert back from homogeneous coordinates
    result = np.zeros((num_points, 2))
    result[:, 0] = transformed[:, 0] / transformed[:, 2]
    result[:, 1] = transformed[:, 1] / transformed[:, 2]
    
    # If input was 1D (single point), return 1D result for backward compatibility
    if was_single_point:
        return result[0]
    else:
        return result


def reverse_projective_interpolation_quad_with_tangents(quad: np.array, uv: np.array) -> tuple:
    """
    Convert UV coordinates to Cartesian coordinates and compute tangent vectors using projective interpolation.
    
    This function maps normalized UV coordinates (where u,v ∈ [0,1]) to the corresponding 
    Cartesian positions within the quadrilateral, and also computes the tangent vectors 
    in the u and v parametric directions at each point using the Jacobian of the perspective 
    transformation.
    
    Args:
        quad (np.array): Shape (4, 2) - Four vertices of the quadrilateral in order:
                        [bottom-left, bottom-right, top-left, top-right]
        uv (np.array): Shape (x, 2) - UV coordinates where [0,0] maps to bottom-left vertex
                        and [1,1] maps to top-right vertex, where x can be any number of points
    
    Returns:
        tuple: (cartesian_coords, tangent_u, tangent_v) where:
            - cartesian_coords: Shape (x, 2) - Cartesian coordinates of interpolated points
            - tangent_u: Shape (x, 2) - Tangent vectors in u-direction at each point (unnormalized)
            - tangent_v: Shape (x, 2) - Tangent vectors in v-direction at each point (unnormalized)
            Returns (None, None, None) if input validation fails
    
    UV Coordinate System:
        - U axis (first coordinate): varies from 0 to 1 across the quadrilateral's horizontal parametric direction
            u=0 corresponds to left edge vertices (bottom-left and top-left)
            u=1 corresponds to right edge vertices (bottom-right and top-right)
        - V axis (second coordinate): varies from 0 to 1 across the quadrilateral's vertical parametric direction
            v=0 corresponds to bottom edge vertices (bottom-left and bottom-right)
            v=1 corresponds to top edge vertices (top-left and top-right)
    
    Note:
        The tangent vectors are computed as the Jacobian of the perspective transformation:
        - ∂P/∂u: tangent in u-direction (horizontal parametric direction)
        - ∂P/∂v: tangent in v-direction (vertical parametric direction)
        
        For perspective transformations, these vectors vary across the quad due to the 
        nonlinear nature of the transformation. The vectors are NOT normalized - their 
        magnitudes represent local stretching in each parametric direction.
        
        Users can normalize them if needed using np.linalg.norm().
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_projective_interpolation_quad(quad, uv)
    if cartesian_coords is None:
        return None, None, None
    
    # Handle both single point and batch of points
    uv = np.atleast_2d(uv)
    
    # Define unit square as source UV space
    unit_square = np.array([
        [0, 0],  # bottom-left
        [1, 0],  # bottom-right
        [0, 1],  # top-left
        [1, 1]   # top-right
    ])
    
    # Get transformation matrix from unit square to quad
    T = getPerspectiveTransform(unit_square, quad)
    
    # Compute Jacobian for each point
    # For perspective transform: P = T * [u, v, 1]^T
    # Where result is [x', y', w']^T and final point is [x'/w', y'/w']
    
    num_points = uv.shape[0]
    
    # Convert to homogeneous coordinates
    homogeneous = np.ones((num_points, 3))
    homogeneous[:, :2] = uv
    
    # Apply transformation
    transformed = (T @ homogeneous.T).T  # Shape (x, 3)
    
    x_prime = transformed[:, 0]  # Shape (x,)
    y_prime = transformed[:, 1]  # Shape (x,)
    w_prime = transformed[:, 2]  # Shape (x,)
    
    # Compute partial derivatives
    # ∂x'/∂u = T[0,0], ∂x'/∂v = T[0,1]
    # ∂y'/∂u = T[1,0], ∂y'/∂v = T[1,1]
    # ∂w'/∂u = T[2,0], ∂w'/∂v = T[2,1]
    
    dx_prime_du = T[0, 0]
    dx_prime_dv = T[0, 1]
    dy_prime_du = T[1, 0]
    dy_prime_dv = T[1, 1]
    dw_prime_du = T[2, 0]
    dw_prime_dv = T[2, 1]
    
    # Apply quotient rule: ∂(x'/w')/∂u = (∂x'/∂u * w' - x' * ∂w'/∂u) / w'^2
    w_prime_sq = w_prime * w_prime  # Shape (x,)
    
    # Tangent in u-direction
    dx_du = (dx_prime_du * w_prime - x_prime * dw_prime_du) / w_prime_sq  # Shape (x,)
    dy_du = (dy_prime_du * w_prime - y_prime * dw_prime_du) / w_prime_sq  # Shape (x,)
    tangent_u = np.stack([dx_du, dy_du], axis=1)  # Shape (x, 2)
    
    # Tangent in v-direction
    dx_dv = (dx_prime_dv * w_prime - x_prime * dw_prime_dv) / w_prime_sq  # Shape (x,)
    dy_dv = (dy_prime_dv * w_prime - y_prime * dw_prime_dv) / w_prime_sq  # Shape (x,)
    tangent_v = np.stack([dx_dv, dy_dv], axis=1)  # Shape (x, 2)
    
    return cartesian_coords, tangent_u, tangent_v