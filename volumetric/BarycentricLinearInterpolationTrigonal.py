import numpy as np


def barycentric_linear_interpolation_trigonal(triangular_prism: np.array, cartesian_coordinates: np.array) -> np.array:
    """
    Convert Cartesian coordinates to barycentric-linear coordinates within a triangular prism.
    
    This function finds the parametric coordinates that correspond to given Cartesian positions
    within a triangular prism. The parametric space consists of:
    - Barycentric coordinates (alpha, beta, gamma) for the triangular cross-section
    - Linear parameter t for position between bottom and top triangles
    
    Args:
        triangular_prism (np.array): Shape (6, 3) - Six vertices of the triangular prism in order:
                                        [v0_bottom, v1_bottom, v2_bottom, v0_top, v1_top, v2_top]
                                        where the first 3 vertices define the bottom triangle
                                        and the last 3 vertices define the top triangle
        cartesian_coordinates (np.array): Shape (x, 3) - Cartesian coordinates of points to convert
                                            where x can be any number of points
    
    Returns:
        np.array: Shape (x, 4) - Parametric coordinates [alpha, beta, gamma, t]
            where alpha + beta + gamma = 1 for each point, and t is the linear parameter
            where t=0 corresponds to bottom triangle and t=1 corresponds to top triangle
    
    Parametric Coordinate System:
        - Alpha, Beta, Gamma: barycentric coordinates within the triangular cross-section
          * alpha corresponds to v0 (first vertex of triangles)
          * beta corresponds to v1 (second vertex of triangles)
          * gamma corresponds to v2 (third vertex of triangles)
          * Constraint: alpha + beta + gamma = 1
        - t: linear parameter along prism axis
          * t=0: point lies on bottom triangle (vertices 0-2)
          * t=1: point lies on top triangle (vertices 3-5)
    
    Note:
        Uses Newton-Raphson iteration to solve the inverse interpolation problem.
        Initial guess is the center of the parametric space [1/3, 1/3, 1/3, 0.5].
    """
    
    # Validate triangular_prism shape
    if triangular_prism.shape != (6, 3):
        return None
    
    # Handle both single point and batch of points
    cartesian_coordinates = np.atleast_2d(cartesian_coordinates)
    if cartesian_coordinates.shape[1] != 3:
        return None
    
    num_points = cartesian_coordinates.shape[0]
    parametric_coords = np.zeros((num_points, 4))
    
    # Newton-Raphson iteration for each point
    for i in range(num_points):
        target = cartesian_coordinates[i]
        
        # Initial guess - center of parametric space
        # [alpha, beta, gamma, t] but we'll work with [alpha, beta, t] since gamma = 1 - alpha - beta
        params = np.array([1/3, 1/3, 0.5])  # [alpha, beta, t]
        
        # Newton-Raphson iteration
        max_iterations = 50
        tolerance = 1e-10
        
        for _ in range(max_iterations):
            alpha, beta, t_current = params
            gamma = 1.0 - alpha - beta
            
            # Compute forward interpolation at current parameters
            interpolated = _barycentric_linear_forward(triangular_prism, alpha, beta, gamma, t_current)
            
            # Residual vector
            residual = interpolated - target
            
            # Check convergence
            if np.linalg.norm(residual) < tolerance:
                break
            
            # Compute Jacobian matrix (3x3)
            jacobian = _compute_jacobian_prism(triangular_prism, alpha, beta, gamma, t_current)
            
            # Solve linear system: J * delta_params = -residual
            try:
                delta_params = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                # If Jacobian is singular, use pseudo-inverse
                delta_params = np.linalg.pinv(jacobian) @ (-residual)
            
            # Update parameters
            params += delta_params
        
        # Extract final parameters
        alpha, beta, t_current = params
        gamma = 1.0 - alpha - beta
        
        parametric_coords[i] = [alpha, beta, gamma, t_current]
    
    return parametric_coords


def _barycentric_linear_forward(triangular_prism: np.array, alpha: float, beta: float, gamma: float, t: float) -> np.array:
    """
    Forward barycentric-linear interpolation - compute XYZ from parametric coordinates.
    
    Args:
        triangular_prism (np.array): Shape (6, 3) - Six vertices of the triangular prism
        alpha (float): First barycentric coordinate
        beta (float): Second barycentric coordinate
        gamma (float): Third barycentric coordinate
        t (float): Linear parameter between bottom (t=0) and top (t=1)
    
    Returns:
        np.array: Shape (3,) - Interpolated XYZ coordinates
    """
    # Bottom triangle vertices (indices 0-2)
    bottom_point = alpha * triangular_prism[0] + beta * triangular_prism[1] + gamma * triangular_prism[2]
    
    # Top triangle vertices (indices 3-5)
    top_point = alpha * triangular_prism[3] + beta * triangular_prism[4] + gamma * triangular_prism[5]
    
    # Linear interpolation between bottom and top
    return (1 - t) * bottom_point + t * top_point


def _compute_jacobian_prism(triangular_prism: np.array, alpha: float, beta: float, gamma: float, t: float) -> np.array:
    """
    Compute the Jacobian matrix for barycentric-linear interpolation.
    
    The Jacobian maps changes in parametric coordinates (alpha, beta, t) to changes in
    Cartesian coordinates (x, y, z). Note: gamma is dependent on alpha and beta.
    
    Args:
        triangular_prism (np.array): Shape (6, 3) - Six vertices of the triangular prism
        alpha (float): First barycentric coordinate
        beta (float): Second barycentric coordinate
        gamma (float): Third barycentric coordinate
        t (float): Linear parameter
    
    Returns:
        np.array: Shape (3, 3) - Jacobian matrix [dX/dalpha, dX/dbeta, dX/dt;
                                                    dY/dalpha, dY/dbeta, dY/dt;
                                                    dZ/dalpha, dZ/dbeta, dZ/dt]
    """
    v0_bottom, v1_bottom, v2_bottom = triangular_prism[0], triangular_prism[1], triangular_prism[2]
    v0_top, v1_top, v2_top = triangular_prism[3], triangular_prism[4], triangular_prism[5]
    
    # Partial derivative with respect to alpha
    # d/dalpha = (1-t) * (v0_bottom - v2_bottom) + t * (v0_top - v2_top)
    # (since gamma = 1 - alpha - beta, d_gamma/d_alpha = -1)
    d_alpha = (1 - t) * (v0_bottom - v2_bottom) + t * (v0_top - v2_top)
    
    # Partial derivative with respect to beta
    # d/dbeta = (1-t) * (v1_bottom - v2_bottom) + t * (v1_top - v2_top)
    # (since gamma = 1 - alpha - beta, d_gamma/d_beta = -1)
    d_beta = (1 - t) * (v1_bottom - v2_bottom) + t * (v1_top - v2_top)
    
    # Partial derivative with respect to t
    # d/dt = (alpha * v0_top + beta * v1_top + gamma * v2_top) - (alpha * v0_bottom + beta * v1_bottom + gamma * v2_bottom)
    bottom_point = alpha * v0_bottom + beta * v1_bottom + gamma * v2_bottom
    top_point = alpha * v0_top + beta * v1_top + gamma * v2_top
    d_t = top_point - bottom_point
    
    # Construct Jacobian: each column is a partial derivative vector
    jacobian = np.column_stack([d_alpha, d_beta, d_t])
    
    return jacobian


def reverse_barycentric_linear_interpolation_trigonal(triangular_prism: np.array, parametric_coordinates: np.array) -> np.array:
    """
    Convert barycentric-linear coordinates to Cartesian coordinates within a triangular prism.
    
    This function maps parametric coordinates to the corresponding Cartesian positions within
    a triangular prism. The parametric space consists of barycentric coordinates for the
    triangular cross-section and a linear parameter for position along the prism axis.
    
    Args:
        triangular_prism (np.array): Shape (6, 3) - Six vertices of the triangular prism in order:
                                        [v0_bottom, v1_bottom, v2_bottom, v0_top, v1_top, v2_top]
        parametric_coordinates (np.array): Shape (x, 4) - Parametric coordinates [alpha, beta, gamma, t]
                                where alpha + beta + gamma = 1 for each point, and t is the linear
                                parameter where t=0 maps to bottom triangle and t=1 maps to top triangle
    
    Returns:
        np.array: Shape (x, 3) - Cartesian coordinates of the interpolated points,
                    or None if input validation fails
    
    Parametric Coordinate System:
        - Alpha, Beta, Gamma: barycentric coordinates within the triangular cross-section
          * alpha corresponds to v0 (first vertex of triangles)
          * beta corresponds to v1 (second vertex of triangles)
          * gamma corresponds to v2 (third vertex of triangles)
          * Constraint: alpha + beta + gamma = 1
        - t: linear parameter along prism axis
          * t=0: point lies on bottom triangle (vertices 0-2)
          * t=1: point lies on top triangle (vertices 3-5)
    
    Note:
        The interpolation combines barycentric interpolation within the triangular
        cross-section with linear interpolation along the prism axis.
    """
    
    # Validate triangular_prism shape
    if triangular_prism.shape != (6, 3):
        return None
    
    # Handle both single point and batch of points
    parametric_coordinates = np.atleast_2d(parametric_coordinates)
    
    if parametric_coordinates.shape[1] != 4:
        return None
    
    # Extract barycentric and linear coordinates
    alpha = parametric_coordinates[:, 0:1]  # Shape (x, 1)
    beta = parametric_coordinates[:, 1:2]   # Shape (x, 1)
    gamma = parametric_coordinates[:, 2:3]  # Shape (x, 1)
    t = parametric_coordinates[:, 3:4]      # Shape (x, 1)
    
    # Extract bottom and top triangle vertices
    v0_bottom, v1_bottom, v2_bottom = triangular_prism[0], triangular_prism[1], triangular_prism[2]
    v0_top, v1_top, v2_top = triangular_prism[3], triangular_prism[4], triangular_prism[5]
    
    # Vectorized barycentric-linear interpolation for all points
    # Interpolate on bottom triangle
    bottom_point = alpha * v0_bottom + beta * v1_bottom + gamma * v2_bottom  # Shape (x, 3)
    
    # Interpolate on top triangle
    top_point = alpha * v0_top + beta * v1_top + gamma * v2_top  # Shape (x, 3)
    
    # Linear interpolation between bottom and top
    xyz = (1 - t) * bottom_point + t * top_point  # Shape (x, 3)
    
    return xyz


def reverse_barycentric_linear_interpolation_trigonal_with_tangents(triangular_prism: np.array, parametric_coordinates: np.array) -> tuple:
    """
    Convert barycentric-linear coordinates to Cartesian coordinates and compute direction vectors.
    
    This function maps parametric coordinates to the corresponding Cartesian positions within
    a triangular prism, and also computes direction vectors in the barycentric coordinate system
    and along the linear axis. These vectors represent the principal directions within the prism.
    
    Args:
        triangular_prism (np.array): Shape (6, 3) - Six vertices of the triangular prism in order:
                                        [v0_bottom, v1_bottom, v2_bottom, v0_top, v1_top, v2_top]
        parametric_coordinates (np.array): Shape (x, 4) - Parametric coordinates [alpha, beta, gamma, t]
                                where alpha + beta + gamma = 1 for each point, and t is the linear
                                parameter where t=0 maps to bottom triangle and t=1 maps to top triangle
    
    Returns:
        tuple: (cartesian_coords, direction_to_v0, direction_to_v1, direction_to_v2, tangent_t) where:
            - cartesian_coords: Shape (x, 3) - Cartesian coordinates of interpolated points
            - direction_to_v0: Shape (x, 3) - Direction vectors to v0 at current t-level (unnormalized)
            - direction_to_v1: Shape (x, 3) - Direction vectors to v1 at current t-level (unnormalized)
            - direction_to_v2: Shape (x, 3) - Direction vectors to v2 at current t-level (unnormalized)
            - tangent_t: Shape (x, 3) - Tangent vectors along prism axis (t-direction, unnormalized)
            Returns (None, None, None, None, None) if input validation fails
    
    Parametric Coordinate System:
        - Alpha, Beta, Gamma: barycentric coordinates within the triangular cross-section
          * alpha corresponds to v0 (first vertex of triangles)
          * beta corresponds to v1 (second vertex of triangles)
          * gamma corresponds to v2 (third vertex of triangles)
          * Constraint: alpha + beta + gamma = 1
        - t: linear parameter along prism axis
          * t=0: point lies on bottom triangle (vertices 0-2)
          * t=1: point lies on top triangle (vertices 3-5)
    
    Note:
        The direction vectors represent:
        - direction_to_v0/v1/v2: Point from the "virtual" vertices at the current t-level
            toward each position. These virtual vertices are linearly interpolated between bottom and top.
          v_i(t) = (1-t) * v_i_bottom + t * v_i_top
        - tangent_t: The tangent vector along the prism axis (∂P/∂t), showing the direction of
            movement when t increases. This is the difference between the top and bottom points
            at the same barycentric coordinates.
        
        All vectors are position-dependent and unnormalized. Their magnitudes provide information
        about distances and rates of change. Users can normalize them using np.linalg.norm().
        
        Properties:
        - At a vertex on bottom (t=0), the corresponding barycentric direction vector is zero
        - At a vertex on top (t=1), the corresponding barycentric direction vector is zero
        - The barycentric directions satisfy: alpha * dir_v0 + beta * dir_v1 + gamma * dir_v2 = 0
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_barycentric_linear_interpolation_trigonal(triangular_prism, parametric_coordinates)
    if cartesian_coords is None:
        return None, None, None, None, None
    
    # Handle both single point and batch of points
    parametric_coordinates = np.atleast_2d(parametric_coordinates)
    
    # Extract barycentric and linear coordinates
    alpha = parametric_coordinates[:, 0:1]  # Shape (x, 1)
    beta = parametric_coordinates[:, 1:2]   # Shape (x, 1)
    gamma = parametric_coordinates[:, 2:3]  # Shape (x, 1)
    t = parametric_coordinates[:, 3:4]      # Shape (x, 1)
    
    # Extract bottom and top triangle vertices
    v0_bottom, v1_bottom, v2_bottom = triangular_prism[0], triangular_prism[1], triangular_prism[2]
    v0_top, v1_top, v2_top = triangular_prism[3], triangular_prism[4], triangular_prism[5]
    
    # Vectorized computation of "virtual" vertices at current t-levels
    # These are the vertices of the triangular cross-section at parameter t
    v0_at_t = (1 - t) * v0_bottom + t * v0_top  # Shape (x, 3)
    v1_at_t = (1 - t) * v1_bottom + t * v1_top  # Shape (x, 3)
    v2_at_t = (1 - t) * v2_bottom + t * v2_top  # Shape (x, 3)
    
    # Direction vectors from virtual vertices to current points
    direction_to_v0 = cartesian_coords - v0_at_t  # Shape (x, 3)
    direction_to_v1 = cartesian_coords - v1_at_t  # Shape (x, 3)
    direction_to_v2 = cartesian_coords - v2_at_t  # Shape (x, 3)
    
    # Tangent along t-axis (from Jacobian)
    # This is ∂P/∂t = top_point - bottom_point
    bottom_point = alpha * v0_bottom + beta * v1_bottom + gamma * v2_bottom  # Shape (x, 3)
    top_point = alpha * v0_top + beta * v1_top + gamma * v2_top  # Shape (x, 3)
    tangent_t = top_point - bottom_point  # Shape (x, 3)
    
    return cartesian_coords, direction_to_v0, direction_to_v1, direction_to_v2, tangent_t
