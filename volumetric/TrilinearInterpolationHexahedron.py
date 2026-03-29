import numpy as np

def trilinear_interpolation_hexahedron(hexahedral_prism: np.array, cartesian_coordinates: np.array) -> np.array:
    """
    Convert Cartesian coordinates to UVW coordinates within a hexahedral prism using trilinear interpolation.
    
    This function finds the normalized UVW coordinates (where u,v,w ∈ [0,1]) that correspond to
    given Cartesian positions within the hexahedral element. Uses Newton-Raphson iteration to solve
    the inverse trilinear interpolation problem.
    
    Args:
        hexahedral_prism (np.array): Shape (8, 3) - Eight vertices of the hexahedral prism in order:
                            [v000, v100, v010, v110, v001, v101, v011, v111]
                            where vijk corresponds to u=i, v=j, w=k in parametric space
        coordinate (np.array): Shape (x, 3) - Cartesian coordinates of the points to convert
                            where x can be any number of points
    
    Returns:
        np.array: Shape (x, 3) - UVW coordinates where [0,0,0] corresponds to v000 vertex
                        and [1,1,1] corresponds to v111 vertex, or None if input validation fails
    
    UVW Coordinate System:
        - U axis (first coordinate): varies from 0 to 1 across the hexahedron's first parametric direction
            u=0 corresponds to vertices v000, v010, v001, v011 (left face)
            u=1 corresponds to vertices v100, v110, v101, v111 (right face)
        - V axis (second coordinate): varies from 0 to 1 across the hexahedron's second parametric direction
            v=0 corresponds to vertices v000, v100, v001, v101 (bottom face)
            v=1 corresponds to vertices v010, v110, v011, v111 (top face)
        - W axis (third coordinate): varies from 0 to 1 across the hexahedron's third parametric direction
            w=0 corresponds to vertices v000, v100, v010, v110 (back face)
            w=1 corresponds to vertices v001, v101, v011, v111 (front face)
    
    Note:
        Uses Newton-Raphson iteration to solve the inverse trilinear interpolation problem.
        Initial guess is the center of the parametric space [0.5, 0.5, 0.5].
    """
    
    # Validate hexahedral_prism shape
    if hexahedral_prism.shape != (8, 3):
        return None
    
    # Handle both single point and batch of points
    cartesian_coordinates = np.atleast_2d(cartesian_coordinates)
    if cartesian_coordinates.shape[1] != 3:
        return None
    
    num_points = cartesian_coordinates.shape[0]
    uvw = np.zeros((num_points, 3))
    
    # Newton-Raphson iteration for each point
    for i in range(num_points):
        target = cartesian_coordinates[i]
        
        # Initial guess - center of parametric space
        uvw_current = np.array([0.5, 0.5, 0.5])
        
        # Newton-Raphson iteration
        max_iterations = 20
        tolerance = 1e-10
        
        for _ in range(max_iterations):
            
            # Compute trilinear interpolation at current UVW
            interpolated = _trilinear_forward(hexahedral_prism, uvw_current)
            
            # Residual vector
            residual = interpolated - target
            
            # Check convergence
            if np.linalg.norm(residual) < tolerance:
                break
            
            # Compute Jacobian matrix (3x3)
            jacobian = _compute_jacobian(hexahedral_prism, uvw_current)
            
            # Solve linear system: J * delta_uvw = -residual
            try:
                delta_uvw = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                # If Jacobian is singular, use pseudo-inverse
                delta_uvw = np.linalg.pinv(jacobian) @ (-residual)
            
            # Update UVW coordinates
            uvw_current += delta_uvw
            
            # Clamp to valid range [0, 1]
            uvw_current = np.clip(uvw_current, 0.0, 1.0)
        
        uvw[i] = uvw_current
    
    return uvw


def _trilinear_forward(hexahedral_prism: np.array, uvw: np.array) -> np.array:
    """
    Forward trilinear interpolation - compute XYZ from UVW coordinates.
    
    Args:
        hexahedral_prism (np.array): Shape (8, 3) - Eight vertices of the hexahedral prism
        uvw (np.array): Shape (3,) - UVW parametric coordinates
    
    Returns:
        np.array: Shape (3,) - Interpolated XYZ coordinates
    """
    u, v, w = uvw
    
    # Trilinear interpolation weights
    w000 = (1-u) * (1-v) * (1-w)
    w100 = u * (1-v) * (1-w)
    w010 = (1-u) * v * (1-w)
    w110 = u * v * (1-w)
    w001 = (1-u) * (1-v) * w
    w101 = u * (1-v) * w
    w011 = (1-u) * v * w
    w111 = u * v * w
    
    # Weighted sum of vertices
    result = (w000 * hexahedral_prism[0] + w100 * hexahedral_prism[1] + w010 * hexahedral_prism[2] + 
              w110 * hexahedral_prism[3] + w001 * hexahedral_prism[4] + w101 * hexahedral_prism[5] + 
              w011 * hexahedral_prism[6] + w111 * hexahedral_prism[7])
    
    return result


def _compute_jacobian(hexahedral_prism: np.array, uvw: np.array) -> np.array:
    """
    Compute the Jacobian matrix for trilinear interpolation.
    
    Args:
        hexahedral_prism (np.array): Shape (8, 3) - Eight vertices of the hexahedral prism
        uvw (np.array): Shape (3,) - UVW parametric coordinates
    
    Returns:
        np.array: Shape (3, 3) - Jacobian matrix [dX/du, dX/dv, dX/dw; dY/du, dY/dv, dY/dw; dZ/du, dZ/dv, dZ/dw]
    """
    u, v, w = uvw
    
    # Partial derivatives of shape functions with respect to u, v, w
    # dN/du
    dNdu = np.array([
        -(1-v)*(1-w),  (1-v)*(1-w), -(v)*(1-w),  (v)*(1-w),
        -(1-v)*(w),    (1-v)*(w),   -(v)*(w),    (v)*(w)
    ])
    
    # dN/dv  
    dNdv = np.array([
        -(1-u)*(1-w), -(u)*(1-w),  (1-u)*(1-w),  (u)*(1-w),
        -(1-u)*(w),   -(u)*(w),    (1-u)*(w),    (u)*(w)
    ])
    
    # dN/dw
    dNdw = np.array([
        -(1-u)*(1-v), -(u)*(1-v), -(1-u)*(v), -(u)*(v),
        (1-u)*(1-v),  (u)*(1-v),  (1-u)*(v),  (u)*(v)
    ])
    
    # Compute Jacobian: J[i][j] = sum(dN[k]/d[j] * vertex[k][i])
    # Vectorized computation: matrix multiply shape functions derivatives with vertex coordinates
    # dN is shape (8,), hexahedral_prism is shape (8, 3)
    # Result: shape (3,) for each partial derivative
    jacobian = np.column_stack([
        hexahedral_prism.T @ dNdu,  # dX/du, dY/du, dZ/du - Shape (3,)
        hexahedral_prism.T @ dNdv,  # dX/dv, dY/dv, dZ/dv - Shape (3,)
        hexahedral_prism.T @ dNdw   # dX/dw, dY/dw, dZ/dw - Shape (3,)
    ])  # Result shape: (3, 3)
    
    return jacobian

def reverse_trilinear_interpolation_hexahedron(hexahedral_prism: np.array, uvw: np.array) -> np.array:
    """
    Convert UVW coordinates to Cartesian coordinates within a hexahedral prism using trilinear interpolation.
    
    This function maps normalized UVW coordinates (where u,v,w ∈ [0,1]) to the corresponding 
    Cartesian positions within the hexahedral element defined by eight vertices. The mapping uses
    trilinear interpolation which preserves straight lines parallel to the UVW axes.
    
    Args:
        hexahedral_prism (np.array): Shape (8, 3) - Eight vertices of the hexahedral prism in order:
                                [v000, v100, v010, v110, v001, v101, v011, v111]
                                where vijk corresponds to u=i, v=j, w=k in parametric space
        uvw (np.array): Shape (x, 3) - UVW coordinates where [0,0,0] maps to v000 vertex
                        and [1,1,1] maps to v111 vertex, where x can be any number of points
    
    Returns:
        np.array: Shape (x, 3) - Cartesian coordinates of the interpolated points,
                        or None if input validation fails
    
    UVW Coordinate System:
        - U axis (first coordinate): varies from 0 to 1 across the hexahedron's first parametric direction
            u=0 corresponds to vertices v000, v010, v001, v011 (left face)
            u=1 corresponds to vertices v100, v110, v101, v111 (right face)
        - V axis (second coordinate): varies from 0 to 1 across the hexahedron's second parametric direction
            v=0 corresponds to vertices v000, v100, v001, v101 (bottom face)
            v=1 corresponds to vertices v010, v110, v011, v111 (top face)
        - W axis (third coordinate): varies from 0 to 1 across the hexahedron's third parametric direction
            w=0 corresponds to vertices v000, v100, v010, v110 (back face)
            w=1 corresponds to vertices v001, v101, v011, v111 (front face)
    
    Note:
        The UVW coordinates are applied using trilinear interpolation weights to blend between
        the eight hexahedral vertices, preserving straight lines in parametric space.
    """
    
    # Validate hexahedral_prism shape
    if hexahedral_prism.shape != (8, 3):
        return None
    
    # Handle both single point and batch of points
    uvw = np.atleast_2d(uvw)
    if uvw.shape[1] != 3:
        return None
    
    # Vectorized trilinear interpolation for all points
    u = uvw[:, 0:1]  # Shape (x, 1)
    v = uvw[:, 1:2]  # Shape (x, 1)
    w = uvw[:, 2:3]  # Shape (x, 1)
    
    # Compute trilinear interpolation weights (shape (x, 1) for each)
    w000 = (1-u) * (1-v) * (1-w)
    w100 = u * (1-v) * (1-w)
    w010 = (1-u) * v * (1-w)
    w110 = u * v * (1-w)
    w001 = (1-u) * (1-v) * w
    w101 = u * (1-v) * w
    w011 = (1-u) * v * w
    w111 = u * v * w
    
    # Vectorized weighted sum: each weight multiplies all 3 coordinates of corresponding vertex
    xyz = (w000 * hexahedral_prism[0] + w100 * hexahedral_prism[1] + w010 * hexahedral_prism[2] + 
           w110 * hexahedral_prism[3] + w001 * hexahedral_prism[4] + w101 * hexahedral_prism[5] + 
           w011 * hexahedral_prism[6] + w111 * hexahedral_prism[7])
    
    return xyz


def reverse_trilinear_interpolation_hexahedron_with_tangents(hexahedral_prism: np.array, uvw: np.array) -> tuple:
    """
    Convert UVW coordinates to Cartesian coordinates and compute tangent vectors.
    
    This function maps normalized UVW coordinates (where u,v,w ∈ [0,1]) to the corresponding 
    Cartesian positions within the hexahedral element, and also computes the tangent vectors 
    in the u, v, and w parametric directions at each point.
    
    Args:
        hexahedral_prism (np.array): Shape (8, 3) - Eight vertices of the hexahedral prism in order:
                                [v000, v100, v010, v110, v001, v101, v011, v111]
                                where vijk corresponds to u=i, v=j, w=k in parametric space
        uvw (np.array): Shape (x, 3) - UVW coordinates where [0,0,0] maps to v000 vertex
                        and [1,1,1] maps to v111 vertex, where x can be any number of points
    
    Returns:
        tuple: (cartesian_coords, tangent_u, tangent_v, tangent_w) where:
            - cartesian_coords: Shape (x, 3) - Cartesian coordinates of interpolated points
            - tangent_u: Shape (x, 3) - Tangent vectors in u-direction at each point (unnormalized)
            - tangent_v: Shape (x, 3) - Tangent vectors in v-direction at each point (unnormalized)
            - tangent_w: Shape (x, 3) - Tangent vectors in w-direction at each point (unnormalized)
            Returns (None, None, None, None) if input validation fails
    
    UVW Coordinate System:
        - U axis (first coordinate): varies from 0 to 1 across the hexahedron's first parametric direction
            u=0 corresponds to vertices v000, v010, v001, v011 (left face)
            u=1 corresponds to vertices v100, v110, v101, v111 (right face)
        - V axis (second coordinate): varies from 0 to 1 across the hexahedron's second parametric direction
            v=0 corresponds to vertices v000, v100, v001, v101 (bottom face)
            v=1 corresponds to vertices v010, v110, v011, v111 (top face)
        - W axis (third coordinate): varies from 0 to 1 across the hexahedron's third parametric direction
            w=0 corresponds to vertices v000, v100, v010, v110 (back face)
            w=1 corresponds to vertices v001, v101, v011, v111 (front face)
    
    Note:
        The tangent vectors are computed as partial derivatives of the trilinear interpolation:
        - ∂P/∂u: tangent in u-direction (first parametric direction)
        - ∂P/∂v: tangent in v-direction (second parametric direction)
        - ∂P/∂w: tangent in w-direction (third parametric direction)
        
        These vectors are position-dependent and vary across the hexahedron. The vectors are 
        NOT normalized - their magnitudes represent local stretching in each parametric direction.
        Users can normalize them if needed using np.linalg.norm().
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_trilinear_interpolation_hexahedron(hexahedral_prism, uvw)
    if cartesian_coords is None:
        return None, None, None, None
    
    # Handle both single point and batch of points
    uvw = np.atleast_2d(uvw)
    
    # Extract UVW components
    u = uvw[:, 0]  # Shape (x,)
    v = uvw[:, 1]  # Shape (x,)
    w = uvw[:, 2]  # Shape (x,)
    
    # Vectorized computation of shape function derivatives for all points
    # Each dN array has shape (x, 8) where x is num_points
    dNdu = np.stack([
        -(1-v)*(1-w),  (1-v)*(1-w), -(v)*(1-w),  (v)*(1-w),
        -(1-v)*(w),    (1-v)*(w),   -(v)*(w),    (v)*(w)
    ], axis=1)  # Shape (x, 8)
    
    dNdv = np.stack([
        -(1-u)*(1-w), -(u)*(1-w),  (1-u)*(1-w),  (u)*(1-w),
        -(1-u)*(w),   -(u)*(w),    (1-u)*(w),    (u)*(w)
    ], axis=1)  # Shape (x, 8)
    
    dNdw = np.stack([
        -(1-u)*(1-v), -(u)*(1-v), -(1-u)*(v), -(u)*(v),
        (1-u)*(1-v),  (u)*(1-v),  (1-u)*(v),  (u)*(v)
    ], axis=1)  # Shape (x, 8)
    
    # Vectorized Jacobian computation: dN @ vertices
    # dNdu shape (x, 8) @ hexahedral_prism shape (8, 3) = shape (x, 3)
    tangent_u = dNdu @ hexahedral_prism  # Shape (x, 3)
    tangent_v = dNdv @ hexahedral_prism  # Shape (x, 3)
    tangent_w = dNdw @ hexahedral_prism  # Shape (x, 3)
    
    return cartesian_coords, tangent_u, tangent_v, tangent_w