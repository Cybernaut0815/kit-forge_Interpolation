import numpy as np

def trilinear_interpolation_pentahedron(pentahedron: np.array, cartesian_coordinates: np.array) -> np.array:
    """
    Convert Cartesian coordinates to UVW coordinates within a pentahedron.
    
    This function finds the normalized UVW coordinates (where u,v,w ∈ [0,1]) that correspond to
    given Cartesian positions within the pyramid. Uses Newton-Raphson iteration to solve
    the inverse interpolation problem.
    
    Args:
        pentahedron (np.array): Shape (5, 3) - Five vertices of the pentahedron in order:
                                [v00, v10, v01, v11, apex]
                                where v00, v10, v01, v11 form the base quadrilateral
                                and apex is the pyramid tip
        cartesian_coordinates (np.array): Shape (x, 3) - Cartesian coordinates of points to convert
                                            where x can be any number of points
    
    Returns:
        np.array: Shape (x, 3) - UVW coordinates where:
                    - UV interpolates within the base quadrilateral (u,v ∈ [0,1])
                    - W interpolates from base (w=0) to apex (w=1)
                    Returns None if input validation fails
    
    UVW Coordinate System:
        - U axis: varies from 0 to 1 across the base's first parametric direction
            u=0 corresponds to base vertices v00, v01 (left edge)
            u=1 corresponds to base vertices v10, v11 (right edge)
        - V axis: varies from 0 to 1 across the base's second parametric direction
            v=0 corresponds to base vertices v00, v10 (bottom edge)
            v=1 corresponds to base vertices v01, v11 (top edge)
        - W axis: varies from 0 to 1 from base to apex
            w=0 corresponds to the base quadrilateral
            w=1 corresponds to the apex point
    
    Note:
        Uses Newton-Raphson iteration to solve the inverse interpolation problem.
        Initial guess is the center of the parametric space [0.5, 0.5, 0.5].
    """
    
    # Validate pentahedron shape
    if pentahedron.shape != (5, 3):
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
            
            # Compute forward interpolation at current UVW
            interpolated = _pentahedron_forward(pentahedron, uvw_current)
            
            # Residual vector
            residual = interpolated - target
            
            # Check convergence
            if np.linalg.norm(residual) < tolerance:
                break
            
            # Compute Jacobian matrix (3x3)
            jacobian = _compute_pentahedron_jacobian(pentahedron, uvw_current)
            
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


def _pentahedron_forward(pentahedron: np.array, uvw: np.array) -> np.array:
    """
    Forward interpolation for pentahedron - compute XYZ from UVW coordinates.
    
    Args:
        pentahedron (np.array): Shape (5, 3) - Five vertices: [v00, v10, v01, v11, apex]
        uvw (np.array): Shape (3,) - UVW parametric coordinates
    
    Returns:
        np.array: Shape (3,) - Interpolated XYZ coordinates
    """
    u, v, w = uvw
    
    # First, bilinearly interpolate the base quadrilateral
    base_weights = np.array([
        (1-u) * (1-v),  # v00
        u * (1-v),      # v10
        (1-u) * v,      # v01
        u * v           # v11
    ])
    
    base_point = np.sum(base_weights[:, np.newaxis] * pentahedron[:4], axis=0)
    
    # Then linearly interpolate between base point and apex using w
    apex = pentahedron[4]
    result = (1-w) * base_point + w * apex
    
    return result


def _compute_pentahedron_jacobian(pentahedron: np.array, uvw: np.array) -> np.array:
    """
    Compute the Jacobian matrix for pentahedron interpolation.
    
    Args:
        pentahedron (np.array): Shape (5, 3) - Five vertices: [v00, v10, v01, v11, apex]
        uvw (np.array): Shape (3,) - UVW parametric coordinates
    
    Returns:
        np.array: Shape (3, 3) - Jacobian matrix [dX/du, dX/dv, dX/dw; dY/du, dY/dv, dY/dw; dZ/du, dZ/dv, dZ/dw]
    """
    u, v, w = uvw
    
    # Extract base vertices and apex
    v00, v10, v01, v11, apex = pentahedron[0], pentahedron[1], pentahedron[2], pentahedron[3], pentahedron[4]
    
    # Partial derivatives with respect to u, v, w
    # The interpolation is: P = (1-w) * [bilinear_base(u,v)] + w * apex
    
    # Compute base point and its derivatives
    base_point = (1-u)*(1-v)*v00 + u*(1-v)*v10 + (1-u)*v*v01 + u*v*v11
    
    # dP/du = (1-w) * d[base]/du
    dbase_du = -(1-v)*v00 + (1-v)*v10 - v*v01 + v*v11
    dP_du = (1-w) * dbase_du
    
    # dP/dv = (1-w) * d[base]/dv  
    dbase_dv = -(1-u)*v00 - u*v10 + (1-u)*v01 + u*v11
    dP_dv = (1-w) * dbase_dv
    
    # dP/dw = apex - base_point (points from base toward apex as w increases)
    dP_dw = apex - base_point
    
    # Assemble Jacobian matrix
    jacobian = np.column_stack([dP_du, dP_dv, dP_dw])
    
    return jacobian


def reverse_trilinear_interpolation_pentahedron(pentahedron: np.array, uvw: np.array) -> np.array:
    """
    Convert UVW coordinates to Cartesian coordinates within a pentahedron.
    
    This function maps normalized UVW coordinates (where u,v,w ∈ [0,1]) to the corresponding 
    Cartesian positions within the pyramid. The UV coordinates interpolate bilinearly within
    the base quadrilateral, and W interpolates linearly from the base to the apex.
    
    Args:
        pentahedron (np.array): Shape (5, 3) - Five vertices of the pentahedron in order:
                                [v00, v10, v01, v11, apex]
                                where v00, v10, v01, v11 form the base quadrilateral
                                and apex is the pyramid tip
        uvw (np.array): Shape (x, 3) - UVW coordinates where:
                        - UV interpolates within the base quadrilateral (u,v ∈ [0,1])
                        - W interpolates from base (w=0) to apex (w=1)
                        where x can be any number of points
    
    Returns:
        np.array: Shape (x, 3) - Cartesian coordinates of the interpolated points,
                    or None if input validation fails
    
    UVW Coordinate System:
        - U axis: varies from 0 to 1 across the base's first parametric direction
            u=0 corresponds to base vertices v00, v01 (left edge)
            u=1 corresponds to base vertices v10, v11 (right edge)
        - V axis: varies from 0 to 1 across the base's second parametric direction
            v=0 corresponds to base vertices v00, v10 (bottom edge)
            v=1 corresponds to base vertices v01, v11 (top edge)
        - W axis: varies from 0 to 1 from base to apex
            w=0 corresponds to the base quadrilateral
            w=1 corresponds to the apex point
    
    Note:
        The interpolation first computes a point on the base using bilinear interpolation,
        then linearly interpolates between that base point and the apex.
    """
    
    # Validate pentahedron shape
    if pentahedron.shape != (5, 3):  
        return None
    
    # Handle both single point and batch of points
    uvw = np.atleast_2d(uvw)
    if uvw.shape[1] != 3:
        return None
    
    # Vectorized interpolation for all points
    u = uvw[:, 0:1]  # Shape (x, 1)
    v = uvw[:, 1:2]  # Shape (x, 1)
    w = uvw[:, 2:3]  # Shape (x, 1)
    
    # Compute base weights for bilinear interpolation
    base_weights = np.concatenate([
        (1-u) * (1-v),  # v00, shape (x, 1)
        u * (1-v),      # v10, shape (x, 1)
        (1-u) * v,      # v01, shape (x, 1)
        u * v           # v11, shape (x, 1)
    ], axis=1)  # Shape (x, 4)
    
    # Bilinearly interpolate the base quadrilateral points
    base_point = base_weights @ pentahedron[:4]  # Shape (x, 3)
    
    # Apex point
    apex = pentahedron[4]  # Shape (3,)
    
    # Linear interpolation between base point and apex using w
    xyz = (1-w) * base_point + w * apex[np.newaxis, :]  # Shape (x, 3)
    
    return xyz


def reverse_trilinear_interpolation_pentahedron_with_tangents(pentahedron: np.array, uvw: np.array) -> tuple:
    """
    Convert UVW coordinates to Cartesian coordinates and compute tangent vectors for a pentahedron.
    
    This function maps normalized UVW coordinates to the corresponding Cartesian positions within
    the pyramid, and also computes the tangent vectors along the U, V, and W parametric directions.
    These vectors represent the principal directions within the pentahedron.
    
    Args:
        pentahedron (np.array): Shape (5, 3) - Five vertices of the pentahedron in order:
                                [v00, v10, v01, v11, apex]
                                where v00, v10, v01, v11 form the base quadrilateral
                                and apex is the pyramid tip
        uvw (np.array): Shape (x, 3) - UVW coordinates where:
                        - UV interpolates within the base quadrilateral (u,v ∈ [0,1])
                        - W interpolates from base (w=0) to apex (w=1)
                        where x can be any number of points
    
    Returns:
        tuple: (cartesian_coords, tangent_u, tangent_v, tangent_w) where:
            - cartesian_coords: Shape (x, 3) - Cartesian coordinates of interpolated points
            - tangent_u: Shape (x, 3) - Tangent vectors along u-direction (∂P/∂u, unnormalized)
            - tangent_v: Shape (x, 3) - Tangent vectors along v-direction (∂P/∂v, unnormalized)
            - tangent_w: Shape (x, 3) - Tangent vectors along w-direction (∂P/∂w, unnormalized)
            Returns (None, None, None, None) if input validation fails
    
    UVW Coordinate System:
        - U axis: varies from 0 to 1 across the base's first parametric direction
            u=0 corresponds to base vertices v00, v01 (left edge)
            u=1 corresponds to base vertices v10, v11 (right edge)
        - V axis: varies from 0 to 1 across the base's second parametric direction
            v=0 corresponds to base vertices v00, v10 (bottom edge)
            v=1 corresponds to base vertices v01, v11 (top edge)
        - W axis: varies from 0 to 1 from base to apex
            w=0 corresponds to the base quadrilateral
            w=1 corresponds to the apex point
    
    Note:
        The tangent vectors are derived from the Jacobian matrix of the interpolation:
        - tangent_u (∂P/∂u): Shows how the position changes when moving in the u-direction
            on the base, scaled by (1-w) as the influence diminishes toward the apex
        - tangent_v (∂P/∂v): Shows how the position changes when moving in the v-direction
            on the base, scaled by (1-w) as the influence diminishes toward the apex
        - tangent_w: Points away from the apex toward the current base position,
            representing the direction from the apex tip toward the base (opposite of ∂P/∂w)
        
        All vectors are position-dependent and unnormalized. Their magnitudes provide
        information about local stretching and distortion. Users can normalize them
        using np.linalg.norm() if needed.
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_trilinear_interpolation_pentahedron(pentahedron, uvw)
    if cartesian_coords is None:
        return None, None, None, None
    
    # Handle both single point and batch of points
    uvw = np.atleast_2d(uvw)
    
    # Extract UVW components
    u = uvw[:, 0]  # Shape (x,)
    v = uvw[:, 1]  # Shape (x,)
    w = uvw[:, 2]  # Shape (x,)
    
    # Extract base vertices and apex
    v00, v10, v01, v11, apex = pentahedron[0], pentahedron[1], pentahedron[2], pentahedron[3], pentahedron[4]
    
    # Vectorized computation of base point and its derivatives
    # base_point shape: (x, 3)
    base_point = ((1-u)*(1-v))[:, np.newaxis]*v00 + (u*(1-v))[:, np.newaxis]*v10 + ((1-u)*v)[:, np.newaxis]*v01 + (u*v)[:, np.newaxis]*v11
    
    # Compute partial derivatives
    # dP/du = (1-w) * d[base]/du
    dbase_du = (-(1-v))[:, np.newaxis]*v00 + ((1-v))[:, np.newaxis]*v10 + (-v)[:, np.newaxis]*v01 + (v)[:, np.newaxis]*v11
    tangent_u = ((1-w)[:, np.newaxis]) * dbase_du  # Shape (x, 3)
    
    # dP/dv = (1-w) * d[base]/dv  
    dbase_dv = (-(1-u))[:, np.newaxis]*v00 + (-u)[:, np.newaxis]*v10 + ((1-u))[:, np.newaxis]*v01 + (u)[:, np.newaxis]*v11
    tangent_v = ((1-w)[:, np.newaxis]) * dbase_dv  # Shape (x, 3)
    
    # tangent_w: direction away from apex (toward base)
    tangent_w = base_point - apex[np.newaxis, :]  # Shape (x, 3)
    
    return cartesian_coords, tangent_u, tangent_v, tangent_w