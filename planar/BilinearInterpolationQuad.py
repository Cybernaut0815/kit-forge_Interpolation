import numpy as np
from src.interpolation.helper import wedge_2d, wedge_2d_batch, lerp


def bilinear_interpolation_quad(quad: np.array, cartesian_coordinates: np.array):
    """
    Convert Cartesian coordinates to UV coordinates within a quadrilateral using bilinear interpolation.
    
    This function finds the normalized UV coordinates (where u,v ∈ [0,1]) that correspond to
    given Cartesian positions within the quadrilateral. The inverse mapping solves the
    bilinear interpolation equations to recover the parametric coordinates.
    
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
        Uses quadratic formula to solve the inverse bilinear interpolation problem.
        For degenerate cases where the quadrilateral becomes triangular (A ≈ 0),
        falls back to linear interpolation. Based on algorithm from:
        https://www.reedbeta.com/blog/quadrilateral-interpolation-part-2/
    """

    # Validate quad shape
    if quad.shape != (4, 2):
        return None
    
    # Handle both single point and batch of points
    cartesian_coordinates = np.atleast_2d(cartesian_coordinates)
    if cartesian_coordinates.shape[1] != 2:
        return None
    
    num_points = cartesian_coordinates.shape[0]
    uv = np.zeros((num_points, 2))

    q = cartesian_coordinates - quad[0]  # Broadcasting: (x,2) - (2,) -> (x,2)

    b1 = quad[1] - quad[0]  # Shape (2,)
    b2 = quad[2] - quad[0]  # Shape (2,)
    b3 = quad[0] - quad[1] - quad[2] + quad[3]  # Shape (2,)

    # Vectorized wedge products
    A = wedge_2d(b1, b3)  # Scalar
    B = wedge_2d_batch(np.tile(b3, (num_points, 1)), q) - wedge_2d(b1, b2)  # Shape (x,)
    C = wedge_2d_batch(np.tile(b1, (num_points, 1)), q)  # Shape (x,)

    # Handle degenerate and normal cases
    degenerate = np.abs(A) < 0.001
    
    if degenerate:
        # For degenerate case (A ≈ 0), use linear solution
        uv[:, 1] = -C / B
    else:
        # For normal case, use quadratic formula
        discr = B * B - 4 * A * C
        uv[:, 1] = 0.5 * (-B + np.sqrt(discr)) / A

    # Vectorized calculation of u coordinate
    denom = b1[np.newaxis, :] + uv[:, 1:2] * b3[np.newaxis, :]  # Shape (x, 2)
    
    # Choose more stable denominator component for each point
    use_x = np.abs(denom[:, 0]) > np.abs(denom[:, 1])
    
    uv[use_x, 0] = (q[use_x, 0] - b2[0] * uv[use_x, 1]) / denom[use_x, 0]
    uv[~use_x, 0] = (q[~use_x, 1] - b2[1] * uv[~use_x, 1]) / denom[~use_x, 1]

    return uv


def reverse_bilinear_interpolation_quad(quad: np.array, uv: np.array) -> np.array:
    """
    Convert UV coordinates to Cartesian coordinates within a quadrilateral using bilinear interpolation.
    
    This function maps normalized UV coordinates (where u,v ∈ [0,1]) to the corresponding 
    Cartesian positions within the quadrilateral defined by four vertices. The mapping uses
    bilinear interpolation which preserves straight lines parallel to the UV axes.
    
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
        The UV coordinates are applied using bilinear interpolation weights to blend between
        the four quadrilateral vertices, preserving straight lines in parametric space.
        Based on quadrilateral interpolation method from:
        https://www.reedbeta.com/blog/quadrilateral-interpolation-part-2/
    """

    # Validate quad shape
    if quad.shape != (4, 2):
        return None
    
    # Handle both single point and batch of points
    uv = np.atleast_2d(uv)
    if uv.shape[1] != 2:
        return None

    v00 = quad[0]  # Shape (2,)
    v10 = quad[1]  # Shape (2,)
    v01 = quad[2]  # Shape (2,)
    v11 = quad[3]  # Shape (2,)

    u = uv[:, 0:1]  # Shape (x, 1)
    v = uv[:, 1:2]  # Shape (x, 1)

    # Vectorized bilinear interpolation
    # First interpolate along u axis for both v=0 and v=1 edges
    bottom_edge = lerp(v00[np.newaxis, :], v10[np.newaxis, :], u)  # Shape (x, 2)
    top_edge = lerp(v01[np.newaxis, :], v11[np.newaxis, :], u)     # Shape (x, 2)
    
    # Then interpolate along v axis
    return lerp(bottom_edge, top_edge, v)


def reverse_bilinear_interpolation_quad_with_tangents(quad: np.array, uv: np.array) -> tuple:
    """
    Convert UV coordinates to Cartesian coordinates and compute tangent vectors.
    
    This function maps normalized UV coordinates (where u,v ∈ [0,1]) to the corresponding 
    Cartesian positions within the quadrilateral, and also computes the tangent vectors 
    in the u and v parametric directions at each point.
    
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
        The tangent vectors are computed as partial derivatives of the bilinear interpolation:
        - ∂P/∂u: tangent in u-direction (horizontal parametric direction)
        - ∂P/∂v: tangent in v-direction (vertical parametric direction)
        The vectors are NOT normalized. Their magnitudes represent local stretching in each
        parametric direction. Users can normalize them if needed using np.linalg.norm().
        Based on quadrilateral interpolation method from:
        https://www.reedbeta.com/blog/quadrilateral-interpolation-part-2/
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_bilinear_interpolation_quad(quad, uv)
    if cartesian_coords is None:
        return None, None, None
    
    # Handle both single point and batch of points
    uv = np.atleast_2d(uv)
    
    v00 = quad[0]  # Shape (2,)
    v10 = quad[1]  # Shape (2,)
    v01 = quad[2]  # Shape (2,)
    v11 = quad[3]  # Shape (2,)

    u = uv[:, 0:1]  # Shape (x, 1)
    v = uv[:, 1:2]  # Shape (x, 1)
    
    # Compute partial derivatives (tangent vectors)
    # ∂P/∂u = (1-v) * (v10 - v00) + v * (v11 - v01)
    tangent_u = (1 - v) * (v10 - v00)[np.newaxis, :] + v * (v11 - v01)[np.newaxis, :]  # Shape (x, 2)
    
    # ∂P/∂v = (1-u) * (v01 - v00) + u * (v11 - v10)
    tangent_v = (1 - u) * (v01 - v00)[np.newaxis, :] + u * (v11 - v10)[np.newaxis, :]  # Shape (x, 2)
    
    return cartesian_coords, tangent_u, tangent_v
