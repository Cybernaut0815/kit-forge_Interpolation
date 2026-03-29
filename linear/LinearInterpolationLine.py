import numpy as np


def linear_interpolation_line(line: np.array, cartesian_coordinates: np.array) -> np.array:
    """
    Convert Cartesian coordinates to parametric coordinate t along a line.
    
    This function finds the parameter t that corresponds to given Cartesian positions
    along a line segment. The parameter t represents the position along the line where
    t=0 is at the start point and t=1 is at the end point.
    
    Args:
        line (np.array): Shape (2, 3) - Two vertices defining the line:
                        [start_point, end_point]
        cartesian_coordinates (np.array): Shape (x, 3) - Cartesian coordinates of points to convert
                                            where x can be any number of points
    
    Returns:
        np.array: Shape (x, 1) - Parametric coordinate t where:
                    - t=0 corresponds to start_point
                    - t=1 corresponds to end_point
                    - 0 < t < 1 for points between start and end
                    Returns None if input validation fails
    
    Parametric Coordinate System:
        - t: linear parameter along the line
          * t=0: point lies at start_point (line[0])
          * t=1: point lies at end_point (line[1])
          * Points outside [0,1] lie on the line extension beyond the segment
    
    Note:
        This function projects the input points onto the line. For points not exactly
        on the line, it returns the parameter of the closest point on the line.
        The projection is computed using: t = dot(P - start, direction) / ||direction||^2
    """
    
    # Validate line shape
    if line.shape != (2, 3):
        return None
    
    # Handle both single point and batch of points
    cartesian_coordinates = np.atleast_2d(cartesian_coordinates)
    if cartesian_coordinates.shape[1] != 3:
        return None
    
    # Extract start and end points
    start_point = line[0]
    end_point = line[1]
    
    # Direction vector along the line
    direction = end_point - start_point
    length_squared = np.dot(direction, direction)
    
    # Check for degenerate line (start and end are the same)
    if length_squared < 1e-10:
        return None
    
    # Vectorized projection of all points onto the line
    # t = dot(P - start, direction) / ||direction||^2
    point_vectors = cartesian_coordinates - start_point[np.newaxis, :]  # Shape (x, 3)
    t = (point_vectors @ direction / length_squared)[:, np.newaxis]  # Shape (x, 1)
    
    return t


def reverse_linear_interpolation_line(line: np.array, t: np.array) -> np.array:
    """
    Convert parametric coordinate t to Cartesian coordinates along a line.
    
    This function maps the parameter t to the corresponding Cartesian positions
    along a line segment. The interpolation is: P(t) = (1-t) * start + t * end
    
    Args:
        line (np.array): Shape (2, 3) - Two vertices defining the line:
                        [start_point, end_point]
        t (np.array): Shape (x, 1) - Parametric coordinate where:
                        - t=0 maps to start_point
                        - t=1 maps to end_point
                        where x can be any number of points
    
    Returns:
        np.array: Shape (x, 3) - Cartesian coordinates of the interpolated points,
                    or None if input validation fails
    
    Parametric Coordinate System:
        - t: linear parameter along the line
          * t=0: point lies at start_point (line[0])
          * t=1: point lies at end_point (line[1])
          * Points with t outside [0,1] lie on the line extension beyond the segment
    
    Note:
        The interpolation is linear: P(t) = start + t * (end - start)
        This is equivalent to: P(t) = (1-t) * start + t * end
    """
    
    # Validate line shape
    if line.shape != (2, 3):
        return None
    
    # Handle both single point and batch of points
    t = np.atleast_2d(t)
    if t.shape[1] != 1:
        return None
    
    # Extract start and end points
    start_point = line[0]
    end_point = line[1]
    
    # Vectorized linear interpolation for all points
    # P(t) = (1-t) * start + t * end
    xyz = (1 - t) * start_point[np.newaxis, :] + t * end_point[np.newaxis, :]
    
    return xyz


def reverse_linear_interpolation_line_with_tangent(line: np.array, t: np.array) -> tuple:
    """
    Convert parametric coordinate t to Cartesian coordinates and compute the tangent vector.
    
    This function maps the parameter t to the corresponding Cartesian positions along
    a line segment, and also computes the tangent vector along the line. For a straight
    line, the tangent vector is constant and points from the start to the end point.
    
    Args:
        line (np.array): Shape (2, 3) - Two vertices defining the line:
                        [start_point, end_point]
        t (np.array): Shape (x, 1) - Parametric coordinate where:
                        - t=0 maps to start_point
                        - t=1 maps to end_point
                        where x can be any number of points
    
    Returns:
        tuple: (cartesian_coords, tangent) where:
            - cartesian_coords: Shape (x, 3) - Cartesian coordinates of interpolated points
            - tangent: Shape (x, 3) - Tangent vectors along the line (constant, unnormalized)
            Returns (None, None) if input validation fails
    
    Parametric Coordinate System:
        - t: linear parameter along the line
          * t=0: point lies at start_point (line[0])
          * t=1: point lies at end_point (line[1])
    
    Note:
        The tangent vector is the direction from start_point to end_point:
        tangent = end_point - start_point
        
        For a straight line, the tangent vector is constant at all points and
        points from the start toward the end. The magnitude of the tangent equals
        the length of the line segment. Users can normalize using np.linalg.norm().
        
        The tangent vector represents dP/dt, the rate of change of position with
        respect to the parameter t.
    """
    
    # Get Cartesian coordinates using existing function
    cartesian_coords = reverse_linear_interpolation_line(line, t)
    if cartesian_coords is None:
        return None, None
    
    # Handle both single point and batch of points
    t = np.atleast_2d(t)
    num_points = t.shape[0]
    
    # Extract start and end points
    start_point = line[0]
    end_point = line[1]
    
    # Tangent vector is constant along a straight line
    tangent_vector = end_point - start_point
    
    # Replicate tangent vector for all points
    tangent = np.tile(tangent_vector, (num_points, 1))
    
    return cartesian_coords, tangent

