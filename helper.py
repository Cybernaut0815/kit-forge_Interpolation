import numpy as np


def wedge_2d(v: np.array, w: np.array) -> float:
    """
    Compute the 2D wedge product of two vectors.
    
    Args:
        v (np.array): First 2D vector
        w (np.array): Second 2D vector
    
    Returns:
        float: Signed area of parallelogram spanned by v and w
                - Positive: v is clockwise from w
                - Negative: v is counterclockwise from w  
                - Zero: vectors are collinear
    """
    return v[0] * w[1] - v[1] * w[0]


def wedge_2d_batch(v: np.array, w: np.array) -> np.array:
    """
    Compute the 2D wedge product for batches of vectors.
    
    Args:
        v (np.array): First vector(s) - Shape (2,) or (x, 2)
        w (np.array): Second vector(s) - Shape (2,) or (x, 2)
    
    Returns:
        np.array: Signed areas - Shape (x,) if inputs are (x, 2), or scalar if inputs are (2,)
                  - Positive: v is clockwise from w
                  - Negative: v is counterclockwise from w  
                  - Zero: vectors are collinear
    """
    v = np.atleast_2d(v)
    w = np.atleast_2d(w)
    return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]



def lerp(a: np.array, b: np.array, t: float) -> np.array:
    """
    Linear interpolation between two points or vectors.
    
    Args:
        a (np.array): Starting point (t=0)
        b (np.array): Ending point (t=1) 
        t (float): Interpolation parameter
    
    Returns:
        np.array: Interpolated point/vector
    """
    return a + (b - a) * t



def check_inputs(positions: np.array, position: np.array, expected_positions_shape: tuple = (4, 2), expected_position_shape: tuple = (2,), verbose: bool = False) -> bool:
    """
    Validate input arrays for geometric operations.
    
    Args:
        positions (np.array): Array of points to validate
        position (np.array): Single point to validate
        expected_positions_shape (tuple): Expected shape for positions array
        expected_position_shape (tuple): Expected shape for position array
        verbose (bool): Print error messages if True
    
    Returns:
        bool: True if inputs have correct dimensions
    """
    if positions.shape != expected_positions_shape or position.shape != expected_position_shape:
        if verbose:
            print(
                f"The positions array must be a {expected_positions_shape} array and the position array must be a {expected_position_shape} array."
            )
        return False
    return True


def generate_uv_grid(grid_size: int = 10, uv_min: float = 0.0, uv_max: float = 1.0) -> np.array:
    """
    Generate a uniform grid of 2D UV coordinates for testing interpolation methods.
    
    Args:
        grid_size (int): Number of points along each axis (default: 10)
        uv_min (float): Minimum UV coordinate value (default: 0.0)
        uv_max (float): Maximum UV coordinate value (default: 1.0)
    
    Returns:
        np.array: Shape (grid_size*grid_size, 2) - Array of UV coordinates in [uv_min, uv_max] range
                  arranged in a grid pattern. Points are ordered row by row.
    
    Example:
        # Generate a 5x5 grid in [0, 1] range
        uv_grid = generate_uv_grid(5)
        # Result shape: (25, 2)
    """
    u = np.linspace(uv_min, uv_max, grid_size)
    v = np.linspace(uv_min, uv_max, grid_size)
    u_grid, v_grid = np.meshgrid(u, v)
    uv_coordinates = np.column_stack([u_grid.ravel(), v_grid.ravel()])
    return uv_coordinates


