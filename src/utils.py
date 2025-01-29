import numpy as np

def get_limits(color, hsv_thresholds):
    """
    Retrieves the lower and upper HSV limits for a given color.

    Args:
        color (str): The name of the color for which the HSV thresholds are being retrieved.
        hsv_thresholds (dict): A dictionary containing HSV threshold values for various colors.
            Each color is expected to have keys `H_lower`, `S_lower`, `V_lower`, `H_upper`, `S_upper`, and `V_upper`.

    Raises:
        ValueError: If the specified color is not found in the `hsv_thresholds` dictionary.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - lower_limit (np.ndarray): The lower HSV limit for the specified color.
            - upper_limit (np.ndarray): The upper HSV limit for the specified color.
    """
    
    if color not in hsv_thresholds:
        raise ValueError(f"Color '{color}' not found in HSV thresholds.")

    color_limits = hsv_thresholds[color]
    
    lower_limit = [color_limits["H_lower"], color_limits["S_lower"], color_limits["V_lower"]]
    upper_limit = [color_limits["H_upper"], color_limits["S_upper"], color_limits["V_upper"]]

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit

