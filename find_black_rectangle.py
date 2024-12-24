import numpy as np

def find_black_rectangle(image: np.ndarray, threshold: float = 0.13, padding: int = 4):
    """
    Finds the largest black rectangular region in an RGB image with precise padding.

    Args:
        image (numpy.ndarray): Input image array with shape (H, W, 3).
        threshold (float, optional): Blackness threshold for all three RGB channels. Default is 0.05.
        padding (int, optional): Fixed padding to apply evenly around the rectangle. Default is 3 pixels.

    Returns:
        tuple: Coordinates of the largest black rectangle with padding (y_min, y_max, x_min, x_max).
    """
    # Convert image to binary mask where black pixels are True
    black_mask = np.all(image < threshold, axis=-1)

    H, W = black_mask.shape
    largest_area = 0
    best_coords = None

    # DP table to store largest rectangle widths
    dp_width = np.zeros((H, W), dtype=int)

    for y in range(H):
        for x in range(W):
            # Calculate widths of black regions
            if black_mask[y, x]:
                dp_width[y, x] = dp_width[y, x - 1] + 1 if x > 0 else 1

            # Check for the largest rectangle ending at (y, x)
            width = dp_width[y, x]
            for k in range(y, -1, -1):  # Iterate upward to calculate height
                width = min(width, dp_width[k, x])
                if width == 0:
                    break
                height = y - k + 1
                area = width * height
                if area > largest_area:
                    largest_area = area
                    best_coords = (k, y, x - width + 1, x)

    # Ensure valid coordinates
    if best_coords is None:
        return (0, 0, 0, 0)  # No black rectangle found

    y_min, y_max, x_min, x_max = best_coords

    # Apply precise padding of 3 pixels evenly
    y_min = max(0, y_min - padding)
    y_max = min(H, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(W, x_max + padding)

    return y_min, y_max, x_min, x_max