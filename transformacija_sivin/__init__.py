import numpy as np


def normaliziraj_sivine_uniformno(slika: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmin >= vmax:
        raise ValueError("vmin must be less than vmax.")
    min_val = slika.min()
    max_val = slika.max()
    if min_val == max_val:
        # If all pixel values are the same, return an array with all values equal to vmin
        return np.full(slika.shape, vmin, dtype=np.float32)
    # Perform the normalization
    normalized_slika = (slika - min_val) / (max_val - min_val) * (vmax - vmin) + vmin
    return normalized_slika.astype(np.float32)


def normaliziraj_sivine_normalno(slika: np.ndarray, povprecje: float, odklon: float) -> np.ndarray:
    # Calculate the mean and standard deviation of the input image
    mean_val = slika.mean()
    std_dev = slika.std()

    # Check for zero standard deviation in the image and raise an error if found
    if std_dev == 0:
        raise ZeroDivisionError("Standard deviation of the image is zero, cannot normalize.")

    # Check for invalid desired standard deviation
    if odklon == 0:
        raise ValueError("Desired standard deviation must be non-zero.")

    # Perform the normalization using the provided formula
    normalized_slika = (slika - mean_val) / std_dev * odklon + povprecje
    return normalized_slika.astype(np.float32)




    
def transformiraj_z_lut(slika: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Transforms the image using the provided lookup table (lut).
    :param slika: Input image array of shape (H, W), type np.uint16
    :param lut: Lookup table array of shape (N,), any type
    :return: Transformed image array of the same shape as slika and same type as lut
    """
    # If the pixel value is greater than the last index of the LUT, use the last LUT value
    transformed_slika = np.take(lut, slika, mode='clip')
    return transformed_slika

    

def izenaci_histogram(slika, vmin=0, vmax=1, bins=256, interpolacija=False):
    # Normalize the image between 0 and bins
    slika_norm = normaliziraj_sivine_uniformno(slika, vmin, vmax) * (bins - 1)
    slika_norm = slika_norm.astype(int)

    # Calculate histogram and cumulative histogram
    hist, _ = np.histogram(slika_norm, bins=bins, range=(0, bins - 1))
    cum_hist = np.cumsum(hist).astype(float)
    cum_hist_normalized = (cum_hist - cum_hist.min()) / (cum_hist.max() - cum_hist.min()) * (bins - 1)

    # Apply the cumulative histogram as a look-up table for the normalized image
    slika_eq = transformiraj_z_lut(slika_norm, cum_hist_normalized)

    # If interpolation is True, perform linear interpolation
    if interpolacija:
        slika_eq = np.interp(slika.flatten(), np.linspace(0, bins - 1, bins), cum_hist_normalized).reshape(slika.shape)

    # Rescale the image back to original range
    slika_rescaled = slika_eq / (bins - 1) * (vmax - vmin) + vmin

    return slika_rescaled.astype(np.float32)
