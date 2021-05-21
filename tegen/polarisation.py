"""
A module facilitating analysis of polarisation images

Usage:

    import matplotlib.pyplot as plt
    from tegen import bioformats, polarisation

    images = bioformats.read_file(path)
    pol_stack = images['image_channel']
    pol_rgb = polarisation.stack_to_rgb(pol_stack)
    plt.imshow(pol_rgb)

"""


import cv2
import matplotlib.colors
import numpy as np
import scipy.ndimage
import scipy.signal


def align_stack(original_image):
    """
    Align a stack of images (order of dimensions: zyx),
    only allowing translational motion
    """

    print('Opencv align...')

    corrected_image = np.zeros_like(original_image)
    corrected_image[0] = original_image[0]

    Z, Y, X = original_image.shape
    opencv_size = (X, Y)
    warp_matrices = []

    for i in range(Z - 1):
        reference = corrected_image[i]
        new_frame = original_image[i+1]

        num_iterations = int(1e6)
        eps = 1e15
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iterations,
            eps
        )

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        (_, warp_matrix) = cv2.findTransformECC(
            reference, new_frame, warp_matrix,
            cv2.MOTION_TRANSLATION, criteria,
        )

        corrected_image[i+1] = cv2.warpAffine(
            new_frame, warp_matrix, opencv_size,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        warp_matrices.append(warp_matrix)

    return corrected_image


def compensate_bleaching(stack):
    """
    Estimate the bleaching rate between the first and 
    last frame and compensate for it.
    """

    i = np.arange(len(stack))
    total_bleaching = stack[-1].sum() / stack[0].sum()
    bleaching_rate = np.power(total_bleaching, 1/len(stack))
    exp = np.power(bleaching_rate, -i).reshape((-1, 1, 1))
    return stack * exp


def gaussian_blur(stack, **kwargs):
    """
    Blur a stack. Keyword arguments are passed to
    scipy.ndimage.gaussian_filter() 
    """

    blurred = np.zeros_like(stack)

    for i in range(len(stack)):
        blurred[i] = scipy.ndimage.gaussian_filter(stack[i], **kwargs)
    return blurred

def stack_to_rgb(stack, pol_axis=np.arange(0, 171, 10),
               blur_sigma=0, saturation=1, brightness=1):
    """
    Convert a stack of polarisation images into an RGB picture.
    """

    _, y, x = stack.shape
    image_hsv = np.zeros((y, x, 3), dtype=np.float32)

    unbleached = compensate_bleaching(stack)
    blurred = gaussian_blur(unbleached, sigma=blur_sigma)

    # Calculate Fourier coefficients
    sin = np.sin(pol_axis * np.pi/90).reshape((-1, 1, 1))
    cos = np.cos(pol_axis * np.pi/90).reshape((-1, 1, 1))
    sin = (sin * blurred).sum(axis=0)
    cos = (cos * blurred).sum(axis=0)

    # Define hue, saturation and value channels
    h = np.arctan2(sin, cos)
    v = unbleached.sum(axis=0)
    s = np.sqrt(np.power(sin, 2) + np.power(cos, 2)) / v

    # Ensure proper normalisation
    image_hsv[..., 0] = h / (2 * np.pi) + .5
    image_hsv[..., 1] = (s/s.max())**saturation
    image_hsv[..., 2] = (v/v.max())**brightness

    return matplotlib.colors.hsv_to_rgb(image_hsv)
