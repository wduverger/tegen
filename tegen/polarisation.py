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
import mpl_toolkits.axes_grid1.anchored_artists as mpl_aa
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
import numpy as np
import scipy.ndimage
import scipy.signal
from PIL import Image, ImageDraw, ImageFont


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


def add_scalebar(axis, len_in_pixels, label=None, position='upper right',
                 color='white', frameon=False, pad=.3, size_vertical=2, **kwargs):
    """ Add scalebar to a matplotlib plot with sensible defaults """

    axis.add_artist(mpl_aa.AnchoredSizeBar(
        axis.transData, len_in_pixels, label, position, color=color,
        frameon=frameon, size_vertical=size_vertical, pad=pad,
        **kwargs
    ))


def add_scalebar_in_place(im, pad, height, width, label=''):
    """
    Add scalebar to an image. Input: numpy array, outputs new array
    """

    Y, X, C = im.shape

    # Draw scale bar
    width = int(width)
    im = im.copy()
    im[pad:pad+height, -pad-width:-pad, :] = [1, 1, 1]

    # Draw text
    pil_img = Image.fromarray(np.uint8(im*255))
    drawer = ImageDraw.Draw(pil_img)
    drawer.text(
        (Y-pad-width, pad+2*height),
        label,
        font=ImageFont.truetype(
            font='./utils/SourceSansPro-Regular.ttf', size=19)
    )
    np_img = np.array(pil_img, dtype=float)/255

    return np_img


def add_colourwheel(axis, loc='upper left', size=.12, **wheel_kw):
    """ 
    Add a colourwheel to a plot. 
    Use `wheel_kw` to choose alpha, saturation, etc of the wheel image
    """

    wheel_im = _generate_wheel(**wheel_kw)

    inset = mpl_il.inset_axes(
        axis, loc='upper left', width=size, height=size
    )
    inset.axis('off')
    inset.imshow(wheel_im)


def add_colourwheel_in_place(im, pad, resolution):
    """ 
    Add a colourwheel to an np.array
    Use `wheel_kw` to choose alpha, saturation, etc of the wheel image
    """
    wheel_im = _generate_wheel(resolution=resolution)
    Y, X, _ = wheel_im.shape
    wheel_rgb = wheel_im[..., 0:3]
    wheel_alpha = wheel_im[..., 3].astype(int).reshape(Y, X, 1)

    im = im.copy()
    im[pad:pad+Y, pad:pad+X] = wheel_alpha * wheel_rgb + \
        (1-wheel_alpha) * im[pad:pad+Y, pad:pad+X]
    return im


def _generate_wheel(alpha=1, sat=1, val=.9, resolution=100):
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)

    xx, yy = np.meshgrid(x, y)
    im_hsv = np.zeros((*xx.shape, 3))

    rr = np.sqrt(xx**2 + yy**2)  # Distance from origin
    tt = np.arctan2(yy, -xx)     # Angle of pixel vector to x axis

    disc = rr < 1  # Image that is 0 if pixel outside unit circle, 1 otherwise

    im_hsv[..., 0] = 1 - (disc*tt / np.pi + 1) % 1  # Hue
    im_hsv[..., 1] = disc*rr * sat                # Saturation
    im_hsv[..., 2] = disc * val                   # Brightness

    # Convert to RGB
    im_hsv = matplotlib.colors.hsv_to_rgb(im_hsv)
    im_rgb = np.zeros((*xx.shape, 4))
    im_rgb[..., 0:3] = im_hsv

    # Add alpha channel
    # alpha should be 0 outside the disk, otherwise the image will be black
    im_rgb[..., 3] = disc * alpha

    return im_rgb
