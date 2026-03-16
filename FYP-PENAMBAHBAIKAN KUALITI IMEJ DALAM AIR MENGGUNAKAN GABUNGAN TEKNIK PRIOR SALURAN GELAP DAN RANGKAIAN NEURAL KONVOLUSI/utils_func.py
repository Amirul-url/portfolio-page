import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import cv2


def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))

def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result

def to_image(image):
    """ converts to PIL image """
    return Image.fromarray((image * 255).astype(np.uint8))

def dark_channel(image, size=15):
    """Compute dark channel prior for an image."""
    # Convert to float
    image = image.astype(np.float64) / 255.0

    # Get minimum channel
    min_channel = np.min(image, axis=2)

    # Apply minimum filter (erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percent=0.1):
    """Estimate atmospheric light from dark channel."""
    image = image.astype(np.float64) / 255.0

    # Get the number of pixels for top 0.1%
    num_pixels = int(image.shape[0] * image.shape[1] * top_percent)

    # Find the indices of the brightest pixels in dark channel
    flat_dark = dark_channel.flatten()
    indices = np.argsort(flat_dark)[-num_pixels:]

    # Get the corresponding pixels from original image
    atmospheric_light = np.zeros(3)
    for i in range(3):
        channel_pixels = image[:, :, i].flatten()[indices]
        atmospheric_light[i] = np.max(channel_pixels)

    return atmospheric_light

def estimate_transmission(image, atmospheric_light, omega=0.95, size=15):
    """Estimate transmission map using dark channel prior."""
    image = image.astype(np.float64) / 255.0

    # Normalize image by atmospheric light
    norm_image = image / atmospheric_light

    # Get dark channel of normalized image
    dark_channel_norm = dark_channel((norm_image * 255).astype(np.uint8), size)

    # Estimate transmission
    transmission = 1 - omega * dark_channel_norm

    return transmission

def recover_image(image, transmission, atmospheric_light, t0=0.1):
    """Recover the scene radiance using transmission and atmospheric light."""
    image = image.astype(np.float64) / 255.0

    # Refine transmission
    transmission = np.maximum(transmission, t0)

    # Recover scene radiance
    recovered = np.zeros_like(image)
    for i in range(3):
        recovered[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

    # Clip to valid range
    recovered = np.clip(recovered, 0, 1)

    return recovered

def enhance_image_dcp(image, omega=0.4, t0=0.2, size=15):
    """Enhance underwater image using Dark Channel Prior method."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Estimate dark channel
    dark_ch = dark_channel(image, size)

    # Estimate atmospheric light
    atmospheric_light = estimate_atmospheric_light(image, dark_ch)

    # Estimate transmission
    transmission = estimate_transmission(image, atmospheric_light, omega, size)

    # Recover image
    enhanced = recover_image(image, transmission, atmospheric_light, t0)

    return enhanced