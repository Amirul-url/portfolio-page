# hlsimage.py
import cv2
import matplotlib.pyplot as plt

class HlsImage:
    def __init__(self):
        self.image = None
        self.hls_image = None

    def load_image(self, image_location):
        """Loads an image and converts it to HLS."""
        self.image = cv2.imread(image_location)
        if self.image is None:
            raise ValueError("Could not load the image. Check the image path.")
        self.hls_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)

    def display(self):
        """Displays the HLS image."""
        plt.figure(figsize=(4, 3))
        plt.title("Hue, Lightness, Saturation (HLS) Image")
        plt.imshow(self.hls_image)
        plt.axis('off')
        plt.show()
