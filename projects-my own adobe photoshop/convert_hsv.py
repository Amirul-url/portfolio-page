# hsvimage.py
import cv2
import matplotlib.pyplot as plt

class HsvImage:
    def __init__(self):
        self.image = None
        self.hsv_image = None

    def load_image(self, image_location):
        """Loads an image and converts it to HSV."""
        self.image = cv2.imread(image_location)
        if self.image is None:
            raise ValueError("Could not load the image. Check the image path.")
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def display(self):
        """Displays the HSV image."""
        plt.figure(figsize=(4, 3))
        plt.title("Hue, Saturation, Value (HSV) Image")
        plt.imshow(self.hsv_image)
        plt.axis('off')
        plt.show()
