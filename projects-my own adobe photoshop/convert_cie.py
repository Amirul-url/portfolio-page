# cieimage.py
import cv2
import matplotlib.pyplot as plt

class CieImage:
    def __init__(self):
        self.image = None
        self.cie_image = None

    def load_image(self, image_location):
        """Loads an image and converts it to CIE Lab."""
        self.image = cv2.imread(image_location)
        if self.image is None:
            raise ValueError("Could not load the image. Check the image path.")
        self.cie_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)

    def display(self):
        """Displays the CIE Lab image."""
        plt.figure(figsize=(4, 3))
        plt.title("CIE Lab Image")
        plt.imshow(self.cie_image)
        plt.axis('off')
        plt.show()
