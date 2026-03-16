# grayimage.py
import cv2
import matplotlib.pyplot as plt

class GrayImage:
    def __init__(self):
        self.image = None
        self.gray_image = None

    def load_image(self, image_location):
        """Loads an image and converts it to grayscale."""
        self.image = cv2.imread(image_location)
        if self.image is None:
            raise ValueError("Could not load the image. Check the image path.")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def display(self):
        """Displays the grayscale image."""
        plt.figure(figsize=(4, 3))
        plt.title("Grayscale Image")
        plt.imshow(self.gray_image, cmap='gray')
        plt.axis('off')
        plt.show()
