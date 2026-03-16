# ycrcbimage.py
import cv2
import matplotlib.pyplot as plt

class YCrCbImage:
    def __init__(self):
        self.image = None
        self.ycrcb_image = None

    def load_image(self, image_location):
        """Loads an image and converts it to YCrCb."""
        self.image = cv2.imread(image_location)
        if self.image is None:
            raise ValueError("Could not load the image. Check the image path.")
        self.ycrcb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)

    def display(self):
        """Displays the YCrCb image."""
        plt.figure(figsize=(4, 3))
        plt.title("Luma (Y), Chroma Red (Cr), \nChroma Blue (Cb) (YCrCb) Image")
        plt.imshow(self.ycrcb_image)
        plt.axis('off')
        plt.show()
