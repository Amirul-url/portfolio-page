# rgbimage.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

class RgbImage:
    def __init__(self):
        self.imagelocation = None
        self.colorID = None
        self.image = None
        self.gray_image = None
        self.height = None
        self.width = None
        self.blankCanvas = np.zeros((512, 512, 3), np.uint8)  # Create blank canvas

    def load_image(self, image_location, color_id):
        """Loads an image and stores its attributes."""
        self.imagelocation = image_location
        self.colorID = color_id
        self.image = cv2.imread(self.imagelocation, self.colorID)
        self.gray_image = cv2.imread(self.imagelocation, 0)
        
        if self.image is None:
            raise ValueError("Could not load the image. Check the image path.")
        
        self.height = self.image.shape[0] 
        self.width = self.image.shape[1]

    def split_image(self):
        """Splits and displays RGB channels of the image using matplotlib."""
        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() first.")

        # Work on a copy of the original image
        image_copy = self.image.copy()  
        # Convert BGR to RGB
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        b, g, r = cv2.split(image_copy)
        combined = cv2.merge((b, g, r))
        
        # Prepare channels for display
        split_channel = [image_copy, b, g, r, combined]
        title_split = ["Original Image", "Blue Channel", "Green Channel", "Red Channel", "Combined Channel"]
        
        # Display the images in a grid of 3 columns
        plt.figure(figsize=(5, 3))  # Adjusted size for better visibility
        for i in range(len(split_channel)):
            plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns
            plt.title(title_split[i])
            plt.imshow(split_channel[i])
            plt.xticks([])  # Hide x ticks
            plt.yticks([])  # Hide y ticks

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
