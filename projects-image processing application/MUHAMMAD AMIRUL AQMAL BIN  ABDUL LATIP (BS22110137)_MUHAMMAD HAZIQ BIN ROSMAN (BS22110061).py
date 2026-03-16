import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, QGroupBox, QToolBar, QMenu, QAction, QToolButton, QSplashScreen, QSlider, QComboBox, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QIcon
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from skimage.metrics import structural_similarity as compare_ssim
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from math import log10, sqrt

class HistogramCanvas(FigureCanvas):
    # Canvas for displaying histograms using Matplotlib
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_histogram(self, image, title):
        # Plots the histogram for the given image
        self.axes.clear()
        if len(image.shape) == 3:  # Color image
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                self.axes.plot(hist, color=color)
        else:  # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            self.axes.plot(hist, color='black')
        self.axes.set_title(title)
        self.axes.set_xlim([0, 256])
        self.draw()

class ImageEnhancer(QMainWindow):
    open_windows = []
    # Main application for image enhancement with real-time histograms
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Project - Image Processing Application by BS22110137 & BS22110061")
        self.setGeometry(150, 80, 1400, 900)

        self.cropping = False
        self.crop_start = None
        self.crop_end = None
        self.original_image = None
        self.enhanced_image = None    
        
        self.setup_menu()

        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background-color: darkred;")

        main_layout = QGridLayout(self.central_widget)

        # Define widgets and layouts
        originalImageWidget = QWidget()
        originalImageWidget.setStyleSheet("background-color: black;") 
        enhancedImageWidget = QWidget()
        enhancedImageWidget.setStyleSheet("background-color: black;") 
        button_widget = QWidget()
        button_widget.setStyleSheet("background-color: black;")
        filter_widget = QWidget()
        filter_widget.setStyleSheet("background-color: black;")
        edge_widget = QWidget()
        edge_widget.setStyleSheet("background-color: black;")
        bit_widget = QWidget()
        bit_widget.setStyleSheet("background-color: black;")
        bit_widget.setFixedWidth(300) 
        metrics_widget = QWidget()
        metrics_widget.setStyleSheet("background-color: black;") 
        metrics2_widget = QWidget()
        metrics2_widget.setStyleSheet("background-color: black;") 

        originalImageLayout = QVBoxLayout(originalImageWidget)
        enhancedImageLayout = QVBoxLayout(enhancedImageWidget)
        button_layout = QVBoxLayout(button_widget)
        filter_layout = QVBoxLayout(filter_widget)
        edge_layout = QVBoxLayout(edge_widget)
        bit_layout = QVBoxLayout(bit_widget)
        metrics_layout = QHBoxLayout(metrics_widget)
        metrics2_layout = QHBoxLayout(metrics2_widget)
        
        main_layout.addWidget(originalImageWidget, 1, 1)
        main_layout.addWidget(enhancedImageWidget, 1, 2)
        main_layout.addWidget(button_widget, 1, 3)
        main_layout.addWidget(filter_widget, 1, 3)
        main_layout.addWidget(edge_widget, 1, 3)
        main_layout.addWidget(bit_widget, 1, 3) 
        main_layout.addWidget(metrics_widget, 2, 1)
        main_layout.addWidget(metrics2_widget, 2, 2)
        
        # Image display areas
        self.original_label = QLabel("Original Image", self)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: lightgray;")
        self.original_label.setFixedSize(650, 400)
        main_layout.addWidget(self.original_label, 0, 0)

        self.enhanced_label = QLabel("Enhanced Image", self)
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setStyleSheet("background-color: lightgreen;")
        self.enhanced_label.setFixedSize(650, 400)
        main_layout.addWidget(self.enhanced_label, 0, 1)
        
        # Enable mouse tracking and assign event handlers for cropping
        self.original_label.setMouseTracking(True)
        self.original_label.mousePressEvent = self.start_crop
        self.original_label.mouseMoveEvent = self.update_crop
        self.original_label.mouseReleaseEvent = self.finish_crop        

        originalImageLayout.addWidget(self.original_label)
        enhancedImageLayout.addWidget(self.enhanced_label)

        # Histogram canvases
        self.original_hist_canvas = HistogramCanvas(self)
        self.enhanced_hist_canvas = HistogramCanvas(self)

        originalImageLayout.addWidget(self.original_hist_canvas)
        enhancedImageLayout.addWidget(self.enhanced_hist_canvas)

        # Metrics labels
        self.original_ssim_label = QLabel("SSIM: N/A")
        self.original_psnr_label = QLabel("PSNR: N/A")
        self.enhanced_ssim_label = QLabel("SSIM: N/A")
        self.enhanced_psnr_label = QLabel("PSNR: N/A")

        for label in [self.original_ssim_label, self.original_psnr_label, self.enhanced_ssim_label, self.enhanced_psnr_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: yellow; font-size: 16px; padding: 5px;")

        metrics_layout.addWidget(QLabel("Original Metrics", alignment=Qt.AlignCenter, styleSheet="background-color: peachpuff; font-size: 18px;"))
        metrics_layout.addWidget(self.original_ssim_label)
        metrics_layout.addWidget(self.original_psnr_label)

        metrics2_layout.addWidget(QLabel("Enhanced Metrics", alignment=Qt.AlignCenter, styleSheet="background-color: peachpuff; font-size: 18px;"))
        metrics2_layout.addWidget(self.enhanced_ssim_label)
        metrics2_layout.addWidget(self.enhanced_psnr_label)    

        # Buttons
        self.load_button = QPushButton("Load Image")
        self.save_button = QPushButton("Save Image")
        self.reset_all_button = QPushButton("Reset All")
        self.reset_image_button = QPushButton("Reset Image")
        self.exit_button = QPushButton("Exit")

        self.noise_button = QPushButton("Noise")
        self.sharp_button = QPushButton("Sharpen")
        self.dcp_button = QPushButton("DCP Enhancement")
        self.hdr_button = QPushButton("HDR Enhancement")
        self.combine_button = QPushButton("Combine Images")

        self.noise_slider = QSlider(Qt.Horizontal)
        self.sharp_slider = QSlider(Qt.Horizontal)
        self.dcp_slider = QSlider(Qt.Horizontal)
        self.hdr_slider = QSlider(Qt.Horizontal)
        self.combine_slider = QSlider(Qt.Horizontal)

        self.noise_slider.setRange(0, 100)
        self.sharp_slider.setRange(0, 100)
        self.dcp_slider.setRange(0, 100)
        self.hdr_slider.setRange(0, 100)
        self.combine_slider.setRange(0, 100)
        
        self.noise_slider.setValue(50)  # Default noise slider value
        self.sharp_slider.setValue(50)  # Default sharpen slider value
        self.dcp_slider.setValue(50)  # Default DCP slider value
        self.hdr_slider.setValue(50)  # Default HDR slider value
        self.combine_slider.setValue(50)  # Default combine slider value

        self.toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # Add buttons to the toolbar
        self.toolbar.addAction(self.reset_all_button.text(), self.reset_all)
        self.toolbar.addAction(self.reset_image_button.text(), self.reset_image)

        # Style the toolbar and buttons
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: red;
                border: 4px solid black; /* Toolbar border */
            }
            QToolButton {
                background-color: yellow;
                color: black;
                border: 2px solid black;
                padding: 5px;
                margin: 5px;
                font-size: 14px;
            }
            QToolButton:pressed {
                background-color: orange; /* Change background on press */
                border-color: red;        /* Change border color on press */
                margin: 7px;             /* Slightly increase margin to mimic press effect */
            }
        """)

        # Preprocessing group
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_group.setStyleSheet("background-color: white;")
        preprocess_layout = QVBoxLayout(preprocess_group)
        preprocess_layout.addWidget(self.noise_button)
        preprocess_layout.addWidget(self.noise_slider)
        preprocess_layout.addWidget(self.sharp_button)
        preprocess_layout.addWidget(self.sharp_slider)
        preprocess_layout.addWidget(self.dcp_button)
        preprocess_layout.addWidget(self.dcp_slider)
        preprocess_layout.addWidget(self.hdr_button)
        preprocess_layout.addWidget(self.hdr_slider)
        preprocess_layout.addWidget(self.combine_button)
        preprocess_layout.addWidget(self.combine_slider)
        self.noise_button.setStyleSheet("background-color: yellow;")
        self.sharp_button.setStyleSheet("background-color: yellow;")
        self.dcp_button.setStyleSheet("background-color: yellow;")
        self.hdr_button.setStyleSheet("background-color: yellow;")
        self.combine_button.setStyleSheet("background-color: yellow;")

        button_layout.addWidget(preprocess_group)

        # Image properties label
        self.image_properties_label = QLabel("")
        self.image_properties_label.setAlignment(Qt.AlignCenter)
        self.image_properties_label.setStyleSheet("background-color: lightgray; font-size: 14px; padding: 5px;")
        main_layout.addWidget(self.image_properties_label, 4, 1, 1, 3)

        # Connect buttons to functions
        self.noise_button.clicked.connect(self.apply_noise)
        self.sharp_button.clicked.connect(self.apply_sharpen)
        self.dcp_button.clicked.connect(self.apply_dcp)
        self.hdr_button.clicked.connect(self.apply_hdr)
        self.combine_button.clicked.connect(self.combine_images)
        self.reset_all_button.clicked.connect(self.reset_all)
        self.reset_image_button.clicked.connect(self.reset_image)

        self.noise_slider.valueChanged.connect(self.apply_noise)
        self.sharp_slider.valueChanged.connect(self.apply_sharpen)
        self.dcp_slider.valueChanged.connect(self.apply_dcp)
        self.hdr_slider.valueChanged.connect(self.apply_hdr)
        self.combine_slider.valueChanged.connect(self.combine_images)
        
        flip_button = QToolButton(self)
        flip_button.setText("Flip")
        flip_button.setPopupMode(QToolButton.MenuButtonPopup)
        flip_menu = QMenu(flip_button)

        flip_vertical_action = QAction("Flip Enhanced Vertical", self)
        flip_horizontal_action = QAction("Flip Enhanced Horizontal", self)
        flip_original_vertical_action = QAction("Flip Original Vertical", self)
        flip_original_horizontal_action = QAction("Flip Original Horizontal", self)

        flip_menu.addAction(flip_vertical_action)
        flip_menu.addAction(flip_horizontal_action)
        flip_menu.addAction(flip_original_vertical_action)
        flip_menu.addAction(flip_original_horizontal_action)

        flip_button.setMenu(flip_menu)

        # Create a dropdown for Rotate (Enhanced and Original)
        rotate_button = QToolButton(self)
        rotate_button.setText("Rotate")
        rotate_button.setPopupMode(QToolButton.MenuButtonPopup)
        rotate_menu = QMenu(rotate_button)

        rotate_left_action = QAction("Rotate Enhanced Left 90°", self)
        rotate_right_action = QAction("Rotate Enhanced Right 90°", self)
        rotate_original_left_action = QAction("Rotate Original Left 90°", self)
        rotate_original_right_action = QAction("Rotate Original Right 90°", self)

        rotate_menu.addAction(rotate_left_action)
        rotate_menu.addAction(rotate_right_action)
        rotate_menu.addAction(rotate_original_left_action)
        rotate_menu.addAction(rotate_original_right_action)

        rotate_button.setMenu(rotate_menu)

        # Add the dropdown buttons to the toolbar
        self.toolbar.addWidget(flip_button)
        self.toolbar.addWidget(rotate_button)
        
        # Add the Crop button to the toolbar
        crop_button = QAction("Crop Image", self)
        self.toolbar.addAction(crop_button)
        
        # Connect the button to the cropping function
        crop_button.triggered.connect(self.reset_cropping)           

        # Connect dropdown actions to respective methods
        flip_vertical_action.triggered.connect(self.flip_enhanced_vertical)
        flip_horizontal_action.triggered.connect(self.flip_enhanced_horizontal)
        flip_original_vertical_action.triggered.connect(self.flip_original_vertical)
        flip_original_horizontal_action.triggered.connect(self.flip_original_horizontal)
        
        rotate_left_action.triggered.connect(self.rotate_enhanced_left)
        rotate_right_action.triggered.connect(self.rotate_enhanced_right)
        rotate_original_left_action.triggered.connect(self.rotate_original_left)
        rotate_original_right_action.triggered.connect(self.rotate_original_right)

        self.toggle_button_layout = QPushButton("Image Processing")
        self.toggle_button_layout.setCheckable(True)
        self.toggle_button_layout.setChecked(True)
        self.toggle_button_layout.clicked.connect(self.toggle_button)
        
        self.toggle_filter_layout = QPushButton("Filter")
        self.toggle_filter_layout.setCheckable(True)
        self.toggle_filter_layout.setChecked(False)
        self.toggle_filter_layout.clicked.connect(self.toggle_filter)
        
        self.toggle_edge_layout = QPushButton("Edge Detection")
        self.toggle_edge_layout.setCheckable(True)
        self.toggle_edge_layout.setChecked(False)
        self.toggle_edge_layout.clicked.connect(self.toggle_edge)
        
        self.toggle_bit_layout = QPushButton("Bit Slicing")
        self.toggle_bit_layout.setCheckable(True)
        self.toggle_bit_layout.setChecked(False)
        self.toggle_bit_layout.clicked.connect(self.toggle_bit)
        
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.toggle_button_layout, 0, 0)
        grid_layout.addWidget(self.toggle_filter_layout, 0, 1)
        grid_layout.addWidget(self.toggle_edge_layout, 1, 0)
        grid_layout.addWidget(self.toggle_bit_layout, 1, 1)
        
        self.button_widget = button_widget
        self.filter_widget = filter_widget
        self.edge_widget = edge_widget
        self.bit_widget = bit_widget
        
        # Wrapping layouts in widgets
        self.button_widget.setVisible(self.toggle_button_layout.isChecked()) 
        self.filter_widget.setVisible(self.toggle_filter_layout.isChecked())  
        self.edge_widget.setVisible(self.toggle_edge_layout.isChecked())  
        self.bit_widget.setVisible(self.toggle_bit_layout.isChecked())  

        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)

        self.toolbar.addWidget(grid_widget)
                
        # Create a container widget for the labels and spacer
        container = QWidget(self)
        layout = QHBoxLayout(container)

        # Add spacer between the images
        spacer = QSpacerItem(100, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addSpacerItem(spacer)
                
        label1 = QLabel(self)
        pixmap1 = QPixmap("amirul.jpg")
        scaled_pixmap1 = pixmap1.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label1.setPixmap(scaled_pixmap1)
        layout.addWidget(label1)

        # Add spacer between the images
        spacer = QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addSpacerItem(spacer)

        label2 = QLabel(self)
        pixmap2 = QPixmap("haziq.jpg")
        scaled_pixmap2 = pixmap2.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label2.setPixmap(scaled_pixmap2)
        layout.addWidget(label2)

        # Set layout spacing and margins
        layout.setSpacing(10)  # Adjust space between items
        layout.setContentsMargins(0, 0, 0, 0)

        # Add the container widget to the toolbar
        self.toolbar.addWidget(container)       
        
        # adding text label
        text_label = QLabel("\t\tCreated by : \nMUHAMMAD AMIRUL AQMAL BIN  ABDUL LATIP(BS22110137)\nMUHAMMAD HAZIQ BIN ROSMAN(BS22110061) ", self)
        text_label.setStyleSheet("color: white; font-size: 14px; margin-left: 10px;")
        self.toolbar.addWidget(text_label)         
        
        filterbox_widget = QGroupBox()
        filterbox_widget.setStyleSheet("background-color: white;")
        filterbox_layout =QVBoxLayout(filterbox_widget)
        
        filter_layout.addWidget(filterbox_widget)
        
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)
        self.gamma_slider.setMaximum(40)
        self.gamma_slider.setValue(10)
        self.gamma_slider.valueChanged.connect(self.apply_gamma)

        self.erode_slider = QSlider(Qt.Horizontal)
        self.erode_slider.setMinimum(1)
        self.erode_slider.setMaximum(20)
        self.erode_slider.setValue(3)
        self.erode_slider.valueChanged.connect(self.apply_erode)

        self.remove_gamma_button = QPushButton("Remove Gamma")
        self.remove_gamma_button.setStyleSheet("background-color: yellow;")
        self.remove_gamma_button.clicked.connect(self.remove_gamma)

        self.remove_erode_button = QPushButton("Remove Erosion")
        self.remove_erode_button.setStyleSheet("background-color: yellow;")
        self.remove_erode_button.clicked.connect(self.remove_erode)

        # Layout for image editor
        filterbox_layout.addWidget(QLabel("Gamma Correction"))
        filterbox_layout.addWidget(self.gamma_slider)
        filterbox_layout.addWidget(self.remove_gamma_button)

        filterbox_layout.addWidget(QLabel("Erosion Kernel Size"))
        filterbox_layout.addWidget(self.erode_slider)
        filterbox_layout.addWidget(self.remove_erode_button)
        
        self.dilate_slider = QSlider(Qt.Horizontal)
        self.dilate_slider.setMinimum(1)  # Min kernel size 1x1
        self.dilate_slider.setMaximum(20)  # Max kernel size 20x20
        self.dilate_slider.setValue(3)  # Default kernel size 3x3
        self.dilate_slider.valueChanged.connect(self.apply_dilate)

        self.remove_dilate_button = QPushButton("Remove Dilation")
        self.remove_dilate_button.setStyleSheet("background-color: yellow;")
        self.remove_dilate_button.clicked.connect(self.remove_dilate)

        # Add the Dilation slider and reset button to the layout (imageEditorLayout)
        filterbox_layout.addWidget(QLabel("Dilation Kernel Size"))
        filterbox_layout.addWidget(self.dilate_slider)
        filterbox_layout.addWidget(self.remove_dilate_button)
        
        # Gaussian Filter Slider
        self.gaussian_slider = QSlider(Qt.Horizontal)
        self.gaussian_slider.setMinimum(1)  # Set minimum to 1 (neutral value)
        self.gaussian_slider.setMaximum(11)
        self.gaussian_slider.setValue(1)  # Default to 1 (no effect)
        self.gaussian_slider.setSingleStep(2)
        self.gaussian_slider.valueChanged.connect(self.apply_gaussian)

        self.remove_gaussian_button = QPushButton("Remove Gaussian")
        self.remove_gaussian_button.setStyleSheet("background-color: yellow;")
        self.remove_gaussian_button.clicked.connect(self.remove_gaussian)

        # Median Filter Slider
        self.median_slider = QSlider(Qt.Horizontal)
        self.median_slider.setMinimum(1)
        self.median_slider.setMaximum(11)
        self.median_slider.setValue(1)
        self.median_slider.setSingleStep(2)  # Odd kernel sizes
        self.median_slider.valueChanged.connect(self.apply_median)

        self.remove_median_button = QPushButton("Remove Median")
        self.remove_median_button.setStyleSheet("background-color: yellow;")
        self.remove_median_button.clicked.connect(self.remove_median)

        # Average Filter Slider
        self.avg_slider = QSlider(Qt.Horizontal)
        self.avg_slider.setMinimum(1)
        self.avg_slider.setMaximum(11)
        self.avg_slider.setValue(1)
        self.avg_slider.setSingleStep(2)  # Odd kernel sizes
        self.avg_slider.valueChanged.connect(self.apply_average)

        self.remove_avg_button = QPushButton("Remove Average")
        self.remove_avg_button.setStyleSheet("background-color: yellow;")
        self.remove_avg_button.clicked.connect(self.remove_avg)

        # Bilateral Filter Slider
        self.bilateral_slider = QSlider(Qt.Horizontal)
        self.bilateral_slider.setMinimum(1)
        self.bilateral_slider.setMaximum(11)
        self.bilateral_slider.setValue(1)
        self.bilateral_slider.setSingleStep(2)
        self.bilateral_slider.valueChanged.connect(self.apply_bilateral)

        self.remove_bilateral_button = QPushButton("Remove Bilateral")
        self.remove_bilateral_button.setStyleSheet("background-color: yellow;")
        self.remove_bilateral_button.clicked.connect(self.remove_bilateral)

        # Add all sliders and reset buttons to the layout
        filterbox_layout.addWidget(QLabel("Gaussian Filter"))
        filterbox_layout.addWidget(self.gaussian_slider)
        filterbox_layout.addWidget(self.remove_gaussian_button)

        filterbox_layout.addWidget(QLabel("Median Filter"))
        filterbox_layout.addWidget(self.median_slider)
        filterbox_layout.addWidget(self.remove_median_button)

        filterbox_layout.addWidget(QLabel("Average Filter"))
        filterbox_layout.addWidget(self.avg_slider)
        filterbox_layout.addWidget(self.remove_avg_button)

        filterbox_layout.addWidget(QLabel("Bilateral Filter"))
        filterbox_layout.addWidget(self.bilateral_slider)
        filterbox_layout.addWidget(self.remove_bilateral_button)
        
        self.gaussian_slider.valueChanged.connect(self.apply_gaussian)
        self.median_slider.valueChanged.connect(self.apply_median)
        self.avg_slider.valueChanged.connect(self.apply_average)
        self.bilateral_slider.valueChanged.connect(self.apply_bilateral)
        
        edgebox_widget = QGroupBox()
        edgebox_widget.setStyleSheet("background-color: white;")
        edgebox_layout =QVBoxLayout(edgebox_widget)
        
        edge_layout.addWidget(edgebox_widget)
 
        self.edge_detection_dropdown = QComboBox()
        self.edge_detection_dropdown.setStyleSheet("background-color: yellow;")
        self.edge_detection_dropdown.addItems(["Canny", "Prewitt", "Sobel"])
        edgebox_layout.addWidget(self.edge_detection_dropdown)

        # Canny Edge Detection Sliders
        self.canny_min_slider = QSlider(Qt.Horizontal)
        self.canny_min_slider.setRange(0, 255)
        self.canny_min_slider.setValue(50)
        self.canny_min_slider.setTickInterval(1)
        edgebox_layout.addWidget(QLabel("Canny Min Threshold"))
        edgebox_layout.addWidget(self.canny_min_slider)

        self.canny_max_slider = QSlider(Qt.Horizontal)
        self.canny_max_slider.setRange(0, 255)
        self.canny_max_slider.setValue(150)
        self.canny_max_slider.setTickInterval(1)
        edgebox_layout.addWidget(QLabel("Canny Max Threshold"))
        edgebox_layout.addWidget(self.canny_max_slider)

        # Prewitt/Sobel Kernel Size Slider
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 31)  # Kernel sizes from 1 to 31
        self.kernel_slider.setValue(3)      # Default kernel size
        self.kernel_slider.setTickInterval(2)
        self.kernel_slider.setSingleStep(2)  # Ensure it only moves by odd steps
        edgebox_layout.addWidget(QLabel("Prewitt/Sobel Kernel Size"))
        edgebox_layout.addWidget(self.kernel_slider)

        # Remove Edge Detection Button
        self.remove_edge_button = QPushButton("Remove Edge Detection")
        self.remove_edge_button.setStyleSheet("background-color: yellow;")
        edgebox_layout.addWidget(self.remove_edge_button)
        self.remove_edge_button.clicked.connect(self.remove_edge_detection)
        
        # Connect sliders and dropdown to automatically apply edge detection
        self.canny_min_slider.valueChanged.connect(self.apply_edge_detection)
        self.canny_max_slider.valueChanged.connect(self.apply_edge_detection)
        self.kernel_slider.valueChanged.connect(self.apply_edge_detection)
        self.edge_detection_dropdown.currentIndexChanged.connect(self.apply_edge_detection)
        
         # Threshold Slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.valueChanged.connect(self.apply_thresholding)

        # Dropdown for Thresholding Method
        self.threshold_method_dropdown = QComboBox()
        self.threshold_method_dropdown.setStyleSheet("background-color: yellow;")
        self.threshold_method_dropdown.addItems([
            "THRESH_BINARY",
            "THRESH_BINARY_INV",
            "THRESH_TRUNC",
            "THRESH_TOZERO",
            "THRESH_TOZERO_INV"
        ])
        self.threshold_method_dropdown.currentIndexChanged.connect(self.apply_thresholding)

        # Remove Threshold Button
        self.remove_threshold_button = QPushButton("Remove Threshold")
        self.remove_threshold_button.setStyleSheet("background-color: yellow;")
        self.remove_threshold_button.clicked.connect(self.remove_threshold)

        # Add controls to the layout
        edgebox_layout.addWidget(QLabel("Threshold Method"))
        edgebox_layout.addWidget(self.threshold_method_dropdown)
        edgebox_layout.addWidget(QLabel("Threshold Value"))
        edgebox_layout.addWidget(self.threshold_slider)
        edgebox_layout.addWidget(self.remove_threshold_button)

        # Default threshold method
        self.threshold_method = cv2.THRESH_BINARY
        
        # Add a button for Bit-Plane Slicing
        self.bit_plane_button = QPushButton("Bit-Plane Slicing", self)
        self.bit_plane_button.setStyleSheet("background-color: yellow;")
        self.bit_plane_button.clicked.connect(self.perform_bit_plane_slicing)
        bit_layout.addWidget(self.bit_plane_button)

        # Create a Matplotlib Canvas for Bit-Plane Visualization
        self.bit_plane_canvas = FigureCanvas(Figure(figsize=(4, 2)))
        bit_layout.addWidget(self.bit_plane_canvas)

        # Create a grid layout for bit-plane buttons
        self.bit_plane_button_layout = QGridLayout()
        bit_layout.addLayout(self.bit_plane_button_layout)

        self.bit_plane_buttons = []
        for i in range(8):
            button = QPushButton(f"Bit {i}", self)
            button.setStyleSheet("background-color: yellow;")
            button.clicked.connect(lambda checked, bit=i: self.apply_bit_plane(bit))
            self.bit_plane_button_layout.addWidget(button, 0, i)
            self.bit_plane_buttons.append(button)

        # Add a button to remove applied bit-plane
        self.remove_bit_plane_button = QPushButton("Remove Bit-Plane", self)
        self.remove_bit_plane_button.setStyleSheet("background-color: yellow;")
        self.remove_bit_plane_button.clicked.connect(self.remove_bit_plane)
        bit_layout.addWidget(self.remove_bit_plane_button)
        
        self.player = QMediaPlayer()  # Initialize the media player
        self.is_paused = False  # Track the pause state
        self.player.stateChanged.connect(self.handle_music_loop)
        
    def toggle_button(self):
        # If the button layout is turned on, turn off the others
        if self.toggle_button_layout.isChecked():
            self.button_widget.setVisible(True)
            self.filter_widget.setVisible(False)
            self.edge_widget.setVisible(False)
            self.bit_widget.setVisible(False)
            
            self.toggle_filter_layout.setChecked(False)
            self.toggle_edge_layout.setChecked(False)
            self.toggle_bit_layout.setChecked(False)
        else:
            self.button_widget.setVisible(False)

    def toggle_filter(self):
        # If the filter layout is turned on, turn off the others
        if self.toggle_filter_layout.isChecked():
            self.filter_widget.setVisible(True)
            self.button_widget.setVisible(False)
            self.edge_widget.setVisible(False)
            self.bit_widget.setVisible(False)
            
            self.toggle_button_layout.setChecked(False)
            self.toggle_edge_layout.setChecked(False)
            self.toggle_bit_layout.setChecked(False)
        else:
            self.filter_widget.setVisible(False)

    def toggle_edge(self):
        # If the edge layout is turned on, turn off the others
        if self.toggle_edge_layout.isChecked():
            self.edge_widget.setVisible(True)
            self.button_widget.setVisible(False)
            self.filter_widget.setVisible(False)
            self.bit_widget.setVisible(False)
            
            self.toggle_button_layout.setChecked(False)
            self.toggle_filter_layout.setChecked(False)
            self.toggle_bit_layout.setChecked(False)
        else:
            self.edge_widget.setVisible(False)

    def toggle_bit(self):
        # If the bit layout is turned on, turn off the others
        if self.toggle_bit_layout.isChecked():
            self.bit_widget.setVisible(True)
            self.button_widget.setVisible(False)
            self.filter_widget.setVisible(False)
            self.edge_widget.setVisible(False)
            
            self.toggle_button_layout.setChecked(False)
            self.toggle_filter_layout.setChecked(False)
            self.toggle_edge_layout.setChecked(False)
        else:
            self.bit_widget.setVisible(False)              

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load Image", self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        new_window_action = QAction("New Window", self)
        new_window_action.triggered.connect(self.open_new_window)
        file_menu.addAction(new_window_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action) 

        # Music Menu
        music_menu = menubar.addMenu("Music")
        music_action = QAction("Music Player", self)
        music_action.triggered.connect(self.play_music)
        music_menu.addAction(music_action)
        
        pause_music_action = QAction("Pause/Resume Music", self)
        pause_music_action.triggered.connect(self.toggle_music_pause)
        music_menu.addAction(pause_music_action)
        
    def load_image(self):
        # Loads an image and displays it with its properties and histogram
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if fname:
            self.loaded_image = cv2.imread(fname)
            self.loaded_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)

            # Initialize both original and enhanced images
            self.original_image = self.loaded_image.copy()
            self.enhanced_image = self.loaded_image.copy()

            # Display the image
            pixmap = self.convert_cv_to_pixmap(self.original_image)
            scaled_pixmap = pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_label.setPixmap(scaled_pixmap)
            self.update_histogram(self.original_image, self.original_hist_canvas, "Original Histogram")

            # Extract and display image properties
            file_size = os.path.getsize(fname) / 1024  # Size in KB
            resolution = f"{self.original_image.shape[1]} x {self.original_image.shape[0]}"
            file_type = os.path.splitext(fname)[-1].upper().strip(".")
            self.image_properties_label.setText(
                f"File Name: {os.path.basename(fname)} | File Size: {file_size:.2f} KB | Resolution: {resolution} | Image Type: {file_type}"
            )

            # Calculate SSIM and PSNR for the original image
            ssim_value = self.calculate_ssim(self.original_image, self.original_image)
            psnr_value = self.calculate_psnr(self.original_image, self.original_image)

            # Update original metrics labels
            self.original_ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.original_psnr_label.setText(f"PSNR: {psnr_value:.2f} dB")

    def calculate_ssim(self, imageA, imageB):
        # Compute SSIM between two images
        grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
        (score, _) = compare_ssim(grayA, grayB, full=True)
        return score

    def calculate_psnr(self, original, compressed):
        # Compute PSNR between two images
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def save_image(self):
        # Saves the enhanced image to a file
        if self.enhanced_image is not None:
            fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)")
            if fname:
                cv2.imwrite(fname, cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2BGR))
                    
    def reset_all(self):
        # Resets the application to its initial state
        # Reset the original and enhanced images
        self.original_image = None
        self.enhanced_image = None

        # Clear image labels
        self.original_label.clear()
        self.enhanced_label.clear()

        # Clear histograms
        self.original_hist_canvas.axes.clear()
        self.enhanced_hist_canvas.axes.clear()
        self.original_hist_canvas.draw()
        self.enhanced_hist_canvas.draw()

        # Reset SSIM and PSNR labels
        self.original_ssim_label.setText("SSIM: N/A")
        self.original_psnr_label.setText("PSNR: N/A")
        self.enhanced_ssim_label.setText("SSIM: N/A")
        self.enhanced_psnr_label.setText("PSNR: N/A")

        # Reset image properties label
        self.image_properties_label.setText("File Name: N/A | File Size: N/A | Resolution: N/A | Image Type: N/A")

        # Reset all sliders to their default positions (e.g., 0)
        self.noise_slider.setValue(50)
        self.sharp_slider.setValue(50)
        self.dcp_slider.setValue(50)
        self.hdr_slider.setValue(50)
        self.combine_slider.setValue(50)

    def reset_image(self):
        # Resets the enhanced image to the original image and resets sliders and cropping
        if self.loaded_image is not None:
            # Reset the original and enhanced images
            self.original_image = self.loaded_image.copy()
            self.enhanced_image = self.loaded_image.copy()

            # Update displays
            self.update_original_image()
            self.update_enhanced_image()

            # Reset cropping state
            self.cropping = False
            self.crop_start = None
            self.crop_end = None

            # Reset all sliders to their default positions
            self.noise_slider.setValue(50)
            self.sharp_slider.setValue(50)
            self.dcp_slider.setValue(50)
            self.hdr_slider.setValue(50)
            self.combine_slider.setValue(50)

    def apply_noise(self):
        # Adds noise to the original image based on the slider value
        if self.original_image is not None:
            # Get the slider value to control noise intensity
            noise_level = self.noise_slider.value()  # Range: 0-100
            noise = np.random.normal(0, noise_level, self.original_image.shape)
            noisy_image = self.original_image + noise
            self.enhanced_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            self.update_enhanced_image()

    def apply_sharpen(self):
        # Applies sharpening to the original image based on the slider value
        if self.original_image is not None:
            sharpness_level = self.sharp_slider.value() / 100.0  # Normalize to [0, 1]
            kernel = np.array([[0, -sharpness_level, 0], 
                            [-sharpness_level, 1 + 4 * sharpness_level, -sharpness_level], 
                            [0, -sharpness_level, 0]])
            self.enhanced_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_enhanced_image()

    def apply_dcp(self):
        # Applies dark channel prior enhancement based on the slider value.
        if self.original_image is not None:
            dcp_intensity = self.dcp_slider.value() / 100.0  # Normalize to [0, 1]
            dark_channel = np.min(self.original_image, axis=2)
            dcp_image = cv2.normalize(dark_channel, None, 0, 255 * dcp_intensity, cv2.NORM_MINMAX)
            self.enhanced_image = cv2.applyColorMap(dcp_image.astype(np.uint8), cv2.COLORMAP_JET)
            self.update_enhanced_image()

    def apply_hdr(self):
        # Applies HDR enhancement based on the slider value
        if self.original_image is not None:
            hdr_intensity = self.hdr_slider.value() / 100.0  # Normalize to [0, 1]
            hdr = cv2.detailEnhance(self.original_image, sigma_s=12, sigma_r=hdr_intensity)
            self.enhanced_image = hdr
            self.update_enhanced_image()

    def combine_images(self):
        # Combines DCP and HDR results based on the slider value
        if self.original_image is not None:
            combine_ratio = self.combine_slider.value() / 100.0  # Normalize to [0, 1]
            
            # Apply DCP
            dark_channel = np.min(self.original_image, axis=2)
            dcp_image = cv2.normalize(dark_channel, None, 0, 255, cv2.NORM_MINMAX)
            dcp_image = cv2.applyColorMap(dcp_image.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Apply HDR
            hdr_image = cv2.detailEnhance(self.original_image, sigma_s=12, sigma_r=0.15)
            
            # Combine DCP and HDR with slider-based weights
            self.enhanced_image = cv2.addWeighted(dcp_image, combine_ratio, hdr_image, 1 - combine_ratio, 0)
            self.update_enhanced_image()

    def update_enhanced_image(self):
        # Updates the enhanced image display and metrics
        # Convert the enhanced image to a QPixmap and scale it to fit the label
        pixmap = self.convert_cv_to_pixmap(self.enhanced_image)
        scaled_pixmap = pixmap.scaled(self.enhanced_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.enhanced_label.setPixmap(scaled_pixmap)

        # Update the enhanced histogram
        self.update_histogram(self.enhanced_image, self.enhanced_hist_canvas, "Enhanced Histogram")

        # Calculate SSIM and PSNR only if dimensions match
        if self.original_image is not None and self.enhanced_image is not None:
            if self.original_image.shape == self.enhanced_image.shape:
                # Calculate SSIM using the required method
                ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
                
                # Calculate PSNR using the required method
                psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)
                
                # Update SSIM and PSNR labels
                self.enhanced_ssim_label.setText(f"SSIM: {ssim_value:.4f}")
                self.enhanced_psnr_label.setText(f"PSNR: {psnr_value:.2f} dB")
            else:
                # Handle case where dimensions do not match
                self.enhanced_ssim_label.setText("SSIM: N/A")
                self.enhanced_psnr_label.setText("PSNR: N/A")

    def update_histogram(self, image, canvas, title):
        # Updates the histogram for the given image
        if image is not None:
            canvas.plot_histogram(image, title)

    def convert_cv_to_pixmap(self, cv_image):
        # Converts an OpenCV image to QPixmap
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        # Convert memoryview to bytes before passing to QImage
        q_image = QImage(cv_image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def flip_enhanced_horizontal(self):
        # Flip the enhanced image horizontally and update the display
        if self.enhanced_image is not None:
            self.enhanced_image = cv2.flip(self.enhanced_image, 1)  # Flip horizontally
            self.update_enhanced_image()

    def flip_enhanced_vertical(self):
        # Flip the enhanced image vertically and update the display
        if self.enhanced_image is not None:
            self.enhanced_image = cv2.flip(self.enhanced_image, 0)  # Flip vertically
            self.update_enhanced_image()

    def flip_original_horizontal(self):
        # Flip the original image horizontally and update the display
        if self.original_image is not None:
            self.original_image = cv2.flip(self.original_image, 1)  # Flip horizontally
            self.update_original_image()

    def flip_original_vertical(self):
        # Flip the original image vertically and update the display
        if self.original_image is not None:
            self.original_image = cv2.flip(self.original_image, 0)  # Flip vertically
            self.update_original_image()

    def rotate_enhanced_left(self):
        # Rotate the enhanced image 90° to the left and update the display
        if self.enhanced_image is not None:
            self.enhanced_image = cv2.rotate(self.enhanced_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.update_enhanced_image()

    def rotate_enhanced_right(self):
        # Rotate the enhanced image 90° to the right and update the display
        if self.enhanced_image is not None:
            self.enhanced_image = cv2.rotate(self.enhanced_image, cv2.ROTATE_90_CLOCKWISE)
            self.update_enhanced_image()

    def rotate_original_left(self):
        # Rotate the original image 90° to the left and update the display
        if self.original_image is not None:
            self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.update_original_image()

    def rotate_original_right(self):
        # Rotate the original image 90° to the right and update the display
        if self.original_image is not None:
            self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
            self.update_original_image()

    def recalculate_metrics(self):
        # Recalculate SSIM and PSNR and update the GUI labels
        if self.original_image is not None and self.enhanced_image is not None:
            # Resize the enhanced image to match the original image's dimensions
            enhanced_resized = cv2.resize(self.enhanced_image, 
                                        (self.original_image.shape[1], self.original_image.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            
            # Convert images to grayscale if necessary
            original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) if len(self.original_image.shape) == 3 else self.original_image
            enhanced_gray = cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2GRAY) if len(enhanced_resized.shape) == 3 else enhanced_resized
            
            # Compute SSIM
            ssim_value = compare_ssim(original_gray, enhanced_gray)
            self.enhanced_ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            
            # Compute PSNR
            psnr_value = cv2.PSNR(self.original_image, enhanced_resized)
            self.enhanced_psnr_label.setText(f"PSNR: {psnr_value:.2f} dB")
                
    def update_original_image(self):
        # Updates the original image display
        # Convert the original image to a QPixmap and scale it to fit the label
        pixmap = self.convert_cv_to_pixmap(self.original_image)
        scaled_pixmap = pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_label.setPixmap(scaled_pixmap)
        
    def open_new_window(self):
        # Create a new MainWindow with the specified size
        new_window = ImageEnhancer()
        ImageEnhancer.open_windows.append(new_window)  # Keep a reference
        new_window.show()
            
    # Gamma Correction
    def apply_gamma(self):
        # Apply gamma correction based on the slider value
        if self.original_image is not None:
            gamma_value = self.gamma_slider.value() / 10.0  # Adjust the slider to gamma scale
            inv_gamma = 1.0 / gamma_value
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            self.enhanced_image = cv2.LUT(self.original_image, table)
            self.update_enhanced_image()

    # Erosion
    def apply_erode(self):
        # Apply erosion based on the slider value
        if self.original_image is not None:
            kernel_size = self.erode_slider.value()
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.enhanced_image = cv2.erode(self.original_image, kernel, iterations=1)
            self.update_enhanced_image()

    # Dilation
    def apply_dilate(self):
        # Apply dilation based on the slider value
        if self.original_image is not None:
            kernel_size = self.dilate_slider.value()
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.enhanced_image = cv2.dilate(self.original_image, kernel, iterations=1)
            self.update_enhanced_image()

    # Gaussian Filter
    def apply_gaussian(self):
        # Apply Gaussian blur based on the slider value
        if self.original_image is not None:
            kernel_size = self.gaussian_slider.value()
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            self.enhanced_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
            self.update_enhanced_image()

    # Median Filter
    def apply_median(self):
        # Apply Median blur based on the slider value
        if self.original_image is not None:
            kernel_size = self.median_slider.value()
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            self.enhanced_image = cv2.medianBlur(self.original_image, kernel_size)
            self.update_enhanced_image()

    # Average Filter
    def apply_average(self):
        # Apply Average blur (mean filter) based on the slider value
        if self.original_image is not None:
            kernel_size = self.avg_slider.value()
            self.enhanced_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
            self.update_enhanced_image()

    # Bilateral Filter
    def apply_bilateral(self):
        # Apply Bilateral filter based on the slider value
        if self.original_image is not None:
            diameter = self.bilateral_slider.value()
            self.enhanced_image = cv2.bilateralFilter(self.original_image, diameter, 75, 75)
            self.update_enhanced_image()

    # Reset functions for each filter
    def remove_gamma(self):
        self.gamma_slider.setValue(10)  # Reset to default
        self.update_enhanced_image()

    def remove_erode(self):
        self.erode_slider.setValue(3)  # Reset to default
        self.update_enhanced_image()

    def remove_dilate(self):
        self.dilate_slider.setValue(3)  # Reset to default
        self.update_enhanced_image()

    def remove_gaussian(self):
        self.gaussian_slider.setValue(1)  # Reset to default
        self.update_enhanced_image()

    def remove_median(self):
        self.median_slider.setValue(1)  # Reset to default
        self.update_enhanced_image()

    def remove_avg(self):
        self.avg_slider.setValue(1)  # Reset to default
        self.update_enhanced_image()

    def remove_bilateral(self):
        self.bilateral_slider.setValue(1)  # Reset to default
        self.update_enhanced_image()
        
    def apply_edge_detection(self):
        # Applies the selected edge detection method to the original image
        if self.original_image is None:
            return

        edge_type = self.edge_detection_dropdown.currentText()
        kernel_size = self.kernel_slider.value()

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        edges = None  # Initialize edges variable

        if edge_type == "Canny":
            # Canny Edge Detection
            min_thresh = self.canny_min_slider.value()
            max_thresh = self.canny_max_slider.value()
            edges = cv2.Canny(gray_image, min_thresh, max_thresh)

        elif edge_type == "Sobel":
            # Sobel Edge Detection
            sobelx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=kernel_size)
            sobel_combined = cv2.magnitude(sobelx, sobely)
            edges = np.uint8(np.clip(sobel_combined / np.max(sobel_combined) * 255, 0, 255))

        elif edge_type == "Prewitt":
            # Generate dynamic Prewitt kernels based on the kernel size
            kernel = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
            kernelx = np.outer(kernel, [1, 0, -1])  # Horizontal gradient kernel
            kernely = np.outer([1, 0, -1], kernel)  # Vertical gradient kernel

            # Apply the Prewitt operator
            edges_x = cv2.filter2D(gray_image, cv2.CV_32F, kernelx)
            edges_y = cv2.filter2D(gray_image, cv2.CV_32F, kernely)

            # Compute gradient magnitude
            edges_combined = cv2.magnitude(edges_x, edges_y)
            edges = np.uint8(np.clip(edges_combined / np.max(edges_combined) * 255, 0, 255))

        # If edges were detected, update the enhanced image
        if edges is not None:
            # Convert single-channel edge-detected image to RGB for consistency
            self.enhanced_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.update_enhanced_image()
            
    def remove_edge_detection(self):
        if self.original_image is not None:
            self.enhanced_image = self.original_image.copy()
            self.update_enhanced_image()

            self.canny_min_slider.setValue(50)
            self.canny_max_slider.setValue(150)
            self.kernel_slider.setValue(3)
            
            self.edge_detection_dropdown.setCurrentIndex(0)
            
    def apply_thresholding(self):
        # Applies thresholding based on the slider value and selected method
        if self.original_image is None:
            return

        # Get the threshold value from the slider
        threshold_value = self.threshold_slider.value()

        # Get the selected thresholding method from the dropdown
        selected_method = self.threshold_method_dropdown.currentText()

        # Map the selected method to OpenCV constants
        method_mapping = {
            "THRESH_BINARY": cv2.THRESH_BINARY,
            "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
            "THRESH_TRUNC": cv2.THRESH_TRUNC,
            "THRESH_TOZERO": cv2.THRESH_TOZERO,
            "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,
        }
        threshold_method = method_mapping.get(selected_method, cv2.THRESH_BINARY)

        # Convert to grayscale if not already
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        # Apply the thresholding method
        _, thresh_image = cv2.threshold(gray_image, threshold_value, 255, threshold_method)

        # Convert back to RGB for display purposes
        self.enhanced_image = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)
        self.update_enhanced_image()

    def remove_threshold(self):
        # Restores the original image, removing the thresholding effect
        if self.original_image is not None:
            self.enhanced_image = self.original_image.copy()
            self.update_enhanced_image()
            
    def perform_bit_plane_slicing(self):
        # Performs bit-plane slicing on the loaded image and displays the results
        if self.original_image is None:
            return

        # Ensure the image is grayscale
        if len(self.original_image.shape) == 3:
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        else:
            grayscale_image = self.original_image

        # Extract bit planes
        self.bit_planes = [(grayscale_image >> i) & 1 for i in range(8)]

        # Display bit planes on the canvas
        self.display_bit_planes_on_canvas(self.bit_planes)


    def extract_bit_planes(self, image):
        # Extracts all 8 bit-planes from a grayscale image
        return [(image >> i) & 1 * 255 for i in range(8)]

    def display_bit_planes_on_canvas(self, bit_planes):
        # Displays the bit-plane images on the dedicated Matplotlib canvas in a 4x2 grid
        self.bit_plane_canvas.figure.clear()  # Clear previous plots
        axes = self.bit_plane_canvas.figure.subplots(4, 2)  # Create a 4x2 grid
        axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            if i < len(bit_planes):
                ax.imshow(bit_planes[i], cmap='gray')
                ax.set_title(f'Bit {i}')
                ax.axis('off')  # Remove axis
            else:
                ax.axis('off')  # Hide unused subplots

        self.bit_plane_canvas.figure.tight_layout()
        self.bit_plane_canvas.draw()

    def apply_bit_plane(self, bit):
        # Applies a specific bit-plane to the loaded image
        if self.original_image is None:
            return

        # Ensure the image is grayscale
        grayscale_image = self.original_image
        if len(self.original_image.shape) == 3:
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        # Extract and apply the specific bit-plane
        bit_plane = ((grayscale_image >> bit) & 1) * 255
        self.enhanced_image = cv2.cvtColor(bit_plane.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Update the display
        self.update_enhanced_image()

    def remove_bit_plane(self):
        # Restores the original image, removing the bit-plane effect
        if self.original_image is not None:
            self.enhanced_image = self.original_image.copy()
            self.update_enhanced_image()
            
    def play_music(self):
        # Plays a hardcoded music file
        music_file = "lofimusic.mp3"  # Update this to your music file's path
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(music_file)))
        self.player.setVolume(50)  # Set default volume to 50%
        self.player.play()

    def toggle_music_pause(self):
        # Pauses or resumes the music playback
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.is_paused = True
        elif self.player.state() == QMediaPlayer.PausedState and self.is_paused:
            self.player.play()
            self.is_paused = False

    def handle_music_loop(self, state):
        # Restarts the music when playback ends
        if state == QMediaPlayer.StoppedState:
            self.player.play()
        
    def start_crop(self, event):
        # Start cropping by recording the initial mouse position
        if event.button() == Qt.LeftButton and self.original_image is not None:
            self.cropping = True
            self.crop_start = (event.x(), event.y())

    def update_crop(self, event):
        # Update the crop rectangle as the mouse moves
        if self.cropping and self.original_image is not None:
            self.crop_end = (event.x(), event.y())
            self.update_crop_rectangle()

    def finish_crop(self, event):
        # Complete the cropping and update the image
        if event.button() == Qt.LeftButton and self.cropping:
            self.cropping = False
            self.crop_end = (event.x(), event.y())
            self.perform_crop()

    def update_crop_rectangle(self):
        # Visual feedback for the crop rectangle
        if self.crop_start and self.crop_end:
            # Get the starting and ending coordinates
            start_x, start_y = self.crop_start
            end_x, end_y = self.crop_end

            # Ensure proper coordinates
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)

            # Create a copy of the original image
            overlay_image = self.original_image.copy()

            # Scale the coordinates to match the original image resolution
            x_scale = self.original_image.shape[1] / self.original_label.width()
            y_scale = self.original_image.shape[0] / self.original_label.height()
            x1, y1, x2, y2 = int(x1 * x_scale), int(y1 * y_scale), int(x2 * x_scale), int(y2 * y_scale)

            # Draw the rectangle on the copy
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Update the QLabel with the updated image
            pixmap = self.convert_cv_to_pixmap(overlay_image)
            scaled_pixmap = pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_label.setPixmap(scaled_pixmap)

    def perform_crop(self):
        # Crop the original image and update the enhanced image
        if self.crop_start and self.crop_end:
            start_x, start_y = self.crop_start
            end_x, end_y = self.crop_end

            # Ensure proper coordinates
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)

            # Scale coordinates to match the original image resolution
            x_scale = self.original_image.shape[1] / self.original_label.width()
            y_scale = self.original_image.shape[0] / self.original_label.height()

            x1, y1 = int(x1 * x_scale), int(y1 * y_scale)
            x2, y2 = int(x2 * x_scale), int(y2 * y_scale)

            # Crop the original image
            cropped_image = self.original_image[y1:y2, x1:x2]
            self.original_image = cropped_image
            self.enhanced_image = cropped_image.copy()

            # Update displays
            self.update_original_image()
            self.update_enhanced_image()

    def convert_pixmap_to_cv(self, pixmap):
        # Converts QPixmap to OpenCV image
        image = pixmap.toImage()
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Assuming RGBA
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)        

    def reset_cropping(self):
        # Resets cropping state and clears any temporary cropping overlays
        self.cropping = False
        self.crop_start = None
        self.crop_end = None
        if self.original_image is not None:
            self.update_original_image()  # Refresh the original image display

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Splash screen setup
    splash_pix = QPixmap('logo.png')  # Add a path to an image you want to display as splash screen
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()    
    
    # Simulate some loading time
    QTimer.singleShot(2500, splash.close)  # 2.5-second delay before closing splash screen
        
    window = ImageEnhancer()
    window.show()
    sys.exit(app.exec_())
