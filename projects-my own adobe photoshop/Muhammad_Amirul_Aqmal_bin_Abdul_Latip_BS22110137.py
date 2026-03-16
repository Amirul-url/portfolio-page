import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMenu, QAction, QPushButton, QGridLayout, QRubberBand, QComboBox,
    QToolBar, QFileDialog, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QDialog, QSlider,
    QScrollArea, QGroupBox, QGridLayout, QMessageBox, QInputDialog, QColorDialog, QLineEdit,  QSplashScreen,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QIcon
from PyQt5.QtCore import QSize, Qt, QPoint, QEvent, QRect, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from convert_rgbimage import RgbImage  # Import the RgbImage class
from convert_gray import GrayImage  # Import the GrayImage class
from convert_hsv import HsvImage  # Import the HsvImage class
from convert_cie import CieImage  # Import the CieImage class
from convert_hls import HlsImage  # Import the HlsImage class
from convert_ycrcb import YCrCbImage  # Import the YCrCbImage class
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  
from matplotlib.figure import Figure

class HistogramWindow(QWidget):
    """A separate window to display the histogram of the loaded image."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Histogram")
        self.setGeometry(1250, 650, 650, 300)

        # Set up layout and figure for the histogram
        self.layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

class PaintingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_zoom_factor = 1.0      # Current zoom level of the canvas
        self.drawing = False                # Flag to enable/disable drawing mode
        self.pen_color = QColor(Qt.black)   # Default pen color for drawing
        self.pen_width = 3                  # Set initial pen width
        self.text_mode = False              # Flag to enable/disable text input mode
        self.text_to_draw = ""              # Text string for drawing on the canvas
        self.circle_mode = False            # Enable/disable circle drawing mode  
        self.original_images = []           # Store original loaded images for resetting 
        self.cropping = False               # Track cropping mode activation
        self.crop_start = QPoint()          # Start point for cropping
        self.crop_end = QPoint()            # End point for cropping
        self.music_playing = False          # Music playback status
        self.thumbnail_images = []          # Store generated thumbnails
        self.thumbnail_size = QSize(100, 100)  # Set the size of the thumbnails  
        self.thumbnail_layout = QVBoxLayout(self)      
        self.ruler_band = QRubberBand(QRubberBand.Line, self) 
        self.ruler_mode = False
        self.gridlines_mode = False
        self.ruler_start = QPoint()
        self.current_color_mode = 'rgb'  # Default color mode

        # Label to display image properties
        self.image_properties_label = QLabel(self)  # Label to show image properties
        self.image_properties_label.setAlignment(Qt.AlignLeft)  # Align to the left
        self.image_properties_label.setStyleSheet("font-size: 12px; color: black; padding: 10px; background-color: #f0f0f0;")  # Customize style

        # Initialize music player settings
        self.media_player = QMediaPlayer()
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile("chill_coding.mp3")))
        self.media_player.setVolume(100)
        self.media_player.mediaStatusChanged.connect(self.loop_music)  # Loop the music when playback ends

        # Undo/Redo Stacks
        self.undo_stack = []
        self.redo_stack = []
        self.save_canvas_state() 

        # Configure the main application window
        self.setWindowTitle("ArtGen Painter Photoshop")
        self.setGeometry(0, 0, 1900, 950)
        self.setStyleSheet("background-color: #f0f0f0;")  # Background color of the main window

        self.rgb_images = []  # List for storing RGB-converted images

        # Initialize main user interface and default canvas setup
        self.initUI()
        self.create_default_canvas(800, 600)

    def initUI(self):
        # Set up the main layout and widgets
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
            
        # Add a histogram area to the right side of the main layout
        self.histogram_canvas = FigureCanvas(Figure(figsize=(12, 3)))
        self.histogram_axes = self.histogram_canvas.figure.add_subplot(111)
        main_layout.addWidget(self.histogram_canvas, alignment=Qt.AlignCenter)        

        # Create and configure the main menu bar
        menu_bar = self.menuBar()
        menu_bar.setFont(QFont("Calibri", 11))

        # Adding 'File' menu options
        file_menu = menu_bar.addMenu("File")

        # Add 'New Canvas' option to 'File' menu and connect to create_new_canvas method
        new_canvas_action = QAction("New Canvas", self)
        new_canvas_action.triggered.connect(self.create_new_canvas)
        file_menu.addAction(new_canvas_action)

        # Add 'Open Image' option to load images and display them on the canvas
        open_image_action = QAction("Open Image", self)
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)

        # Add 'Save' action to save the current state of the canvas
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_canvas)
        file_menu.addAction(save_action)

        # Add 'Image Properties' action to show metadata of the current image
        image_properties_action = QAction("Image Properties", self)
        image_properties_action.triggered.connect(self.show_image_properties)
        file_menu.addAction(image_properties_action)

        # Add 'Exit' action to close the application
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.confirm_exit)
        file_menu.addAction(exit_action)
        
        # Create a new group box for edit actions (Undo, Redo, Clear Drawing)
        edit_group_box = QGroupBox("Edit Tools")
        edit_group_box.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        edit_layout = QGridLayout(edit_group_box)

        undo_button = QPushButton("")
        undo_button.setIcon(QIcon('undo.png'))  # Inserted icon
        undo_button.setIconSize(QSize(30,30))
        undo_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        undo_button.setFixedSize(80, 50)
        undo_button.clicked.connect(self.undo)
        edit_layout.addWidget(undo_button, 0, 0)

        redo_button = QPushButton("")
        redo_button.setIcon(QIcon('redo.png'))  # Inserted icon
        redo_button.setIconSize(QSize(30,30))
        redo_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        redo_button.setFixedSize(80, 50)
        redo_button.clicked.connect(self.redo)
        edit_layout.addWidget(redo_button, 0, 1)

        clear_button = QPushButton("")
        clear_button.setIcon(QIcon('clear.png'))  # Inserted icon
        clear_button.setIconSize(QSize(30,30))
        clear_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        clear_button.setFixedSize(80, 50)
        clear_button.clicked.connect(self.clear)
        edit_layout.addWidget(clear_button, 0, 2)
        
        # Thumbnail Generation Button
        thumbnail_button = QPushButton("")
        thumbnail_button.setIcon(QIcon('ArtGenPP.png'))  # Inserted icon 
        thumbnail_button.setIconSize(QSize(50,50))
        thumbnail_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        thumbnail_button.setFixedSize(80, 50)
        thumbnail_button.clicked.connect(self.generate_thumbnail)  # Call to thumbnail generation function
        edit_layout.addWidget(thumbnail_button, 0, 3)
  
        edit_toolbar = QToolBar("Edit Tools")
        edit_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        edit_toolbar.addWidget(edit_group_box)
        edit_toolbar.setAllowedAreas(Qt.TopToolBarArea | Qt.LeftToolBarArea)
        self.addToolBar(edit_toolbar)

        # Create a new group box for image actions (Crop, Combine)
        image_group_box = QGroupBox("Image Tools")
        image_group_box.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        image_layout = QGridLayout(image_group_box)

        crop_button = QPushButton("")
        crop_button.setIcon(QIcon('crop.png'))  # Inserted icon
        crop_button.setIconSize(QSize(30,30))
        crop_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        crop_button.setFixedSize(80, 50)
        crop_button.clicked.connect(self.enable_crop_mode)
        image_layout.addWidget(crop_button, 0, 0)

        combine_button = QPushButton("")
        combine_button.setIcon(QIcon('combine.png'))  # Inserted icon
        combine_button.setIconSize(QSize(30,30))
        combine_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        combine_button.setFixedSize(80, 50)
        combine_button.clicked.connect(self.combine_image)
        image_layout.addWidget(combine_button, 0, 1)

        image_toolbar = QToolBar("Image Tools")
        image_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        image_toolbar.addWidget(image_group_box)
        self.addToolBar(image_toolbar)

        # Drawing tools group
        drawing_group = QGroupBox("Drawing Tools")
        drawing_group.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        drawing_layout = QGridLayout(drawing_group)

        pen_button = QPushButton("")
        pen_button.setIcon(QIcon('pen.png'))  # Inserted icon
        pen_button.setIconSize(QSize(30,30))
        pen_button.setStyleSheet("background-color: #D3C5E5; color: white;")
        pen_button.setFixedSize(80, 50)
        pen_button.clicked.connect(self.toggle_pen)
        drawing_layout.addWidget(pen_button, 0, 0)
        
        text_button = QPushButton("")
        text_button.setIcon(QIcon('text.png'))  # Inserted icon
        text_button.setIconSize(QSize(30,30))
        text_button.setStyleSheet("background-color: #D3C5E5; color: white;")
        text_button.setFixedSize(80, 50)
        text_button.clicked.connect(self.enable_text_input)
        drawing_layout.addWidget(text_button, 0, 1)

        circle_button = QPushButton("")
        circle_button.setIcon(QIcon('circle.png'))  # Inserted icon
        circle_button.setIconSize(QSize(30,30))
        circle_button.setStyleSheet("background-color: #D3C5E5; color: white;")
        circle_button.setFixedSize(80, 50)
        circle_button.clicked.connect(self.toggle_circle_mode)
        drawing_layout.addWidget(circle_button, 0, 2)

        reset_button = QPushButton("RESET")
        reset_button.setIcon(QIcon('reset.png'))  # Inserted icon
        reset_button.setIconSize(QSize(30,30))
        reset_button.setStyleSheet("background-color: #D3C5E5; color: black; padding: 10px; font: bold;")
        reset_button.setFixedSize(80, 50)
        reset_button.clicked.connect(self.reset)
        drawing_layout.addWidget(reset_button, 0, 3)

        drawing_toolbar = QToolBar("Drawing Tools")
        drawing_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        drawing_toolbar.addWidget(drawing_group)

        # Color palette next to the drawing toolbar
        color_palette = QWidget()
        color_palette_layout = QGridLayout(color_palette)
        color_palette_layout.setSpacing(5)
        colors = [
            "#000000", "#550000", "#aa0000", "#ff0000", "#005500", "#555500", "#aa5500", "#ff5500", "#00aa00", "#55aa00", "#aaaa00", "#ffaa00", "#00ff00", "#55ff00", "#aaff00", "#ffff00",
            "#00007f", "#55007f", "#aa007f", "#ff007f", "#00557f", "#55557f", "#aa557f", "#ff557f", "#00aa7f", "#55aa7f", "#aaaa7f", "#ffaa7f", "#00ff7f", "#55ff7f", "#aaff7f", "#ffff7f",
            "#0000ff", "#5500ff", "#aa00ff", "#ff00ff", "#0055ff", "#5555ff", "#aa55ff", "#ff55ff", "#00aaff", "#55aaff", "#aaaaff", "#ffaaff", "#00ffff", "#55ffff", "#aaffff", "#ffffff"
        ]
        
        # Add color buttons to the color palette
        for i, color in enumerate(colors):
            color_button = QPushButton()
            color_button.setStyleSheet(f"background-color: {color};")
            color_button.setFixedSize(25, 25)
            color_button.clicked.connect(lambda _, clr=color: self.set_pen_color(clr))
            color_palette_layout.addWidget(color_button, i // 16, i % 16)

        # Create a widget to hold both the color palette and the reset button
        color_and_reset_widget = QWidget()
        color_and_reset_layout = QHBoxLayout(color_and_reset_widget)
        color_and_reset_layout.addWidget(color_palette)
        # Move the reset button next to the color palette
        color_and_reset_layout.addWidget(reset_button)
        color_and_reset_layout.setContentsMargins(0, 0, 0, 0)
        color_and_reset_layout.setSpacing(10)

        drawing_toolbar.addWidget(color_and_reset_widget)
        self.addToolBar(drawing_toolbar)   

        # create VIEW_TOOLBAR and add it to the left side, below EDIT_TOOLBAR
        view_group_box = QGroupBox("View Tools")
        view_group_box.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        view_layout = QGridLayout(view_group_box)

        zoom_in_button = QPushButton("")
        zoom_in_button.setIcon(QIcon('zoom in.png'))  # Inserted icon
        zoom_in_button.setIconSize(QSize(30,30))
        zoom_in_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        zoom_in_button.setFixedSize(90, 50)
        zoom_in_button.clicked.connect(self.zoom_in)
        view_layout.addWidget(zoom_in_button, 0, 0)

        zoom_out_button = QPushButton("")
        zoom_out_button.setIcon(QIcon('zoom out.png'))  # Inserted icon
        zoom_out_button.setIconSize(QSize(30,30))
        zoom_out_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        zoom_out_button.setFixedSize(90, 50)
        zoom_out_button.clicked.connect(self.zoom_out)
        view_layout.addWidget(zoom_out_button, 1, 0)

        rotate_left_button = QPushButton("")
        rotate_left_button.setIcon(QIcon('rotate left.png'))  # Inserted icon
        rotate_left_button.setIconSize(QSize(30,30))
        rotate_left_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        rotate_left_button.setFixedSize(90, 50)
        rotate_left_button.clicked.connect(self.rotate_left_90)
        view_layout.addWidget(rotate_left_button, 0, 1)

        rotate_right_button = QPushButton("")
        rotate_right_button.setIcon(QIcon('rotate right.png'))  # Inserted icon
        rotate_right_button.setIconSize(QSize(30,30))
        rotate_right_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        rotate_right_button.setFixedSize(90, 50)
        rotate_right_button.clicked.connect(self.rotate_right_90)
        view_layout.addWidget(rotate_right_button, 1, 1)

        flip_vertical_button = QPushButton("")
        flip_vertical_button.setIcon(QIcon('vertical.png'))  # Inserted icon
        flip_vertical_button.setIconSize(QSize(30,30))
        flip_vertical_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        flip_vertical_button.setFixedSize(90, 50)
        flip_vertical_button.clicked.connect(self.flip_vertical)
        view_layout.addWidget(flip_vertical_button, 0, 2)

        flip_horizontal_button = QPushButton("")
        flip_horizontal_button.setIcon(QIcon('horizontal.png'))  # Inserted icon
        flip_horizontal_button.setIconSize(QSize(30,30))
        flip_horizontal_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        flip_horizontal_button.setFixedSize(90, 50)
        flip_horizontal_button.clicked.connect(self.flip_horizontal)
        view_layout.addWidget(flip_horizontal_button, 1, 2)           

        view_toolbar = QToolBar("View Tools")
        view_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        view_toolbar.addWidget(view_group_box)
        view_toolbar.setAllowedAreas(Qt.LeftToolBarArea)
        self.addToolBar(Qt.LeftToolBarArea, view_toolbar)

        # Measurement tools group
        measurement_group = QGroupBox("Measurement Tools")
        measurement_group.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        measurement_layout = QGridLayout(measurement_group)

        gridlines_button = QPushButton("")
        gridlines_button.setIcon(QIcon('grid.png'))
        gridlines_button.setIconSize(QSize(30,30))
        gridlines_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        gridlines_button.setFixedSize(80, 50)
        gridlines_button.setCheckable(True)  # Make the button checkable
        gridlines_button.clicked.connect(self.toggle_gridlines)
        measurement_layout.addWidget(gridlines_button, 0, 0)

        ruler_button = QPushButton("")
        ruler_button.setIcon(QIcon('ruler.png'))  # Inserted icon
        ruler_button.setIconSize(QSize(30,30))
        ruler_button.setStyleSheet("background-color: #D3C5E5; color: white; padding: 10px;")
        ruler_button.setFixedSize(80, 50)
        ruler_button.setCheckable(True)  # Make the button checkable
        ruler_button.clicked.connect(self.toggle_ruler)
        measurement_layout.addWidget(ruler_button, 0, 1)

        measurement_toolbar = QToolBar("Measurement Tools")
        measurement_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        measurement_toolbar.addWidget(measurement_group)

        # Insert the measurement toolbar before the view toolbar
        self.insertToolBar(image_toolbar, measurement_toolbar)

        # Create a new toolbar for color conversion options
        self.conversion_toolbar = QToolBar("Color Conversion Options")
        self.conversion_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        self.conversion_toolbar.setAllowedAreas(Qt.LeftToolBarArea)

        # Create a group box for color conversion options
        self.conversion_group = QGroupBox("Various Colour Conversion Options")
        self.conversion_group.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        conversion_layout = QGridLayout(self.conversion_group)

        # Create and add conversion buttons to the conversion group box
        rgb_button = QPushButton("RGB")
        rgb_button.setStyleSheet("background-color: #D3C5E5; color: black; font: bold; font-size: 14px;")
        rgb_button.clicked.connect(self.show_rgb_channels)

        gray_button = QPushButton("GRAY")
        gray_button.setStyleSheet("background-color: #D3C5E5; color: black; font: bold; font-size: 14px;")
        gray_button.clicked.connect(self.show_gray_image)

        hsv_button = QPushButton("HSV")
        hsv_button.setStyleSheet("background-color: #D3C5E5; color: black; font: bold; font-size: 14px;")
        hsv_button.clicked.connect(self.show_hsv_image)

        cie_button = QPushButton("CIE")
        cie_button.setStyleSheet("background-color: #D3C5E5; color: black; font: bold; font-size: 14px;")
        cie_button.clicked.connect(self.show_cie_image)

        hls_button = QPushButton("HLS")
        hls_button.setStyleSheet("background-color: #D3C5E5; color: black; font: bold; font-size: 14px;")
        hls_button.clicked.connect(self.show_hls_image)

        ycrcb_button = QPushButton("YCrCb")
        ycrcb_button.setStyleSheet("background-color: #D3C5E5; color: black; font: bold; font-size: 14px;")
        ycrcb_button.clicked.connect(self.show_ycrcb_image)

        # Adjust the layout to have two buttons per row
        conversion_layout.addWidget(rgb_button, 0, 0)
        conversion_layout.addWidget(gray_button, 0, 1)
        conversion_layout.addWidget(hsv_button, 0, 2)
        conversion_layout.addWidget(cie_button, 1, 0)
        conversion_layout.addWidget(hls_button, 1, 1)
        conversion_layout.addWidget(ycrcb_button, 1, 2)

        # Horizontal layout for the label and slider
        brightness_layout = QHBoxLayout()

        # Brightness slider label
        brightness_label = QLabel("Brightness")
        brightness_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        brightness_label.setAlignment(Qt.AlignCenter)

        # Brightness slider
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setRange(-100, 100)  # Range for brightness adjustment
        brightness_slider.setValue(0)  # Default value (center)
        brightness_slider.setTickInterval(20)  # Interval between ticks
        brightness_slider.setTickPosition(QSlider.TicksBelow)  # Show ticks below the slider
        brightness_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: blue;
                border: 1px solid #5c5c5c;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #87CEEB;
                border: 1px solid #777;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #e0e0e0;
                border: 1px solid #777;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::tick-mark:horizontal {
                background: black;
                width: 2px;
                height: 8px;
            }
        """)
        brightness_slider.valueChanged.connect(self.adjust_brightness)

        # Assign the slider to an instance variable
        self.brightness_slider = brightness_slider

        # Add the label and slider to the horizontal layout
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(brightness_slider)

        # Add the horizontal layout to the main layout
        conversion_layout.addLayout(brightness_layout, 4, 0, 1, 3) 

        # Add the conversion group box to the conversion toolbar
        self.conversion_toolbar.addWidget(self.conversion_group)
        self.addToolBar(Qt.LeftToolBarArea, self.conversion_toolbar) 

        # Music control in menu
        music_menu = menu_bar.addMenu("Music")
        toggle_music_action = QAction("Music [ON/OFF]", self)
        toggle_music_action.triggered.connect(self.toggle_music)
        music_menu.addAction(toggle_music_action)        

        # Scroll area for image display
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        # Container widget for images
        self.image_container = QWidget()
        self.scroll_layout = QHBoxLayout(self.image_container)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.image_container)
        
        # Label for image properties
        self.image_properties_label = QLabel(self)
        self.image_properties_label.setText("Image Properties:")
        main_layout.addWidget(self.image_properties_label)
        
        # Image Processing Group Box
        image_processing_group = QGroupBox("Image Processing Part 1")
        image_processing_group.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        image_processing_layout = QGridLayout(image_processing_group)

        # --- Bit-Plane Slicing ---
        bit_plane_button = QPushButton("Bit-Plane Slicing")
        bit_plane_button.setStyleSheet("background-color: #D3C5E5; color: black; font: bold; font-size: 14px;")
        bit_plane_button.clicked.connect(self.display_bit_planes)  # Connect to the bit-plane slicing function
        image_processing_layout.addWidget(bit_plane_button, 0, 0, 1, 2)  # Row 0, spans 2 columns

        # --- Canny Edge Detection ---
        canny_label = QLabel("Canny Edge Detection")
        canny_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_layout.addWidget(canny_label, 1, 0, 1, 2)

        low_threshold_label = QLabel("Low Threshold")
        low_threshold_label.setStyleSheet("font-size: 14px; color: white;")
        image_processing_layout.addWidget(low_threshold_label, 2, 0)

        low_threshold_slider = QSlider(Qt.Horizontal)
        low_threshold_slider.setRange(0, 255)
        low_threshold_slider.setValue(100)  # Default
        low_threshold_slider.setTickPosition(QSlider.TicksBelow)
        low_threshold_slider.setTickInterval(10)

        high_threshold_label = QLabel("High Threshold")
        high_threshold_label.setStyleSheet("font-size: 14px; color: white;")
        image_processing_layout.addWidget(high_threshold_label, 3, 0)

        high_threshold_slider = QSlider(Qt.Horizontal)
        high_threshold_slider.setRange(0, 255)
        high_threshold_slider.setValue(200)  # Default
        high_threshold_slider.setTickPosition(QSlider.TicksBelow)
        high_threshold_slider.setTickInterval(10)

        # Connect sliders to live preview
        low_threshold_slider.valueChanged.connect(lambda: self.apply_canny(
            low_threshold_slider.value(), high_threshold_slider.value()))
        high_threshold_slider.valueChanged.connect(lambda: self.apply_canny(
            low_threshold_slider.value(), high_threshold_slider.value()))

        image_processing_layout.addWidget(low_threshold_slider, 2, 1)
        image_processing_layout.addWidget(high_threshold_slider, 3, 1)

        # --- Prewitt Edge Detection ---
        prewitt_label = QLabel("Prewitt Edge Detection")
        prewitt_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_layout.addWidget(prewitt_label, 5, 0, 1, 2)

        prewitt_kernel_label = QLabel("Kernel Size")
        prewitt_kernel_label.setStyleSheet("font-size: 14px; color: white;")
        image_processing_layout.addWidget(prewitt_kernel_label, 6, 0)

        prewitt_slider = QSlider(Qt.Horizontal)
        prewitt_slider.setRange(1, 31)  # Odd numbers for kernel size
        prewitt_slider.setValue(3)  # Default kernel size
        prewitt_slider.setTickPosition(QSlider.TicksBelow)
        prewitt_slider.setTickInterval(2)

        prewitt_slider.valueChanged.connect(lambda: self.apply_prewitt(prewitt_slider.value()))

        image_processing_layout.addWidget(prewitt_slider, 6, 1)

        # --- Sobel Edge Detection ---
        sobel_label = QLabel("Sobel Edge Detection")
        sobel_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_layout.addWidget(sobel_label, 8, 0, 1, 2)

        sobel_kernel_label = QLabel("Kernel Size")
        sobel_kernel_label.setStyleSheet("font-size: 14px; color: white;")
        image_processing_layout.addWidget(sobel_kernel_label, 9, 0)

        sobel_slider = QSlider(Qt.Horizontal)
        sobel_slider.setRange(1, 31)  # Range for kernel size
        sobel_slider.setValue(3)  # Default kernel size
        sobel_slider.setTickPosition(QSlider.TicksBelow)
        sobel_slider.setTickInterval(2)

        # Ensure the kernel size is odd by adjusting the value in the handler
        sobel_slider.valueChanged.connect(
            lambda value: self.apply_sobel(value if value % 2 != 0 else value + 1)
        )

        image_processing_layout.addWidget(sobel_slider, 9, 1)

        # --- Image Contouring ---
        contour_label = QLabel("Image Contouring")
        contour_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_layout.addWidget(contour_label, 10, 0, 1, 2)

        contour_threshold_label = QLabel("Threshold")
        contour_threshold_label.setStyleSheet("font-size: 14px; color: white;")
        image_processing_layout.addWidget(contour_threshold_label, 11, 0)

        contour_threshold_slider = QSlider(Qt.Horizontal)
        contour_threshold_slider.setRange(0, 255)
        contour_threshold_slider.setValue(127)  # Default threshold
        contour_threshold_slider.setTickPosition(QSlider.TicksBelow)
        contour_threshold_slider.setTickInterval(10)

        # Connect the slider to the contouring function
        contour_threshold_slider.valueChanged.connect(
            lambda: self.perform_contouring(contour_threshold_slider.value())
        )

        image_processing_layout.addWidget(contour_threshold_slider, 11, 1)
        
        # --- Image Thresholding ---
        threshold_label = QLabel("Image Thresholding")
        threshold_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_layout.addWidget(threshold_label, 12, 0, 1, 2)

        # Horizontal layout for dropdown and slider
        threshold_hbox = QHBoxLayout()

        # Dropdown for thresholding types
        threshold_type_dropdown = QComboBox()
        threshold_type_dropdown.addItems(["Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted"])
        threshold_type_dropdown.setStyleSheet("background-color: white; font-size: 14px; color: black;")
        threshold_hbox.addWidget(threshold_type_dropdown)

        # Slider for threshold value
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(0, 255)
        threshold_slider.setValue(127)  # Default value
        threshold_slider.setTickPosition(QSlider.TicksBelow)
        threshold_slider.setTickInterval(10)
        threshold_slider.setStyleSheet("")
        threshold_hbox.addWidget(threshold_slider)

        # Add the horizontal layout to the image processing layout
        image_processing_layout.addLayout(threshold_hbox, 13, 0, 1, 2)

        # Connect slider and dropdown to the thresholding function
        threshold_slider.valueChanged.connect(
            lambda: self.apply_thresholding(threshold_slider.value(), threshold_type_dropdown.currentIndex())
        )
        threshold_type_dropdown.currentIndexChanged.connect(
            lambda: self.apply_thresholding(threshold_slider.value(), threshold_type_dropdown.currentIndex())
        )

        # --- Power Law Transformation ---
        gamma_label = QLabel("Power Law Transformation")
        gamma_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_layout.addWidget(gamma_label, 14, 0, 1, 2)

        gamma_value_label = QLabel("Gamma Value")
        gamma_value_label.setStyleSheet("font-size: 14px; color: white;")
        image_processing_layout.addWidget(gamma_value_label, 15, 0)

        gamma_slider = QSlider(Qt.Horizontal)
        gamma_slider.setRange(10, 300)  # Slider range for gamma (scaled by 100 to allow decimals)
        gamma_slider.setValue(100)  # Default value (gamma = 1.0)
        gamma_slider.setTickPosition(QSlider.TicksBelow)
        gamma_slider.setTickInterval(10)

        # Connect slider to the Power Law Transformation function
        gamma_slider.valueChanged.connect(lambda: self.apply_power_law(gamma_slider.value() / 100.0))

        image_processing_layout.addWidget(gamma_slider, 15, 1)

        # Sharpening Label
        sharpening_label = QLabel("Image Sharpening")
        sharpening_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_layout.addWidget(sharpening_label, 18, 0, 1, 2)

        # Sharpening Slider
        sharpening_slider = QSlider(Qt.Horizontal)
        sharpening_slider.setRange(1, 5)  # Sharpening levels (1 to 5)
        sharpening_slider.setValue(1)  # Default sharpening level
        sharpening_slider.setTickPosition(QSlider.TicksBelow)
        sharpening_slider.setTickInterval(1)
        sharpening_slider.valueChanged.connect(self.apply_sharpening)
        image_processing_layout.addWidget(sharpening_slider, 19, 0, 1, 2)

        # Create the Toolbar for Image Processing
        image_processing_toolbar = QToolBar("Image Processing Part 1")
        image_processing_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        image_processing_toolbar.addWidget(image_processing_group)

        # Add the Toolbar to the Right Side
        self.addToolBar(Qt.RightToolBarArea, image_processing_toolbar)

        # Create a new group box for Image Processing Part 2
        image_processing_part2_group = QGroupBox("Image Processing Part 2")
        image_processing_part2_group.setStyleSheet("font-size: 18px; padding: 10px; background-color: #735DA5; color: white;")
        image_processing_part2_layout = QGridLayout(image_processing_part2_group)

        # --- Piecewise Linear Transformation ---
        piecewise_label = QLabel("Piecewise Linear Transformation")
        piecewise_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_part2_layout.addWidget(piecewise_label, 0, 0, 1, 2)

        # Slider for Point 1
        point1_label = QLabel("Point 1")
        point1_label.setStyleSheet("font-size: 14px; color: black;")
        image_processing_part2_layout.addWidget(point1_label, 1, 0)

        point1_slider = QSlider(Qt.Horizontal)
        point1_slider.setRange(0, 255)
        point1_slider.setValue(50)  # Default value
        point1_slider.setTickPosition(QSlider.TicksBelow)
        point1_slider.setTickInterval(10)
        image_processing_part2_layout.addWidget(point1_slider, 1, 1)

        # Slider for Point 2
        point2_label = QLabel("Point 2")
        point2_label.setStyleSheet("font-size: 14px; color: black;")
        image_processing_part2_layout.addWidget(point2_label, 2, 0)

        point2_slider = QSlider(Qt.Horizontal)
        point2_slider.setRange(0, 255)
        point2_slider.setValue(200)  # Default value
        point2_slider.setTickPosition(QSlider.TicksBelow)
        point2_slider.setTickInterval(10)
        image_processing_part2_layout.addWidget(point2_slider, 2, 1)

        # Connect sliders to the transformation function
        point1_slider.valueChanged.connect(lambda: self.apply_piecewise_linear(
            point1_slider.value(), point2_slider.value()))
        point2_slider.valueChanged.connect(lambda: self.apply_piecewise_linear(
            point1_slider.value(), point2_slider.value()))

        # --- Image Erosion ---
        erosion_label = QLabel("Image Erosion")
        erosion_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_part2_layout.addWidget(erosion_label, 3, 0, 1, 2)

        erosion_kernel_label = QLabel("Kernel Size")
        erosion_kernel_label.setStyleSheet("font-size: 14px; color: black;")
        image_processing_part2_layout.addWidget(erosion_kernel_label, 4, 0)

        erosion_slider = QSlider(Qt.Horizontal)
        erosion_slider.setRange(1, 21)  # Kernel size from 1 to 21 (odd numbers preferred)
        erosion_slider.setValue(3)  # Default kernel size
        erosion_slider.setTickPosition(QSlider.TicksBelow)
        erosion_slider.setTickInterval(2)
        image_processing_part2_layout.addWidget(erosion_slider, 4, 1)

        erosion_slider.valueChanged.connect(lambda: self.apply_erosion(erosion_slider.value()))

        # --- Image Dilation ---
        dilation_label = QLabel("Image Dilation")
        dilation_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        image_processing_part2_layout.addWidget(dilation_label, 5, 0, 1, 2)

        dilation_kernel_label = QLabel("Kernel Size")
        dilation_kernel_label.setStyleSheet("font-size: 14px; color: black;")
        image_processing_part2_layout.addWidget(dilation_kernel_label, 6, 0)

        dilation_slider = QSlider(Qt.Horizontal)
        dilation_slider.setRange(1, 21)  # Kernel size from 1 to 21 (odd numbers preferred)
        dilation_slider.setValue(3)  # Default kernel size
        dilation_slider.setTickPosition(QSlider.TicksBelow)
        dilation_slider.setTickInterval(2)
        image_processing_part2_layout.addWidget(dilation_slider, 6, 1)

        dilation_slider.valueChanged.connect(lambda: self.apply_dilation(dilation_slider.value()))

        # Create the Toolbar for Image Processing Part 2
        image_processing_part2_toolbar = QToolBar("Image Processing Part 2")
        image_processing_part2_toolbar.setStyleSheet("background-color: #735DA5; color: white;")
        image_processing_part2_toolbar.addWidget(image_processing_part2_group)

        # Add the Toolbar to the Left Side, below the existing Image Processing toolbar
        self.addToolBar(Qt.LeftToolBarArea, image_processing_part2_toolbar)

    def save_canvas_state(self):
        """Saves the current canvas state to the undo stack."""
        if hasattr(self, 'canvas_label') and self.canvas_label.pixmap():
            pixmap_copy = self.canvas_label.pixmap().copy()
            self.undo_stack.append(pixmap_copy)
            # Clear the redo stack on any new action
            self.redo_stack.clear()

    def generate_thumbnail(self):
        if not hasattr(self, 'canvas_label') or self.canvas_label.pixmap() is None:
            QMessageBox.warning(self, "No Image", "Please load or transform an image first.")
            return

        # Extract the current image from the canvas
        canvas_pixmap = self.canvas_label.pixmap()
        original_width = canvas_pixmap.width()
        original_height = canvas_pixmap.height()

        # Scale to 60% of the original size
        thumbnail_width = int(original_width * 0.6)
        thumbnail_height = int(original_height * 0.6)
        thumbnail_pixmap = canvas_pixmap.scaled(thumbnail_width, thumbnail_height, Qt.KeepAspectRatio)

        # Load the logo
        logo = QImage('ArtGenPP.png')
        logo_pixmap = QPixmap.fromImage(logo)

        # Scale the logo to 1/5 of the thumbnail size
        logo_pixmap = logo_pixmap.scaled(thumbnail_width // 2, thumbnail_height // 2, Qt.KeepAspectRatio)

        # Overlay the logo on the thumbnail
        painter = QPainter(thumbnail_pixmap)
        painter.drawPixmap(10, 10, logo_pixmap)  # Position logo at the top-left corner
        painter.end()

        # Display the thumbnail in a dialog
        thumbnail_dialog = QDialog(self)
        thumbnail_dialog.setWindowTitle("Thumbnail Preview")
        thumbnail_dialog.setStyleSheet("background-color: white; color: black;")
        layout = QVBoxLayout(thumbnail_dialog)

        # Show the thumbnail
        label = QLabel()
        label.setPixmap(thumbnail_pixmap)
        layout.addWidget(label)

        # Add a Save button
        save_button = QPushButton("Save Thumbnail")
        save_button.clicked.connect(lambda: self.save_image(thumbnail_pixmap))
        layout.addWidget(save_button)

        thumbnail_dialog.exec_()

    # Method for saving the image with the logo overlay
    def save_image(self, pixmap):
        # Open a file dialog to allow the user to choose a location and filename
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.setDefaultSuffix("png")
        
        if file_dialog.exec_():
            # Get the file path from the dialog
            file_path = file_dialog.selectedFiles()[0]
            
            # Save the pixmap to the selected file path
            if pixmap.save(file_path):
                QMessageBox.information(self, "Image Saved", f"Image has been saved to {file_path}.")
            else:
                QMessageBox.warning(self, "Save Failed", "Failed to save the image.")    
    
    def undo(self):
        """Undo the last action."""
        if self.undo_stack:
            pixmap = self.undo_stack.pop()
            self.redo_stack.append(self.canvas_label.pixmap().copy())
            self.canvas_label.setPixmap(pixmap)
        else:
            QMessageBox.warning(self, "Undo", "No more actions to undo.")            

    def redo(self):
        """Redo the last undone action."""
        if self.redo_stack:
            pixmap = self.redo_stack.pop()
            self.undo_stack.append(self.canvas_label.pixmap().copy())
            self.canvas_label.setPixmap(pixmap)
        else:
            QMessageBox.warning(self, "Redo", "No more actions to redo.")            

    def clear(self):
        """Clears the current drawing on the canvas."""
        if hasattr(self, 'canvas_label') and self.canvas_label.pixmap():
            # Save the current state before clearing
            self.save_canvas_state()
            pixmap = QPixmap(self.canvas_label.pixmap().size())
            pixmap.fill(Qt.white)
            self.canvas_label.setPixmap(pixmap)   

    def keyPressEvent(self, event):
        """Handle key press events to confirm cropping with Enter key."""
        if event.key() == Qt.Key_Return and self.cropping:
            # Confirm the crop if Enter is pressed
            self.crop()

    def enable_crop_mode(self):
        """Enables cropping mode and sets up the initial conditions."""
        if self.canvas_label.pixmap() is not None:  # Check if an image is loaded
            self.cropping = True  # Enable cropping mode
            self.canvas_label.setCursor(Qt.CrossCursor)  # Change cursor to cross for visual feedback
            self.original_pixmap = self.canvas_label.pixmap().copy()  # Store original image for cropping and reset purposes
            self.installEventFilter(self)  # Enable event filtering for additional controls like the Enter key

    def crop(self):
        """Crops the selected region from the canvas and displays it."""
        if not self.canvas_label.pixmap():  # Ensure there is an image to crop
            QMessageBox.warning(self, "No Image", "No image loaded to crop.")
            return

        # Calculate the crop rectangle dimensions
        x1 = min(self.crop_start.x(), self.crop_end.x())
        y1 = min(self.crop_start.y(), self.crop_end.y())
        width = abs(self.crop_end.x() - self.crop_start.x())
        height = abs(self.crop_end.y() - self.crop_start.y())

        # Use the stored original image to crop, avoiding any layered effects from previous crops
        cropped_pixmap = self.original_pixmap.copy(QRect(x1, y1, width, height))
        self.canvas_label.setPixmap(cropped_pixmap)  # Display cropped area on the canvas
        self.canvas_label.setFixedSize(cropped_pixmap.size())  # Adjust canvas size to fit cropped image
        self.cropping = False  # Disable cropping mode
        self.canvas_label.setCursor(Qt.ArrowCursor)  # Reset cursor to default
        self.removeEventFilter(self)  # Remove event filter after cropping is complete

    def rotate_left_90(self):
        """Rotates the loaded images 90 degrees counterclockwise."""
        if self.rgb_images:  # Ensure there are images loaded
            for rgb_image in self.rgb_images:
                # Perform rotation and update each image in rgb_images
                rotated_image = cv2.rotate(rgb_image.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rgb_image.image = rotated_image  # Store the rotated version
            self.display_images()  # Refresh the display to show the rotated image

    def rotate_right_90(self):
        """Rotates the loaded images 90 degrees clockwise."""
        if self.rgb_images:
            for rgb_image in self.rgb_images:
                # Rotate each image clockwise
                rotated_image = cv2.rotate(rgb_image.image, cv2.ROTATE_90_CLOCKWISE)
                rgb_image.image = rotated_image  # Update with the rotated image
            self.display_images()  # Refresh to display rotated images

    def flip_vertical(self):
        """Flips the loaded images vertically."""
        if self.rgb_images:
            # Flip each image along the vertical axis
            for rgb_image in self.rgb_images:
                # Flip the image vertically
                flipped_image = cv2.flip(rgb_image.image, 0)
                rgb_image.image = flipped_image  # Update each image with flipped version
            self.display_images()  # Update display

    def flip_horizontal(self):
        """Flips the loaded images horizontally."""
        if self.rgb_images:
            for rgb_image in self.rgb_images:
                # Flip each image along the horizontal axis
                flipped_image = cv2.flip(rgb_image.image, 1)
                rgb_image.image = flipped_image  # Save flipped imagen
            self.display_images()  # Refresh to show flipped images

    def gridlines(self):
        if hasattr(self, 'canvas_label') and self.canvas_label.pixmap():
            pixmap = self.canvas_label.pixmap().copy()  # Make a copy of the current pixmap

            with QPainter(pixmap) as painter:
                painter.setPen(QColor(128, 128, 128))  # Set gridline color

                # Calculate grid spacing (adjust as needed)
                grid_spacing = 20  # Example spacing

                # Draw vertical gridlines
                for x in range(0, pixmap.width(), grid_spacing):
                    painter.drawLine(x, 0, x, pixmap.height())

                # Draw horizontal gridlines
                for y in range(0, pixmap.height(), grid_spacing):
                    painter.drawLine(0, y, pixmap.width(), y)

            self.canvas_label.setPixmap(pixmap)  # Update the canvas with the gridlines

    def ruler(self):
        if hasattr(self, 'canvas_label') and self.canvas_label.pixmap():
            pixmap = self.canvas_label.pixmap().copy()  # Make a copy of the current pixmap

            with QPainter(pixmap) as painter:
                painter.setPen(QColor(0, 0, 0))  # Set ruler color

                # Calculate ruler spacing (adjust as needed)
                ruler_spacing = 50  # Example spacing

                # Draw vertical ruler
                for x in range(0, pixmap.width(), ruler_spacing):
                    painter.drawLine(x, 0, x, 10)  # Short vertical lines
                    painter.drawText(x, 20, str(x))  # Display x-coordinate

                # Draw horizontal ruler
                for y in range(0, pixmap.height(), ruler_spacing):
                    painter.drawLine(0, y, 10, y)  # Short horizontal lines
                    painter.drawText(20, y, str(y))  # Display y-coordinate

            self.canvas_label.setPixmap(pixmap)  # Update the canvas with the ruler
  
    # Implement the reset method
    def reset(self):
        """Resets all images and settings to their original states."""
        self.current_zoom_factor = 1.0  # Reset zoom
        self.drawing = False  # Disable drawing if it’s currently active

        # Reset the brightness slider
        self.brightness_slider.setValue(0)  # Reset slider to its default value

        # Loop through all images and reset them to their original versions
        for i, rgb_image in enumerate(self.rgb_images):
            rgb_image.image = self.original_images[i].copy()  # Restore original image
        self.display_images()  # Refresh display to show reset images

    def update_image_properties(self):
        """Updates and displays basic properties of the currently loaded image."""
        if self.rgb_images:
            # Extract the last loaded image for property display
            rgb_image = self.rgb_images[-1]  # Get the last loaded image
            file_name = os.path.basename(rgb_image.imagelocation)
            file_size = os.path.getsize(rgb_image.imagelocation) / (1024 * 1024)  # Convert bytes to MB
            height, width, channels = rgb_image.image.shape
            resolution = f"{width} x {height}"
            image_type = 'Image Type: ' + rgb_image.imagelocation.split('.')[-1].upper()

            # Format the properties string and display it in the properties label
            details = (f"File Name: {file_name}  |  "
                       f"File Size: {file_size:.2f} MB  |  "
                       f"Resolution: {resolution}  |  "
                       f"{image_type}")
            
            self.image_properties_label.setText(details)  # Update label with image properties

    def show_image_properties(self):
        """Displays a dialog box with properties of the currently loaded image."""
        if self.rgb_images:
            rgb_image = self.rgb_images[-1]  # Get the latest loaded image
            file_name = os.path.basename(rgb_image.imagelocation)
            file_size = os.path.getsize(rgb_image.imagelocation) / (1024 * 1024)  # File size in MB
            height, width, channels = rgb_image.image.shape
            resolution = f"{width} x {height}"
            image_type = 'Image Type: ' + rgb_image.imagelocation.split('.')[-1].upper()

            # Construct property details and display in an information dialog
            details = (f"File Name: {file_name}\n"
                       f"File Size: {file_size:.2f} MB\n"
                       f"Resolution: {resolution}\n"
                       f"{image_type}")

            # Show message box with image properties
            QMessageBox.information(self, "Image Properties", details)
        else:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")

    def zoom_in(self):
        """Increases the zoom factor for the displayed image(s)."""
        self.current_zoom_factor += 0.1 # Increment zoom factor
        self.display_images()  # Update display with new zoom level

    def zoom_out(self):
        """Decreases the zoom factor for the displayed image(s)."""
        if self.current_zoom_factor > 0.1:  # Ensure minimum zoom level
            self.current_zoom_factor -= 0.1
            self.display_images()  # Update display with new zoom level

    def combine_image(self):
        """Combines all loaded images into a single vertically stacked image."""
        if len(self.rgb_images) < 2:
            QMessageBox.warning(self, "Not Enough Images", "Please load at least two images to combine.")
            return

        # Determine the maximum width among the loaded images
        max_width = max(img.image.shape[1] for img in self.rgb_images)

        # Resize images to the maximum width and ensure color format consistency
        resized_images = []
        for img in self.rgb_images:
            # Convert to BGR if necessary
            if len(img.image.shape) == 2 or img.image.shape[2] != 3:
                img_bgr = cv2.cvtColor(img.image, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img.image

            # Resize to match the maximum width
            height = int(img_bgr.shape[0] * (max_width / img_bgr.shape[1]))
            resized_img = cv2.resize(img_bgr, (max_width, height))
            resized_images.append(resized_img)

        # Concatenate vertically
        combined_image = cv2.vconcat(resized_images)

        # Resize combined image if it exceeds maximum dimensions
        max_combined_width = 800  # Maximum width for the combined image
        max_combined_height = 600  # Maximum height for the combined image

        combined_height, combined_width = combined_image.shape[:2]
        if combined_width > max_combined_width or combined_height > max_combined_height:
            scaling_factor = min(max_combined_width / combined_width, max_combined_height / combined_height)
            new_width = int(combined_width * scaling_factor)
            new_height = int(combined_height * scaling_factor)
            combined_image = cv2.resize(combined_image, (new_width, new_height))

        # Display the combined image
        self.display_combined_image(combined_image)

    def display_combined_image(self, combined_image):
        """Displays the combined image on the canvas."""
        height, width, _ = combined_image.shape
        qt_image = QImage(combined_image.data, width, height, 3 * width, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Update the main canvas with the combined image
        self.canvas_label.setPixmap(pixmap) # Display combined image on the canvas
        self.canvas_label.setFixedSize(width, height) # Adjust canvas size
        self.combined_image = combined_image  # Store combined image for saving

    def set_pen_color(self, color_hex):
        """Sets the pen color to the selected color from the palette."""
        # Convert hex color to QColor and set as the current pen color
        self.pen_color = QColor(color_hex)

    def create_default_canvas(self, width, height):
        """Creates a blank canvas of the specified size and displays it."""
        # Initialize a blank QImage with a white background
        blank_image = QImage(width, height, QImage.Format_RGB32)
        blank_image.fill(Qt.white)
        pixmap = QPixmap.fromImage(blank_image)

        # Create a QLabel to display the canvas and configure mouse events
        self.canvas_label = QLabel(self)
        self.canvas_label.setPixmap(pixmap)
        self.canvas_label.setFixedSize(width, height)
        self.canvas_label.mousePressEvent = self.start_drawing
        self.canvas_label.mouseMoveEvent = self.draw
        
        # Center the canvas in the scrollable layout
        self.scroll_layout.addWidget(self.canvas_label, alignment=Qt.AlignCenter)

    def save_canvas(self):
        """Saves the current canvas or combined image."""
        # Check if there's a combined image to save
        if hasattr(self, 'combined_image'):
            # Open save dialog for the combined image
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Combined Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
            if file_path:
                # Save the image using OpenCV
                cv2.imwrite(file_path, self.combined_image)
                QMessageBox.information(self, "Image Saved", f"Combined image successfully saved to: {file_path}")
        # If no combined image, save the current canvas if available
        elif hasattr(self, 'canvas_label') and self.canvas_label.pixmap():
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Canvas", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
            if file_path:
                # Save the pixmap directly from the QLabel
                self.canvas_label.pixmap().save(file_path)
                QMessageBox.information(self, "Canvas Saved", f"Canvas successfully saved to: {file_path}")
        else:
            # Warn if there's nothing to save
            QMessageBox.warning(self, "No Canvas", "There is no canvas to save.")

    def toggle_pen(self):
        """Enables or disables the drawing mode for the pen."""
        # Toggle the drawing mode state
        self.drawing = not self.drawing

    def create_new_canvas(self):
        """Creates a new blank canvas with user-defined dimensions."""
        # Prompt for canvas width
        width, ok1 = QInputDialog.getInt(self, "Canvas Width", "Enter width (pixels):", 800, 1, 3000)
        if ok1:
            # Prompt for canvas height if width was accepted
            height, ok2 = QInputDialog.getInt(self, "Canvas Height", "Enter height (pixels):", 600, 1, 3000)
            if ok2:
                # Clear all existing items in the layout (for fresh canvas display)
                for i in reversed(range(self.scroll_layout.count())): 
                    widget = self.scroll_layout.itemAt(i).widget()
                    if widget: 
                        widget.deleteLater()

                # Create and display the new canvas with specified dimensions
                self.create_default_canvas(width, height)

    def open_image(self):
        """Opens a file dialog to select images, then loads and displays them."""
        # Open file dialog to select images
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Image Files (*.png *.jpg *.bmp)")
        for file_name in file_names:
            if file_name:  # Ensure a file was selected
                self.current_image_path = file_name  # Store the current image path
                rgb_image = RgbImage()  # Create a new instance for the loaded image
                rgb_image.load_image(file_name, cv2.IMREAD_COLOR)  # Load the image
                self.rgb_images.append(rgb_image)  # Store it in the list
                self.original_images.append(rgb_image.image.copy()) # Store a copy for reference
        self.display_images()  # Update the display to show all images

    def display_images(self):
        """Clears the layout and displays all loaded images with applied zoom factor."""
        # Clear any existing widgets in scroll_layout
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Display each loaded RGB image with current zoom settings
        if self.rgb_images:
            for rgb_image in self.rgb_images:
                # Calculate new dimensions based on zoom factor
                height, width, channels = rgb_image.image.shape
                new_width = int(width * self.current_zoom_factor)
                new_height = int(height * self.current_zoom_factor)
                resized_image = cv2.resize(rgb_image.image, (new_width, new_height))
                
                # Convert to QImage and display in QLabel
                qt_image = QImage(resized_image.data, new_width, new_height, 3 * new_width, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image)
                image_label = QLabel(self)
                image_label.setPixmap(pixmap)
                image_label.setFixedSize(new_width, new_height)

                # Attach mouse events for drawing
                image_label.mousePressEvent = self.start_drawing
                image_label.mouseMoveEvent = self.draw

                # Add the QLabel to the layout and update the canvas_label reference
                self.scroll_layout.addWidget(image_label, alignment=Qt.AlignCenter)
                self.canvas_label = image_label  # Update the canvas_label to the loaded image

            # Update image properties after display
            self.update_image_properties()

    def enable_text_input(self):
        """Prompts user to enter text for drawing on the canvas."""
        text, ok = QInputDialog.getText(self, "Input Text", "Enter text to draw:")
        if ok and text:
            # Enable text drawing mode and store the entered text
            self.text_mode = True
            self.text_to_draw = text  # Store the input text

    def toggle_circle_mode(self):
        """Toggles the circle drawing mode on or off."""
        # Toggle the circle mode state
        self.circle_mode = not self.circle_mode
        # Disable pen drawing mode if circle mode is active
        self.drawing = False  # Turn off pen mode if circle mode is enabled

    def start_drawing(self, event):
        self.save_canvas_state()
        if self.ruler_mode:
            self.ruler_start = event.pos()
            self.ruler_band.setGeometry(QRect(self.ruler_start, QSize()))
            self.ruler_band.show()
        if self.text_mode:
            # Draw text at mouse position
            painter = QPainter(self.canvas_label.pixmap())
            painter.setPen(self.pen_color)
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(event.pos(), self.text_to_draw)
            self.canvas_label.update()
            # Reset text mode after drawing
            self.text_mode = False
            self.text_to_draw = ""
        if self.circle_mode:
            # Initialize circle starting point and enable drawing state
            self.start_point = event.pos()
            self.drawing_circle = True
            # Store a temporary copy of the canvas for circle preview
            self.temp_pixmap = self.canvas_label.pixmap().copy()
        if self.cropping:
            # Initialize crop region start and end points
            self.crop_start = event.pos()
            self.crop_end = event.pos()
        elif self.drawing:
            # Initialize freehand drawing starting point
            self.last_point = event.pos()

    def draw(self, event):
        if self.ruler_mode and event.buttons() == Qt.LeftButton:
            self.ruler_end = event.pos()
            # self.ruler_band.setGeometry(QRect(self.ruler_start, self.ruler_end).normalized())
        if self.circle_mode and event.buttons() == Qt.LeftButton and self.drawing_circle:
            # Restore the original image each time before drawing a new circle
            self.canvas_label.setPixmap(self.temp_pixmap)
            # Draw the current circle
            painter = QPainter(self.canvas_label.pixmap())
            pen = painter.pen()
            pen.setColor(self.pen_color)
            pen.setWidth(self.pen_width)  # Set pen width for the circle outline
            painter.setPen(pen)
            center = self.start_point
            # Calculate radius based on mouse position
            self.final_radius = int(((event.pos().x() - center.x()) ** 2 + (event.pos().y() - center.y()) ** 2) ** 0.5)
            painter.drawEllipse(center, self.final_radius, self.final_radius)
            self.canvas_label.update()
        elif self.cropping:
            # Update crop region end point for live rectangle preview
            self.crop_end = event.pos()
            self.update_crop_rectangle()  # Visual feedback for cropping area
        elif self.drawing and event.buttons() == Qt.LeftButton:
            # Perform freehand drawing
            pixmap = self.canvas_label.pixmap()
            if pixmap is not None:
                with QPainter(pixmap) as painter:
                    pen = painter.pen()
                    pen.setColor(self.pen_color)
                    pen.setWidth(self.pen_width)  # Set pen width for freehand drawing
                    painter.setPen(pen)
                    painter.drawLine(self.last_point, event.pos())
                    self.canvas_label.update()
                    # Update last point for continuous line drawing
                    self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if self.ruler_mode:
            self.ruler_end = event.pos()
            self.ruler_band.hide()
        if self.circle_mode and self.drawing_circle:
            # Draw the final circle using stored radius on the original pixmap
            painter = QPainter(self.temp_pixmap)
            painter.setPen(self.pen_color)
            center = self.start_point
            painter.drawEllipse(center, self.final_radius, self.final_radius)
            # Update canvas with finalized circle
            self.canvas_label.setPixmap(self.temp_pixmap)
            self.canvas_label.update()
            # Reset circle drawing state
            self.drawing_circle = False
                
    def toggle_gridlines(self):
        self.gridlines_mode = not self.gridlines_mode
        if self.gridlines_mode:
            self.gridlines()
        else:
            self.display_images()  # Clear gridlines by refreshing the image

    def toggle_ruler(self):
        self.ruler_mode = not self.ruler_mode
        if self.ruler_mode:
            self.ruler()
        else:
            self.display_images()  # Clear ruler by refreshing the image            
            
    def show_rgb_channels(self):
        """Displays RGB channels of the loaded image."""
        if self.rgb_images:
            self.current_color_mode = 'rgb'  # Update current color mode
            rgb_image = self.rgb_images[-1].image  # Get the last loaded image
            self.update_canvas_image(rgb_image)  # Display the RGB image
            self.update_histogram(rgb_image, color_mode='rgb')  # Update histogram

    def show_gray_image(self):
        """Displays the grayscale image of the loaded image."""
        if self.rgb_images:
            self.current_color_mode = 'grayscale'  # Update current color mode
            gray_image = cv2.cvtColor(self.rgb_images[-1].image, cv2.COLOR_BGR2GRAY)
            # Convert grayscale to 3 channels for consistent display
            gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            self.update_canvas_image(gray_image_3ch)
            self.update_histogram(gray_image, color_mode='grayscale')  # Update histogram

    def show_hsv_image(self):
        """Displays the HSV image of the loaded image."""
        if self.rgb_images:
            self.current_color_mode = 'hsv'  # Update current color mode
            hsv_image = cv2.cvtColor(self.rgb_images[-1].image, cv2.COLOR_BGR2HSV)
            self.update_canvas_image(hsv_image)
            self.update_histogram(hsv_image, color_mode='hsv')  # Update histogram

    def show_cie_image(self):
        """Displays the CIE Lab image of the loaded image."""
        if self.rgb_images:
            self.current_color_mode = 'cie'  # Update current color mode
            cie_image = cv2.cvtColor(self.rgb_images[-1].image, cv2.COLOR_BGR2LAB)
            self.update_canvas_image(cie_image)
            self.update_histogram(cie_image, color_mode='cie')  # Update histogram

    def show_hls_image(self):
        """Displays the HLS image of the loaded image."""
        if self.rgb_images:
            self.current_color_mode = 'hls'  # Update current color mode
            hls_image = cv2.cvtColor(self.rgb_images[-1].image, cv2.COLOR_BGR2HLS)
            self.update_canvas_image(hls_image)
            self.update_histogram(hls_image, color_mode='hls')  # Update histogram

    def show_ycrcb_image(self):
        """Displays the YCrCb image of the loaded image."""
        if self.rgb_images:
            self.current_color_mode = 'ycrcb'  # Update current color mode
            ycrcb_image = cv2.cvtColor(self.rgb_images[-1].image, cv2.COLOR_BGR2YCrCb)
            self.update_canvas_image(ycrcb_image)
            self.update_histogram(ycrcb_image, color_mode='ycrcb')  # Update histogram

    def update_canvas_image(self, image):
        """Updates the main canvas with the given image."""
        height, width = image.shape[:2]
        if len(image.shape) == 2:  # Grayscale image
            qt_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        else:  # Color image
            qt_image = QImage(image.data, width, height, 3 * width, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.canvas_label.setPixmap(pixmap)
        self.canvas_label.setFixedSize(width, height)

    def update_crop_rectangle(self):
        """Provides a visual rectangle for the cropping area during selection."""
        if self.cropping:
            # Work on a copy of the original pixmap
            temp_pixmap = self.original_pixmap.copy()

            if temp_pixmap is not None:
                with QPainter(temp_pixmap) as painter:
                    painter.setPen(Qt.red)  # Set color for crop rectangle
                    rect = QRect(self.crop_start, self.crop_end)
                    painter.drawRect(rect)

                # Display the temporary pixmap with the crop preview
                self.canvas_label.setPixmap(temp_pixmap)

    def toggle_music(self):
        """Toggles the background music on or off."""
        if self.music_playing:
            self.media_player.pause()
            self.music_playing = False
        else:
            self.media_player.play()
            self.music_playing = True

    def loop_music(self, status):
        """Restarts the music when it finishes to keep it looping."""
        if status == QMediaPlayer.EndOfMedia:
            self.media_player.play()
            
    def confirm_exit(self):
        """Show a confirmation dialog before exiting the application."""
        reply = QMessageBox.question(self, "Exit", "Are you sure you want to exit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close_application()

    def close_application(self):
        """Closes the application."""
        QApplication.instance().quit()            
            
    def show_histogram(self, color_mode='rgb'):
        """Displays the histogram of the image in a new window."""
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        self.current_color_mode = color_mode  # Track current color mode

        # Get the currently displayed image
        displayed_image = self.rgb_images[-1].image  # Use the adjusted image

        self.histogram_window = HistogramWindow()
        self.histogram_window.update_histogram(displayed_image, color_mode)
        self.histogram_window.show()

    def update_histogram(self, image, color_mode='rgb'):
        """Update the histogram for the current image in the main window."""
        self.histogram_axes.clear()  # Clear any existing histogram plot

        if color_mode == 'rgb':
            # Calculate and plot histograms for R, G, B channels
            colors = ('b', 'g', 'r')  # OpenCV uses BGR ordering
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                self.histogram_axes.plot(hist, color=color)
        elif color_mode == 'grayscale':
            # Convert to grayscale if necessary
            if len(image.shape) == 3:  # Check if it's a color image
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            self.histogram_axes.plot(hist, color='black')
        elif color_mode == 'hsv':
            # Convert to HSV and plot histograms for H, S, V channels
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channels = ['H', 'S', 'V']
            for i, channel in enumerate(channels):
                hist = cv2.calcHist([hsv_image], [i], None, [256], [0, 256])
                self.histogram_axes.plot(hist, label=channel)
            self.histogram_axes.legend()
        elif color_mode == 'cie':
            # Convert to CIE LAB and plot histograms for L, A, B channels
            cie_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            channels = ['L', 'A', 'B']
            for i, channel in enumerate(channels):
                hist = cv2.calcHist([cie_image], [i], None, [256], [0, 256])
                self.histogram_axes.plot(hist, label=channel)
            self.histogram_axes.legend()
        elif color_mode == 'hls':
            # Convert to HLS and plot histograms for H, L, S channels
            hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            channels = ['H', 'L', 'S']
            for i, channel in enumerate(channels):
                hist = cv2.calcHist([hls_image], [i], None, [256], [0, 256])
                self.histogram_axes.plot(hist, label=channel)
            self.histogram_axes.legend()
        elif color_mode == 'ycrcb':
            # Convert to YCrCb and plot histograms for Y, Cr, Cb channels
            ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = ['Y', 'Cr', 'Cb']
            for i, channel in enumerate(channels):
                hist = cv2.calcHist([ycrcb_image], [i], None, [256], [0, 256])
                self.histogram_axes.plot(hist, label=channel)
            self.histogram_axes.legend()

        # Set titles and labels for the histogram
        self.histogram_axes.set_title(f"{color_mode.upper()} Histogram")
        self.histogram_axes.set_ylabel("Frequency")
        self.histogram_axes.set_xlim([0, 256])

        # Manually add the "Pixel Intensity" label at the right side
        self.histogram_axes.text(1.06, 0, 'Pixel Intensity', transform=self.histogram_axes.transAxes,
                                rotation=360, ha='center', va='center', fontsize=10)

        # Render the updated histogram
        self.histogram_canvas.draw()

    def adjust_brightness(self, value):
        """Adjusts the brightness of the displayed image and updates the histogram."""
        if self.rgb_images:
            # Get the last loaded image
            original_image = self.rgb_images[-1].image

            # Adjust brightness using OpenCV's convertScaleAbs function
            # The alpha is 1 (no scaling), and beta is the brightness adjustment (value)
            adjusted_image = cv2.convertScaleAbs(original_image, alpha=1, beta=value)

            # Convert the adjusted image to the current color mode if necessary
            if self.current_color_mode == 'hsv':
                display_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
            elif self.current_color_mode == 'grayscale':
                display_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            elif self.current_color_mode == 'cie':
                display_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2LAB)
            elif self.current_color_mode == 'hls':
                display_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HLS)
            elif self.current_color_mode == 'ycrcb':
                display_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2YCrCb)
            else:  # Default to RGB
                display_image = adjusted_image

            # Update the canvas with the adjusted image
            self.update_canvas_image(display_image)

            # Update the histogram in real-time with the new image
            self.update_histogram(display_image, self.current_color_mode)

    def bit_plane_slicing(self, image):
        """
        Perform bit-plane slicing on a grayscale image.
        Parameters:
            image (numpy.ndarray): Grayscale image.
        Returns:
            list: A list of 8 bit-plane images.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
        
        bit_planes = []
        for i in range(8):  # 8-bit image has planes 0-7
            plane = (image >> i) & 1  # Extract the ith bit
            bit_planes.append(plane * 255)  # Scale to 0-255 for visualization

        return bit_planes

    def display_bit_planes(self):
        """Displays the bit-plane slices of the loaded image with titles in a 4-per-row layout on the main canvas."""
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        # Get the latest loaded image
        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        height, width = gray_image.shape

        # Resize parameters for manageable display
        scale_factor = 0.35
        scaled_height = int(height * scale_factor)
        scaled_width = int(width * scale_factor)
        padding = 10  # Space between images
        title_height = 30  # Space for titles above each bit-plane

        # Calculate rows needed for 4 per row
        rows = (8 + 3) // 4  # Ceiling division for 8 images, 4 per row
        combined_height = rows * (scaled_height + padding + title_height) - padding
        combined_width = 4 * scaled_width + 3 * padding  # 4 columns with padding in between
        bit_planes_combined = np.full((combined_height, combined_width), 255, dtype=np.uint8)  # White background

        # Extract and place each bit plane with spacing and titles
        for i in range(8):
            bit_plane = ((gray_image >> i) & 1) * 255
            resized_bit_plane = cv2.resize(bit_plane, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

            row = (i // 4) * (scaled_height + padding + title_height)  # Row position
            col = (i % 4) * (scaled_width + padding)  # Column position

            # Place the bit-plane on the canvas
            bit_planes_combined[row + title_height:row + title_height + scaled_height, col:col + scaled_width] = resized_bit_plane

            # Add title for each bit plane
            title_text = f"Bit Plane {i}"
            cv2.putText(
                bit_planes_combined, title_text,
                (col + 5, row + 20),  # Slight padding inside each title area
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA
            )

        # Update the main canvas with the combined bit-plane image
        qt_image = QImage(bit_planes_combined.data, bit_planes_combined.shape[1],
                        bit_planes_combined.shape[0], bit_planes_combined.strides[0],
                        QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        self.canvas_label.setPixmap(pixmap)
        self.canvas_label.setFixedSize(bit_planes_combined.shape[1], bit_planes_combined.shape[0])

    def apply_canny(self, low_threshold, high_threshold):
        """
        Apply Canny edge detection.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)

        self.update_canvas_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    def apply_prewitt(self, kernel_size):
        """
        Apply Prewitt edge detection with dynamic kernel size.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create dynamic Prewitt kernels
        kernel_x = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel_y = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        mid = kernel_size // 2

        # Fill kernels
        kernel_x[:, :mid] = -1
        kernel_x[:, mid+1:] = 1

        kernel_y[:mid, :] = -1
        kernel_y[mid+1:, :] = 1

        # Apply filters
        prewitt_x = cv2.filter2D(gray_image, -1, kernel_x)
        prewitt_y = cv2.filter2D(gray_image, -1, kernel_y)
        edges = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

        self.update_canvas_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    def apply_sobel(self, kernel_size):
        """
        Apply Sobel edge detection.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)  # X direction
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)  # Y direction
        edges = cv2.magnitude(sobel_x, sobel_y).astype(np.uint8)

        self.update_canvas_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    def perform_contouring(self, threshold_value):
        """
        Perform contouring on the current image with the specified threshold.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        # Get the current image
        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply thresholding
        _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on a copy of the original image
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

        # Display the result
        self.update_canvas_image(contour_image)

    def apply_thresholding(self, threshold_value, threshold_type_index):
        """Applies thresholding to each channel of a color image."""
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        # Get the last loaded image
        original_image = self.rgb_images[-1].image

        # Ensure the image is in color (3 channels)
        if len(original_image.shape) != 3 or original_image.shape[2] != 3:
            QMessageBox.warning(self, "Invalid Image", "Thresholding for color requires a color image.")
            return

        # Map index to OpenCV thresholding types
        threshold_types = [
            cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV,
            cv2.THRESH_TRUNC,
            cv2.THRESH_TOZERO,
            cv2.THRESH_TOZERO_INV
        ]

        # Apply thresholding to each channel
        channels = cv2.split(original_image)
        thresholded_channels = []
        for channel in channels:
            _, thresh_channel = cv2.threshold(channel, threshold_value, 255, threshold_types[threshold_type_index])
            thresholded_channels.append(thresh_channel)

        # Merge the thresholded channels back into a color image
        thresholded_image = cv2.merge(thresholded_channels)

        # Update the canvas with the thresholded image
        height, width, _ = thresholded_image.shape
        qt_image = QImage(thresholded_image.data, width, height, 3 * width, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.canvas_label.setPixmap(pixmap)
        self.canvas_label.setFixedSize(width, height)

    def apply_power_law(self, gamma_value):
        """
        Apply Power Law Transformation (Gamma Correction) to the current image.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        # Get the current image
        image = self.rgb_images[-1].image

        # Normalize the image to [0, 1], apply gamma correction, and scale back to [0, 255]
        normalized = image / 255.0
        corrected = np.power(normalized, gamma_value)
        result = (corrected * 255).astype(np.uint8)

        # Display the result
        self.update_canvas_image(result)

    def apply_piecewise_linear(self, point1, point2):
        """
        Apply Piecewise Linear Transformation to the current image.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        # Get the current image
        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define transformation function
        def piecewise_linear(val):
            if val < point1:
                return int(val * (point1 / 255.0))
            elif val < point2:
                return int(point1 + (val - point1) * ((point2 - point1) / 255.0))
            else:
                return int(255 - (255 - point2) * ((255 - val) / (255 - point2)))

        # Apply the transformation
        vectorized_func = np.vectorize(piecewise_linear)
        transformed_image = vectorized_func(gray_image)

        # Display the transformed image
        self.update_canvas_image(cv2.cvtColor(transformed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    def apply_erosion(self, kernel_size):
        """
        Apply image erosion with the specified kernel size.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        # Get the current image
        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure the kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create the kernel and apply erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded_image = cv2.erode(gray_image, kernel, iterations=1)

        # Display the result
        self.update_canvas_image(cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR))

    def apply_dilation(self, kernel_size):
        """
        Apply image dilation with the specified kernel size.
        """
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        # Get the current image
        image = self.rgb_images[-1].image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure the kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create the kernel and apply dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        # Display the result
        self.update_canvas_image(cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR))

    def apply_sharpening(self, value):
        """Applies sharpening based on the slider value."""
        if not self.rgb_images:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        # Get the original image
        original_image = self.rgb_images[-1].image.copy()

        # Define sharpening kernels
        kernels = [
            np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]]),  # Mild sharpening
            np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]]),  # Medium sharpening
            np.array([[-1, -1, -1],
                    [-1, 10, -1],
                    [-1, -1, -1]]),  # High sharpening
        ]

        # Limit the value to the kernel index range
        kernel_index = min(value - 1, len(kernels) - 1)

        # Apply the selected kernel
        sharpened_image = cv2.filter2D(original_image, -1, kernels[kernel_index])

        # Update the display
        self.update_canvas_image(sharpened_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Splash screen setup
    splash_pix = QPixmap('ArtGenPP.png')  # Add a path to an image you want to display as splash screen
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()    
    
    # Simulate some loading time
    QTimer.singleShot(2500, splash.close)  # 2.5-second delay before closing splash screen
    
    window = PaintingApp()
    window.show()
    sys.exit(app.exec_())
