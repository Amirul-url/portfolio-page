
import os
import joblib
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

# TabManager Class: Handles dynamic tabs in the app
class TabManager(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TabManager, self).__init__(parent)

        # Setup layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create top layout for + button
        top_layout = QtWidgets.QHBoxLayout()
        self.add_tab_btn = QtWidgets.QPushButton('+')
        self.add_tab_btn.setFixedSize(40, 40)
        
        # Style + button
        self.add_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 20px;
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.add_tab_btn.clicked.connect(self.add_new_tab)

        top_layout.addStretch()
        top_layout.addWidget(self.add_tab_btn)

        # Create tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.setMovable(True)
        
        # Style tabs
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: #fafafa;
            }
            QTabBar::tab {
                background: #e0e0e0;
                padding: 10px 20px; /* <-- More padding inside the tab */
                margin: 4px;
                min-width: 120px; /* <-- Minimum width for each tab */
                border-radius: 8px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                font-weight: bold;
                border: 2px solid #3498db;
            }
        """)

        # Assemble layouts
        layout.addLayout(top_layout)
        layout.addWidget(self.tabs)

        # Add an initial tab
        self.add_new_tab(initial=True)

    def add_new_tab(self, initial=False):
        # Add a new tab to the tab widget
        new_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Label for loaded image
        image_label = QtWidgets.QLabel('Loaded Image will appear here')
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label.setStyleSheet("border: 2px solid #999; background-color: #fafafa;")
        image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        image_label.setFixedHeight(150) 

        # Label for evaluation/output result
        output_label = QtWidgets.QLabel('Evaluation Results / Output will appear here')
        output_label.setAlignment(QtCore.Qt.AlignCenter)
        output_label.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
        output_label.setFixedHeight(200)  # still keep output smaller height
        output_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # Add widgets to tab layout
        layout.addWidget(image_label, stretch=4)
        layout.addWidget(output_label, stretch=1)
        new_tab.setLayout(layout)

        # Add the new tab into the tab widget
        index = self.tabs.addTab(new_tab, f"Untitled {self.tabs.count() + 1}")
        self.tabs.setCurrentIndex(index)

    def close_tab(self, index):
        # Close the selected tab
        self.tabs.removeTab(index)

class ImageClassificationAppSystem(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set application window title and size
        self.setWindowTitle('Image Classification System')
        self.setGeometry(400, 100, 1100, 900)
        
        # Initialize classifier manager and music player
        self.manager = ClassifierManager()
        self.music_player = QMediaPlayer()
        self.music_playing = False  # Track music status

        self.initUI()

    def initUI(self):
        # Initialize and design the full UI layout
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Set background color for main area
        central_widget.setStyleSheet("background-color: #d5f5e3;")

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Sidebar (left menu) setup
        self.sidebar = QtWidgets.QVBoxLayout()

        # Define button fixed size
        button_width = 225
        button_height = 60

        # Common button style (green buttons)
        button_style = """
            QPushButton {
                background-color: #B2FF59;
                color: black;
                font-size: 14px;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """

        # Create sidebar buttons
        self.load_dataset_btn = QtWidgets.QPushButton('Load Dataset')
        self.train_model_btn = QtWidgets.QPushButton('Train Classification Model')
        self.load_model_btn = QtWidgets.QPushButton('Load Classification Model')
        self.load_image_btn = QtWidgets.QPushButton('Load New Image')
        self.verify_image_btn = QtWidgets.QPushButton('Verify Image')
        self.evaluation_btn = QtWidgets.QPushButton('Evaluation Metrics')
        self.export_btn = QtWidgets.QPushButton('Export Report')
        self.info_btn = QtWidgets.QPushButton('Info / Help')
        
        # Set music button icon
        self.music_btn = QtWidgets.QPushButton('Play Music')
        self.music_btn.setIcon(QtGui.QIcon('icon_music.png'))  # Optional, if you have music icon
        self.music_btn.clicked.connect(self.toggle_music)
        
        # Set icons for all sidebar buttons
        self.load_dataset_btn.setIcon(QtGui.QIcon('icon_load.png'))
        self.train_model_btn.setIcon(QtGui.QIcon('icon_train.png'))
        self.load_model_btn.setIcon(QtGui.QIcon('icon_load_model.png'))
        self.load_image_btn.setIcon(QtGui.QIcon('icon_load_image.png'))
        self.verify_image_btn.setIcon(QtGui.QIcon('icon_verify.png'))
        self.evaluation_btn.setIcon(QtGui.QIcon('icon_eval.png'))
        self.export_btn.setIcon(QtGui.QIcon('icon_export.png'))
        self.info_btn.setIcon(QtGui.QIcon('icon_info.png'))

       # Set consistent icon size and style for all buttons
        for btn in [self.load_dataset_btn, self.train_model_btn, self.load_model_btn,
                    self.load_image_btn, self.verify_image_btn, self.evaluation_btn,
                    self.export_btn, self.info_btn, self.music_btn]:
            btn.setIconSize(QtCore.QSize(50, 50))
            btn.setStyleSheet("""
                QPushButton {
                    qproperty-iconSize: 32px 32px;
                }
            """)
            btn.setFixedSize(button_width, button_height)

        # Info button has special blue style
        self.info_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        # Connect buttons to respective functions
        self.load_dataset_btn.clicked.connect(lambda: self.run_with_progress(self.load_dataset))
        self.train_model_btn.clicked.connect(lambda: self.run_with_progress(self.train_model))
        self.load_model_btn.clicked.connect(lambda: self.run_with_progress(self.load_model))
        self.load_image_btn.clicked.connect(lambda: self.run_with_progress(self.load_image))
        self.verify_image_btn.clicked.connect(lambda: self.run_with_progress(self.verify_image))
        self.evaluation_btn.clicked.connect(lambda: self.run_with_progress(self.show_evaluation))
        self.export_btn.clicked.connect(lambda: self.run_with_progress(self.export_report))
        self.info_btn.clicked.connect(self.show_info)  # Info button no progress bar needed

        # Create and setup progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Initially hidden
        self.sidebar.addWidget(self.progress_bar)
        
        # Add all buttons to sidebar
        for btn in [self.load_dataset_btn, self.train_model_btn, self.load_model_btn,
                    self.load_image_btn, self.verify_image_btn, self.evaluation_btn,
                    self.export_btn, self.music_btn]:
            btn.setStyleSheet(button_style)
            self.sidebar.addWidget(btn)

        # Add info button separately
        self.sidebar.addWidget(self.info_btn)
        self.sidebar.addWidget(self.progress_bar)
        self.sidebar.addWidget(self.music_btn)
        self.sidebar.addStretch()

        # Sidebar container with dark background
        sidebar_widget = QtWidgets.QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setStyleSheet("background-color: #2c3e50;")

        # Right Panel - Tabs
        self.tab_widget = TabManager()

        # Split left (sidebar) and right (tab panel)
        main_layout.addWidget(sidebar_widget, stretch=1)
        main_layout.addWidget(self.tab_widget, stretch=5)

    # Music Control: Play / Pause background music
    def toggle_music(self):
        if not self.music_playing:
            # Start playing background music
            music_file = QUrl.fromLocalFile(os.path.abspath('chill_coding.mp3'))  # make sure file exists
            self.music_player.setMedia(QMediaContent(music_file))
            self.music_player.setVolume(30)  # Volume 0-100
            self.music_player.play()
            self.music_playing = True
            self.music_btn.setText('Pause Music')
        else:
            # Pause music
            self.music_player.pause()
            self.music_playing = False
            self.music_btn.setText('Play Music')

    # Helper: Run action with animated progress bar
    def run_with_progress(self, action_function):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Animate the progress bar smoothly
        for i in range(1, 101):
            QtCore.QCoreApplication.processEvents()
            QtCore.QThread.msleep(5)  # Fast, smooth animation
            self.progress_bar.setValue(i)

        action_function()  # Now call the real function
        self.progress_bar.setVisible(False)

    # Export the evaluation report to a TXT file
    def export_report(self):
        if not hasattr(self, 'last_report_text') or self.last_report_text is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please evaluate model first!")
            return

        # Open save file dialog
        options = QtWidgets.QFileDialog.Options()
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Report", "", "Text Files (*.txt)", options=options)

        if filepath:
            # Write the report into the selected file
            with open(filepath, 'w') as f:
                f.write(self.last_report_text)
            QtWidgets.QMessageBox.information(self, "Success", f"Report saved to:\n{filepath}")

    # Load Dataset: Iris or Rose dataset
    def load_dataset(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        dataset_choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Dataset", "Choose Dataset:", ["Iris", "Rose"], 0, False
        )

        if ok and dataset_choice:
            if dataset_choice == "Iris":
                msg = self.manager.load_iris_dataset()
                QtWidgets.QMessageBox.information(self, "Dataset Loaded", msg)
                self.show_iris_data()

                # Set tab title after loading
                current_index = self.tab_widget.tabs.currentIndex()
                if current_index != -1:
                    self.tab_widget.tabs.setTabText(current_index, "Iris Dataset")

            else:
                folder_path = QtWidgets.QFileDialog.getExistingDirectory(
                    self, "Select Rose Dataset Folder", "", options=options
                )
                if folder_path:
                    msg = self.manager.load_rose_dataset(folder_path)
                    QtWidgets.QMessageBox.information(self, "Dataset Loaded", msg)
                    self.show_rose_interface()

                    # Set tab title after loading
                    current_index = self.tab_widget.tabs.currentIndex()
                    if current_index != -1:
                        self.tab_widget.tabs.setTabText(current_index, "Rose Dataset")

    def show_rose_interface(self):
        current_index = self.tab_widget.tabs.currentIndex()
        if current_index == -1:
            return

        current_tab = self.tab_widget.tabs.widget(current_index)

        # Clear previous widgets (image + output area)
        for i in reversed(range(current_tab.layout().count())):
            widget = current_tab.layout().itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Create new frame like Iris
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
        frame.setFixedHeight(150)
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder inside frame
        image_label = QtWidgets.QLabel('Loaded Image will appear here')
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label.setStyleSheet("background-color: #f5f5f5;") 

        frame_layout.addWidget(image_label)

        # Create Evaluation output area
        output_label = QtWidgets.QLabel('Evaluation Results / Output will appear here')
        output_label.setAlignment(QtCore.Qt.AlignCenter)
        output_label.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
        output_label.setFixedHeight(200)

        # Add frame and output to current tab
        current_tab.layout().addWidget(frame)
        current_tab.layout().addWidget(output_label)

    def train_model(self):
        if self.manager.X_train is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Simulate progress
        for i in range(1, 101):
            QtCore.QCoreApplication.processEvents() 
            QtCore.QThread.msleep(10)
            self.progress_bar.setValue(i)

        # Actually train the model
        msg = self.manager.train_model()
        QtWidgets.QMessageBox.information(self, "Training", msg)

        # Hide progress bar after training
        self.progress_bar.setVisible(False)

        # Ask user to save
        reply = QtWidgets.QMessageBox.question(
            self, 'Save Model', "Do you want to save the trained model?", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Model", "", "Joblib Files (*.joblib)")
            if filepath:
                save_msg = self.manager.save_model(filepath)
                QtWidgets.QMessageBox.information(self, "Save Model", save_msg)

    def load_model(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Model", "", "Joblib Files (*.joblib)")
        if filepath:
            msg = self.manager.load_model(filepath)
            QtWidgets.QMessageBox.information(self, "Model Loaded", msg)

    def load_image(self):
        if self.manager.dataset_name == 'Iris':
            QtWidgets.QMessageBox.warning(self, "Warning", "Load New Image is only available for Rose dataset.")
            return

        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if filepath:
            msg = self.manager.load_new_image(filepath)
            QtWidgets.QMessageBox.information(self, "Image Loaded", msg)

            current_index = self.tab_widget.tabs.currentIndex()
            if current_index == -1:
                return

            current_tab = self.tab_widget.tabs.widget(current_index)

            # Clear previous widgets (image + output area)
            for i in reversed(range(current_tab.layout().count())):
                widget = current_tab.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()

            # Load and show image
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            qimg = QtGui.QImage(img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)

            # Resize pixmap properly to fit frame height (150px)
            target_height = 150  # Frame height
            aspect_ratio = width / height
            target_width = int(target_height * aspect_ratio)
            pixmap = pixmap.scaled(target_width, target_height, QtCore.Qt.KeepAspectRatio)

            # Create a frame for image (solid border, same style as dataset frame)
            frame = QtWidgets.QFrame()
            frame.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
            frame.setFixedHeight(150)
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)

            image_label = QtWidgets.QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(QtCore.Qt.AlignCenter)
            image_label.setStyleSheet("background-color: #f5f5f5;") 

            frame_layout.addWidget(image_label)

            # Output label
            output_label = QtWidgets.QLabel('Evaluation Results / Output will appear here')
            output_label.setAlignment(QtCore.Qt.AlignCenter)
            output_label.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
            output_label.setFixedHeight(200)

            # Add frame and output label back to the tab
            current_tab.layout().addWidget(frame)
            current_tab.layout().addWidget(output_label)

    def verify_image(self):
        if self.manager.dataset_name == 'Iris':
            QtWidgets.QMessageBox.warning(self, "Warning", "Verify Image is only available for Rose dataset.")
            return
        result = self.manager.predict_loaded_image()
        QtWidgets.QMessageBox.information(self, "Prediction Result", f"Predicted Color of Rose: {result}")

        # Also show inside current tab (if any)
        current_index = self.tab_widget.tabs.currentIndex()
        if current_index != -1:
            current_tab = self.tab_widget.tabs.widget(current_index)
            label = QtWidgets.QLabel(f"Prediction Color of Rose: {result}")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("font-size: 18px; color: green; margin: 10px;")
            current_tab.layout().addWidget(label)

    def show_evaluation(self):
        acc, prec, rec, cm, report = self.manager.evaluate_model()
        if acc is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please train or load a model first!")
            return

        current_index = self.tab_widget.tabs.currentIndex()
        if current_index == -1:
            return

        current_tab = self.tab_widget.tabs.widget(current_index)

        # Find the output label
        output_widget = None
        for i in range(current_tab.layout().count()):
            widget = current_tab.layout().itemAt(i).widget()
            if isinstance(widget, QtWidgets.QLabel) and "Evaluation Results" in widget.text():
                output_widget = widget
                break

        if output_widget is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No output area found in the current tab.")
            return

        # Clear old text
        output_widget.clear()

        # Create container
        eval_container = QtWidgets.QWidget()
        eval_layout = QtWidgets.QVBoxLayout(eval_container)

        # Build text report (as plain text, aligned)
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-family: Courier; font-size: 12px; background-color: #ffffff;")

        # Format the full text
        full_text = f"Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}\n\nClassification Report:\n{report}"
        text_edit.setText(full_text)
        self.last_report_text = full_text

        eval_layout.addWidget(text_edit)

        # Create Horizontal Layout for the two graphs
        graph_layout = QtWidgets.QHBoxLayout()

        # Build Confusion Matrix Plot
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        cax = ax_cm.matshow(cm, cmap='Blues')
        fig_cm.colorbar(cax)
        ax_cm.set_xlabel('Predicted', fontsize=10)
        ax_cm.set_ylabel('True', fontsize=10)

        # Dynamic title based on dataset
        dataset_title = self.manager.dataset_name if self.manager.dataset_name else "Dataset"
        ax_cm.set_title(f"{dataset_title} - Confusion Matrix", fontsize=12)

        # Set axis labels
        if self.manager.dataset_name == "Iris":
            class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        elif self.manager.dataset_name == "Rose":
            class_labels = self.manager.categories
        else:
            class_labels = [str(i) for i in range(cm.shape[0])]

        ax_cm.set_xticks(np.arange(len(class_labels)))
        ax_cm.set_yticks(np.arange(len(class_labels)))
        ax_cm.set_xticklabels(class_labels, rotation=45, ha='right')
        ax_cm.set_yticklabels(class_labels, fontsize=8)

        for (i, j), val in np.ndenumerate(cm):
            ax_cm.text(j, i, f'{val}', ha='center', va='center', fontsize=8)

        plt.tight_layout()

        canvas_cm = FigureCanvas(fig_cm)
        graph_layout.addWidget(canvas_cm)

        # Build Bar Chart (Precision / Recall / F1)
        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))

        # Parse classification report
        from sklearn.metrics import classification_report
        report_dict = classification_report(self.manager.y_test, self.manager.model.predict(self.manager.X_test), output_dict=True, zero_division=1)

        classes = []
        precision = []
        recall = []
        f1 = []

        for label in report_dict:
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(label)
                precision.append(report_dict[label]['precision'])
                recall.append(report_dict[label]['recall'])
                f1.append(report_dict[label]['f1-score'])

        x = np.arange(len(classes))
        width = 0.25

        rects1 = ax_bar.bar(x - width, precision, width, label='Precision', color='blue')
        rects2 = ax_bar.bar(x, recall, width, label='Recall', color='orange')
        rects3 = ax_bar.bar(x + width, f1, width, label='F1-Score', color='green')

        ax_bar.set_ylabel('Score')
        ax_bar.set_xlabel('Label')
        ax_bar.set_title('Per-Class Metrics')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(classes, rotation=45, ha='right')
        ax_bar.legend()

        # Annotate bars with values
        for rect in rects1 + rects2 + rects3:
            height = rect.get_height()
            ax_bar.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        canvas_bar = FigureCanvas(fig_bar)
        graph_layout.addWidget(canvas_bar)

        # Add the graph layout to the evaluation layout
        eval_layout.addLayout(graph_layout)

        # Replace output label with our container
        parent_layout = current_tab.layout()
        index = parent_layout.indexOf(output_widget)
        parent_layout.removeWidget(output_widget)
        output_widget.deleteLater()
        parent_layout.insertWidget(index, eval_container)

    def show_iris_data(self):
        if self.manager.X_train is None:
            return

        current_index = self.tab_widget.tabs.currentIndex()
        if current_index == -1:
            return

        current_tab = self.tab_widget.tabs.widget(current_index)

        # Clear previous widgets
        for i in reversed(range(current_tab.layout().count())):
            widget = current_tab.layout().itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Outer dashed frame like image area
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
        frame.setFixedHeight(150)   # Fixed Height for Dataset Table
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        # Create Table inside
        table = QtWidgets.QTableWidget()
        X = np.vstack((self.manager.X_train, self.manager.X_test))
        y = np.hstack((self.manager.y_train, self.manager.y_test))

        table.setRowCount(X.shape[0])
        table.setColumnCount(X.shape[1] + 1)
        headers = self.manager.feature_names + [self.manager.label_name]
        table.setHorizontalHeaderLabels(headers)

        table.setShowGrid(True)
        table.setAlternatingRowColors(True)
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f9f9f9;
                gridline-color: #cccccc;
                border: none;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
        """)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{X[i, j]:.2f}"))
            table.setItem(i, X.shape[1], QtWidgets.QTableWidgetItem(str(y[i])))

        table.resizeColumnsToContents()

        # Add table into frame
        frame_layout.addWidget(table)

        # Create Evaluation output area
        output_label = QtWidgets.QLabel('Evaluation Results / Output will appear here')
        output_label.setAlignment(QtCore.Qt.AlignCenter)
        output_label.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
        output_label.setFixedHeight(200)

        # Add frame and output to current tab
        current_tab.layout().addWidget(frame)
        current_tab.layout().addWidget(output_label)

    def show_info(self):
        info_text = (
            "Welcome to the Flower Image Classification App!\n\n"
            "Instructions:\n"
            "1. Load your dataset (Iris or Rose dataset).\n"
            "2. Train a classification model using the dataset.\n"
            "3. Save the trained model if needed.\n"
            "4. Load a trained model if you have one saved.\n"
            "5. Load a new flower image to verify its class (color/species).\n"
            "6. Verify the image and see the predicted class.\n"
            "7. Evaluate the model to see accuracy, precision, recall, and confusion matrix.\n"
            "8. Export the evaluation report to a TXT file for your records.\n\n"
            "Tip: Use clear, centered images for better prediction results.\n\n"
            "Enjoy!"
        )

        QtWidgets.QMessageBox.information(self, "Information / Help", info_text)

class ClassifierManager:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.loaded_image = None
        self.loaded_image_path = None
        self.dataset_name = None

    def load_iris_dataset(self, filepath=None):
        if filepath is None:
            # If no filepath provided, ask the user
            options = QtWidgets.QFileDialog.Options()
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "Select Iris CSV File", "", "CSV Files (*.csv)", options=options
            )
            if not filepath:
                return "No file selected."

        data = pd.read_csv(filepath)

        # Remove ID column if exists
        if 'Id' in data.columns:
            data = data.drop('Id', axis=1)

        X = data.drop('Species', axis=1).values
        y = data['Species'].values

        self.dataset_name = 'Iris'
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.feature_names = list(data.drop('Species', axis=1).columns)
        self.label_name = 'Species'

        return f"Iris Dataset loaded from CSV: {X.shape[0]} samples."

    def load_rose_dataset(self, folder_path):
        categories = sorted(os.listdir(folder_path))  # sort for consistency
        X = []
        y = []
        for category in categories:
            category_path = os.path.join(folder_path, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        X.append(img.flatten())
                        y.append(category)  
        X = np.array(X)
        y = np.array(y)
        self.categories = categories 
        self.dataset_name = 'Rose'
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return f"Rose Dataset loaded: {X.shape[0]} images, {len(categories)} categories."

    def train_model(self):
        self.model = SVC(kernel='linear')
        self.model.fit(self.X_train, self.y_train)
        return "Model trained successfully."

    def save_model(self, filepath):
        if self.model:
            joblib.dump(self.model, filepath)
            return "Model saved successfully."
        else:
            return "No model to save."

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        return "Model loaded successfully."

    def load_new_image(self, filepath):
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            self.loaded_image = img.flatten().reshape(1, -1)
            self.loaded_image_path = filepath
            return "New image loaded."
        else:
            return "Failed to load image."

    def predict_loaded_image(self):
        if self.model and self.loaded_image is not None:
            prediction = self.model.predict(self.loaded_image)
            return prediction[0] 
        else:
            return "No model or image loaded."

    def evaluate_model(self):
        if self.model:
            y_pred = self.model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=1)  
            rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=1)      
            cm = confusion_matrix(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, zero_division=1)               
            return acc, prec, rec, cm, report
        else:
            return None, None, None, None, None

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    # Show splash screen
    splash_pix = QtGui.QPixmap('splash.png') 
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()

    # Process app events (important for smooth splash)
    app.processEvents()

    # Simulate loading (you can remove this delay or adjust)
    QtCore.QThread.sleep(2)  # Delay 2 seconds

    # Launch main window
    window = ImageClassificationAppSystem()
    window.show()

    splash.finish(window)  # Close splash screen when main window shows

    sys.exit(app.exec_())

