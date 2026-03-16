import sys
import subprocess
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt


class MainMenu(QMainWindow):
    """Main menu window for the Underwater Image Enhancement application"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Set window properties
        self.setWindowTitle("Menu Utama")
        self.setGeometry(500, 200, 900, 500)
        self.setStyleSheet(self.get_stylesheet())
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(15)
        
        # Top logo layout
        logo_layout = QHBoxLayout()
        logo_layout.setSpacing(20)
        
        # Left logo
        left_logo_label = QLabel()
        left_logo_path = os.path.join(os.path.dirname(__file__), "logo_ums.png")
        if os.path.exists(left_logo_path):
            left_pixmap = QPixmap(left_logo_path)
            left_pixmap = left_pixmap.scaledToHeight(120, Qt.SmoothTransformation)
            left_logo_label.setPixmap(left_pixmap)
        left_logo_label.setMinimumHeight(100)
        logo_layout.addWidget(left_logo_label, alignment=Qt.AlignLeft)
        
        # Right logo
        right_logo_label = QLabel()
        right_logo_path = os.path.join(os.path.dirname(__file__), "logo_mcg.png")
        if os.path.exists(right_logo_path):
            right_pixmap = QPixmap(right_logo_path)
            right_pixmap = right_pixmap.scaledToHeight(115, Qt.SmoothTransformation)
            right_logo_label.setPixmap(right_pixmap)
        right_logo_label.setMinimumHeight(90)
        logo_layout.addStretch()
        logo_layout.addWidget(right_logo_label, alignment=Qt.AlignRight)
        
        main_layout.addLayout(logo_layout)
        
        # Title
        title_label = QLabel(
            "PENAMBAHBAIKAN KUALITI IMEJ DALAM AIR MENGGUNAKAN\n"
            "GABUNGAN PRIOR SALURAN GELAP DAN\n"
            "RANGKAIAN NEURAL KONVOLUSI"
        )
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #FFFFFF;")
        main_layout.addWidget(title_label)
        
        # Information section
        main_layout.addSpacing(10)
        
        # Developer info
        dev_label = QLabel("DIBANGUNKAN OLEH")
        dev_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        dev_label.setAlignment(Qt.AlignCenter)
        dev_label.setStyleSheet("color: #B0E0E6;")
        main_layout.addWidget(dev_label)
        
        dev_name = QLabel("MUHAMMAD AMIRUL AQMAL BIN ABDUL LATIP (BS22110137)")
        dev_name.setFont(QFont("Segoe UI", 11))
        dev_name.setAlignment(Qt.AlignCenter)
        dev_name.setStyleSheet("color: #FFFFFF;")
        main_layout.addWidget(dev_name)
        
        # Supervisor info
        main_layout.addSpacing(8)
        
        sup_label = QLabel("PENYELIA")
        sup_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        sup_label.setAlignment(Qt.AlignCenter)
        sup_label.setStyleSheet("color: #B0E0E6;")
        main_layout.addWidget(sup_label)
        
        sup_name = QLabel("PROF. DR. ABDULLAH BIN BADE")
        sup_name.setFont(QFont("Segoe UI", 11))
        sup_name.setAlignment(Qt.AlignCenter)
        sup_name.setStyleSheet("color: #FFFFFF;")
        main_layout.addWidget(sup_name)
        
        # Buttons
        main_layout.addSpacing(15)
        
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        button_layout.setAlignment(Qt.AlignCenter)
        
        # Start button
        start_btn = QPushButton("MULA")
        start_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        start_btn.setMinimumHeight(45)
        start_btn.setMinimumWidth(200)
        start_btn.setMaximumWidth(200)
        start_btn.setStyleSheet(self.get_button_style("#2196F3"))
        start_btn.clicked.connect(self.start_application)
        button_layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        
        # Exit button
        exit_btn = QPushButton("KELUAR")
        exit_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        exit_btn.setMinimumHeight(45)
        exit_btn.setMinimumWidth(200)
        exit_btn.setMaximumWidth(200)
        exit_btn.setStyleSheet(self.get_button_style("#E74C3C"))
        exit_btn.clicked.connect(self.close_application)
        button_layout.addWidget(exit_btn, alignment=Qt.AlignCenter)
        
        main_layout.addLayout(button_layout)
        

    
    def get_stylesheet(self):
        """Return the stylesheet for the main window"""
        return """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #0a1929, stop:1 #1a3a52);
            }
        """
    
    def get_button_style(self, color):
        """Return button stylesheet"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color)};
            }}
        """
    
    def lighten_color(self, color):
        """Lighten a hex color"""
        color_map = {
            "#2196F3": "#42A5F5",
            "#E74C3C": "#EF5350",
        }
        return color_map.get(color, color)
    
    def darken_color(self, color):
        """Darken a hex color"""
        color_map = {
            "#2196F3": "#1976D2",
            "#E74C3C": "#C62828",
        }
        return color_map.get(color, color)
    
    def start_application(self):
        """Start the main application (GUI.py)"""
        try:
            gui_path = os.path.join(os.path.dirname(__file__), "GUI.py")
            if os.path.exists(gui_path):
                subprocess.Popen([sys.executable, gui_path])
                self.close()
            else:
                print(f"Error: GUI.py not found at {gui_path}")
        except Exception as e:
            print(f"Error starting application: {e}")
    
    def close_application(self):
        """Close the application"""
        self.close()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = MainMenu()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
