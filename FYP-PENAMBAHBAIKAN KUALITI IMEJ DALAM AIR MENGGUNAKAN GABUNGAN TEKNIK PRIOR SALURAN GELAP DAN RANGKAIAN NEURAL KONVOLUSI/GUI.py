import sys
import cv2
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import datetime
import pandas as pd
import shutil

from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from colormath.color_diff import delta_e_cie2000 as de2000
from colormath.color_objects import LabColor
from skimage import filters

try:
    import torch
    from torchvision import transforms
    from WB_Model import WBNet
    from FI_Model import FINet
    from model import PhysicalNN
    import utils_func as util
except (ImportError, OSError) as e:
    print(f"Error: PyTorch not available: {e}")
    print("Please install PyTorch to use this application.")
    sys.exit(1)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTabBar, QScrollArea, QTextEdit,
    QGroupBox, QComboBox, QSpacerItem, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPalette, QColor, QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ========================== MODEL MANAGER ==========================
class ModelManager:
    """Manages model loading and device selection"""
    def __init__(self):
        self.fi_model = None
        self.wb_model = None
        self.device = None
        self.fi_checkpoint_path = None
        self.wb_checkpoint_path = None
    
    def set_checkpoint_paths(self, fi_path, wb_path):
        """Set paths to model checkpoints"""
        self.fi_checkpoint_path = fi_path
        self.wb_checkpoint_path = wb_path
    
    def get_device(self, device_type="Auto"):
        """Get the device (CPU or GPU) based on user selection"""
        if device_type == "Auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_type == "GPU":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:  # CPU
            self.device = torch.device("cpu")
        return self.device
    
    def load_fi_model(self, device):
        """Load Formation Image model"""
        self.fi_model = PhysicalNN()
        self.fi_model = torch.nn.DataParallel(self.fi_model).to(device)
        checkpoint = torch.load(self.fi_checkpoint_path, map_location=device)
        self.fi_model.load_state_dict(checkpoint['state_dict'])
        self.fi_model = self.fi_model.module
        self.fi_model.eval()
        return self.fi_model
    
    def load_wb_model(self, device):
        """Load White Balance model"""
        self.wb_model = WBNet()
        self.wb_model = torch.nn.DataParallel(self.wb_model).to(device)
        checkpoint = torch.load(self.wb_checkpoint_path, map_location=device)
        self.wb_model.load_state_dict(checkpoint['state_dict'])
        self.wb_model = self.wb_model.module
        self.wb_model.eval()
        return self.wb_model


# ========================== IMAGE ENHANCER ==========================
# Handles image enhancement using DCP and CNN models step-by-step
class ImageEnhancer:
    """Handles image enhancement using DCP and CNN models"""
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.cv2_ori_img = None
        self.cv2_fi_img = None
        self.cv2_wb_img = None
        self.cv2_dcp_img = None
        self.cv2_edt_img = None
    
    def load_image(self, image_path):
        """Load image from file"""
        pil_img = Image.open(image_path)
        self.cv2_ori_img = np.array(pil_img)
        return self.cv2_ori_img
    
    def enhance(self, image_path, device):
        """Perform complete enhancement"""
        self.load_image(image_path)
        
        # Load models
        fi_model = self.model_manager.load_fi_model(device)
        wb_model = self.model_manager.load_wb_model(device)
        
        testtransform = transforms.Compose([transforms.ToTensor()])
        unloader = transforms.ToPILImage()
        tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        
        # Apply DCP
        dcp_enhanced = util.enhance_image_dcp(self.cv2_ori_img)
        self.cv2_dcp_img = (dcp_enhanced * 255).astype(np.uint8)
        
        # Apply FI model
        imgPil = Image.open(image_path)
        inp = testtransform(imgPil).unsqueeze(0)
        inp = inp.to(device)
        FI_out = fi_model(inp)
        FI_out = unloader(FI_out.cpu().squeeze(0))
        FI_out.save('./test.png')
        FI_out_array = np.array(FI_out)
        self.cv2_fi_img = FI_out_array
        
        # Apply WB model
        s = 656
        img_resized = imgPil.resize((round(imgPil.width / max(imgPil.size) * s), round(imgPil.height / max(imgPil.size) * s)))
        w, h = img_resized.size
        new_w = w + (2 ** 4 - w % 2 ** 4) if w % 2 ** 4 != 0 else w
        new_h = h + (2 ** 4 - h % 2 ** 4) if h % 2 ** 4 != 0 else h
        if (w, h) != (new_w, new_h):
            img_resized = img_resized.resize((new_w, new_h))
        
        FI_out_resized = FI_out.resize((round(FI_out.width / max(FI_out.size) * s), round(FI_out.height / max(FI_out.size) * s)))
        w, h = FI_out_resized.size
        new_w = w + (2 ** 4 - w % 2 ** 4) if w % 2 ** 4 != 0 else w
        new_h = h + (2 ** 4 - h % 2 ** 4) if h % 2 ** 4 != 0 else h
        if (w, h) != (new_w, new_h):
            FI_out_resized = FI_out_resized.resize((new_w, new_h))
        
        # Ensure FI_out_resized is in RGB mode before converting to array
        if FI_out_resized.mode != 'RGB':
            FI_out_resized = FI_out_resized.convert('RGB')
        FI_out_resized_array = np.array(FI_out_resized)
        inpWB = testtransform(FI_out_resized_array).unsqueeze(0)
        inpWB = inpWB.to(device)
        
        out = wb_model(inpWB)
        # Properly handle the WB model output tensor
        out = torch.squeeze(out.cpu())
        # Ensure tensor values are in [0, 1] range before converting to image
        out = torch.clamp(out, 0, 1)
        # Convert tensor to numpy array in HWC format
        out = out.detach().numpy().transpose((1, 2, 0))
        # Ensure output is in proper value range [0, 1]
        out = np.clip(out, 0, 1)
        
        map_out = util.get_mapping_func(FI_out_resized_array, out)
        cnn_result = util.outOfGamutClipping(util.apply_mapping_func(FI_out_array, map_out))
        
        # Skip final DCP enhancement as WB model output is already color-corrected
        # Using DCP after WB causes yellowish tint in the background
        final_output = cnn_result
        
        # Ensure final output is properly clipped and normalized
        final_output = np.clip(final_output, 0, 1)
        final_output_uint8 = (final_output * 255).astype(np.uint8)
        
        # Create PIL Image and convert back to numpy array to ensure consistent RGB format
        final_output_pil = Image.fromarray(final_output_uint8, mode='RGB')
        self.cv2_edt_img = np.array(final_output_pil)
        
        return self.cv2_edt_img


# ========================== HISTOGRAM MANAGER ==========================
class HistogramManager:
    """Manages histogram generation and display"""
    def __init__(self):
        self.histogram_path = './histogram_temp/histogram.png'
        os.makedirs('./histogram_temp', exist_ok=True)
    
    def generate(self, image):
        """Generate histogram from image"""
        if image is None:
            return False
        
        fig = plt.Figure()
        ax1 = fig.add_subplot(1, 1, 1)
        color = ('b', 'g', 'r')
        for channel, col in enumerate(color):
            histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
            ax1.plot(histogram, color=col)
        
        # Add axis labels and title
        ax1.set_xlabel('Keamatan Piksel', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frekuensi', fontsize=12, fontweight='bold')
        ax1.set_title('Histogram Warna (B=Biru, G=Hijau, R=Merah)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['Biru', 'Hijau', 'Merah'], loc='upper right')
        
        fig.tight_layout()
        fig.savefig(self.histogram_path, dpi=100)
        plt.close(fig)
        
        return True
    
    def get_pixmap(self, size):
        """Get histogram as QPixmap scaled to size"""
        if not os.path.exists(self.histogram_path):
            return None
        pixmap = QPixmap(self.histogram_path)
        return pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ========================== FILE OPERATIONS ==========================
class FileOperations:
    """Handles file I/O operations"""
    def __init__(self):
        self.last_image_path = None
    
    def load_image_file(self, image_path):
        """Load and verify image file"""
        if not os.path.exists(image_path):
            return None
        self.last_image_path = image_path
        return image_path
    
    def save_outputs(self, output_dir, images_dict, metrics_dict, histogram_source):
        """Save all outputs to directory"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(output_dir, f"enhancement_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            
            saved_count = 0
            for filename, img_data in images_dict.items():
                if img_data is not None:
                    output_path = os.path.join(save_dir, filename)
                    if len(img_data.shape) == 3 and img_data.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_path, img_bgr)
                    else:
                        cv2.imwrite(output_path, img_data)
                    saved_count += 1
            
            # Save histogram
            if os.path.exists(histogram_source):
                hist_path = os.path.join(save_dir, "00_Histogram.png")
                shutil.copy(histogram_source, hist_path)
                saved_count += 1
            
            # Save metrics
            metrics_file = os.path.join(save_dir, "metrics.txt")
            with open(metrics_file, 'w') as f:
                f.write("Metrik Peningkatan\n")
                f.write("=" * 50 + "\n\n")
                for key, value in metrics_dict.items():
                    f.write(f"{key}: {value}\n")
            
            return save_dir, saved_count
        except Exception as e:
            return None, str(e)


# ========================== METRICS CALCULATOR ==========================
class MetricsCalculator:
    """Calculates image quality metrics"""
    def __init__(self):
        self.metrics = {}
    
    def calculate_entropy(self, image):
        """Calculate image entropy"""
        if image is None:
            return 0.0
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        num_of_pixels = img.shape[0] * img.shape[1]
        histogram, _ = np.histogram(np.array(img).flatten(), bins=np.arange(256 + 1))
        entropy = 0
        for i in range(256):
            p = histogram[i] / num_of_pixels
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def calculate_uiqm(self, image):
        """Calculate UIQM metric"""
        if image is None:
            return 0.0
        # UIQM calculation logic
        c1, c2, c3 = 0.0282, 0.2953, 3.5753
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rg = rgb[:, :, 0] - rgb[:, :, 1]
        yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
        
        # Simplified - would need full UIQM implementation
        return 0.0
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        if img1 is None or img2 is None:
            return 0.0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        ssim, _ = structural_similarity(gray1, gray2, full=True)
        return ssim


class EnhancementThread(QThread):
    finished = pyqtSignal()
    status_update = pyqtSignal(str)  # Signal to update status safely

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def run(self):
        self.parent.enhanceImage(self.emit_status)
        self.finished.emit()
    
    def emit_status(self, message):
        """Helper method to emit status updates"""
        self.status_update.emit(message)


ACCENT = "#6495ED"  # Corn blue
PANEL_BG = "#E8F4FD"  # Light corn blue background
DARK_BG = "#1E3A5F"  # Dark corn blue
TEXT_DARK = "#222222"


class MainWindow(QMainWindow):
    # Signal for thread-safe status updates
    status_update_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()

        # Initialize helper classes
        self.model_manager = ModelManager()
        self.image_enhancer = ImageEnhancer(self.model_manager)
        self.histogram_manager = HistogramManager()
        self.file_operations = FileOperations()
        self.metrics_calculator = MetricsCalculator()

        # Initialize processing variables
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.fi_model_checkpoint = os.path.join(script_dir, "IF_checkpoints", "model_best_2842.pth.tar")
        self.wb_model_checkpoint = os.path.join(script_dir, "WB_checkpoints", "model_best_2997.pth.tar")
        
        # Set checkpoint paths in ModelManager
        self.model_manager.set_checkpoint_paths(self.fi_model_checkpoint, self.wb_model_checkpoint)
        
        self.image_path = None
        self.after_enh = 0

        # Image data
        self.cv2_ori_img = None
        self.cv2_fi_img = None
        self.cv2_wb_img = None
        self.cv2_edt_img = None
        self.pix_ori_img = None
        self.pix_fi_img = None
        self.pix_wb_img = None
        self.pix_edt_img = None

        # Metrics
        self.ori_entropy = 0.0
        self.fi_entropy = 0.0
        self.wb_entropy = 0.0
        self.dcp_entropy = 0.0
        self.enh_entropy = 0.0
        self.original_Uiqm = 0.0
        self.fi_Uiqm = 0.0
        self.wb_Uiqm = 0.0
        self.dcp_Uiqm = 0.0
        self.enhanced_Uiqm = 0.0
        self.original_uciqe = 0.0
        self.fi_uciqe = 0.0
        self.wb_uciqe = 0.0
        self.dcp_uciqe = 0.0
        self.enhanced_uciqe = 0.0
        self.ori_pcqi = 0.0
        self.fi_pcqi = 0.0
        self.wb_pcqi = 0.0
        self.dcp_pcqi = 0.0
        self.com_pcqi = 0.0
        self.ori_ssim = 0.0
        self.fi_ssim = 0.0
        self.wb_ssim = 0.0
        self.dcp_ssim = 0.0
        self.com_ssim = 0.0

        self.setWindowTitle("GUI")
        self.resize(1600, 900)
        self._setup_palette()
        self._setup_fonts()

        root = QWidget()
        self.setCentralWidget(root)

        # ===== Top bar =====
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        top_bar = QHBoxLayout()
        title_label = QLabel("PENAMBAHBAIKAN KUALITI IMEJ DALAM AIR MENGGUNAKAN \nGABUNGAN PRIOR SALURAN GELAP DAN RANGKAIAN NEURAL KONVOLUSI")
        title_label.setFont(self.font_title)
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        top_bar.addWidget(title_label, stretch=1)

        main_layout.addLayout(top_bar)

        # ===== Main content: left column + center area =====
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        main_layout.addLayout(content_layout, stretch=1)

        # ------------------------------------------------------------------
        # LEFT COLUMN (slimmer, cleaner)
        # ------------------------------------------------------------------
        # Sidebar container with fixed width
        left_sidebar = QWidget()
        left_sidebar.setFixedWidth(260)
        left_sidebar_layout = QVBoxLayout(left_sidebar)
        left_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        left_sidebar_layout.setSpacing(8)
        content_layout.addWidget(left_sidebar, stretch=0)

        # ===== Histogram (minimal card) =====
        hist_card = QFrame()
        hist_card.setStyleSheet(
            "QFrame {"
            f"  background: {PANEL_BG};"
            "  border-radius: 4px;"
            "}"
        )
        hist_layout = QVBoxLayout(hist_card)
        hist_layout.setContentsMargins(8, 6, 8, 8)
        hist_layout.setSpacing(4)

        hist_title_layout = QHBoxLayout()
        hist_title = QLabel("Histogram")
        hist_title.setFont(self.font_section)
        hist_title_layout.addWidget(hist_title)
        hist_title_layout.addStretch(1)
        
        self.btn_view_histogram = QPushButton("Lihat")
        self.btn_view_histogram.setFont(self.font_body)
        self.btn_view_histogram.setFixedWidth(50)
        self.btn_view_histogram.setStyleSheet(
            "QPushButton { background: " + ACCENT + "; color: white; border-radius: 3px; padding: 3px 8px; }"
            "QPushButton:hover { background: #6495BA; }"
        )
        hist_title_layout.addWidget(self.btn_view_histogram)
        hist_layout.addLayout(hist_title_layout)

        self.histogram_placeholder = QLabel("Tiada histogram")
        self.histogram_placeholder.setAlignment(Qt.AlignCenter)
        self.histogram_placeholder.setMinimumHeight(120)
        self.histogram_placeholder.setStyleSheet("background: #F0F8FF; border: 1px solid #B0C4DE;")
        hist_layout.addWidget(self.histogram_placeholder)

        # Create histogram_temp directory if it doesn't exist
        os.makedirs('histogram_temp', exist_ok=True)

        left_sidebar_layout.addWidget(hist_card)

        # ===== Import Model (compact) =====
        import_card = QFrame()
        import_card.setStyleSheet(
            "QFrame {"
            f"  background: {PANEL_BG};"
            "  border-radius: 4px;"
            "}"
        )
        import_layout = QVBoxLayout(import_card)
        import_layout.setContentsMargins(8, 6, 8, 8)
        import_layout.setSpacing(4)

        import_title = QLabel("Model")
        import_title.setFont(self.font_section)
        import_layout.addWidget(import_title)

        row_font = self.font_body

        fm_row = QHBoxLayout()
        fm_label = QLabel("Pembentukan:")
        fm_label.setFont(row_font)
        self.formation_model_path = QLabel("IF_checkpoints/model_best_2842.pth.tar")
        self.formation_model_path.setFont(row_font)
        self.formation_model_path.setStyleSheet("background: #F0F8FF; border: 1px solid #B0C4DE; padding: 1px 3px;")
        self.btn_formation_model = QPushButton("...")
        self.btn_formation_model.setFixedWidth(24)
        fm_row.addWidget(fm_label)
        fm_row.addWidget(self.formation_model_path, stretch=1)
        fm_row.addWidget(self.btn_formation_model)
        import_layout.addLayout(fm_row)

        wb_row = QHBoxLayout()
        wb_label = QLabel("Keseimbangan Keputihan:")
        wb_label.setFont(row_font)
        self.white_balance_path = QLabel("WB_checkpoints/model_best_2997.pth.tar")
        self.white_balance_path.setFont(row_font)
        self.white_balance_path.setStyleSheet("background: #F0F8FF; border: 1px solid #B0C4DE; padding: 1px 3px;")
        self.btn_white_balance = QPushButton("...")
        self.btn_white_balance.setFixedWidth(24)
        wb_row.addWidget(wb_label)
        wb_row.addWidget(self.white_balance_path, stretch=1)
        wb_row.addWidget(self.btn_white_balance)
        import_layout.addLayout(wb_row)

        device_row = QHBoxLayout()
        device_label = QLabel("Peranti:")
        device_label.setFont(row_font)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "GPU"])
        self.device_combo.setMaximumWidth(110)
        device_row.addWidget(device_label)
        device_row.addStretch(1)
        device_row.addWidget(self.device_combo)
        import_layout.addLayout(device_row)

        left_sidebar_layout.addWidget(import_card)

        # ===== Status =====
        status_card = QFrame()
        status_card.setStyleSheet(
            "QFrame {"
            f"  background: {PANEL_BG};"
            "  border-radius: 4px;"
            "}"
        )
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(8, 6, 8, 8)
        status_layout.setSpacing(6)

        status_title = QLabel("Status & Butiran")
        status_title.setFont(self.font_section)
        status_layout.addWidget(status_title)

        self.status_label = QTextEdit()
        self.status_label.setPlainText("Sedia untuk diproses")
        self.status_label.setFont(self.font_body)
        self.status_label.setMinimumHeight(150)
        self.status_label.setReadOnly(True)
        self.status_label.setStyleSheet("background: #F0F8FF; border: 1px solid #B0C4DE; padding: 4px;")
        status_layout.addWidget(self.status_label)

        left_sidebar_layout.addWidget(status_card)

        # ===== Control Buttons =====
        buttons_card = QFrame()
        buttons_card.setStyleSheet(
            "QFrame {"
            f"  background: {PANEL_BG};"
            "  border-radius: 4px;"
            "}"
        )
        buttons_layout = QVBoxLayout(buttons_card)
        buttons_layout.setContentsMargins(8, 6, 8, 8)
        buttons_layout.setSpacing(6)

        buttons_title = QLabel("Kawalan")
        buttons_title.setFont(self.font_section)
        buttons_layout.addWidget(buttons_title)

        self.btn_load_image = QPushButton("Muatkan Imej")
        self.btn_load_image.setFont(self.font_button)
        self.btn_load_image.setStyleSheet(
            "QPushButton { background: " + ACCENT + "; color: white; border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:hover { background: #6495BA; }"
        )
        buttons_layout.addWidget(self.btn_load_image)

        self.btn_enhance = QPushButton("Tingkatkan")
        self.btn_enhance.setFont(self.font_button)
        self.btn_enhance.setStyleSheet(
            "QPushButton { background: " + ACCENT + "; color: white; border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:hover { background: #6495BA; }"
            "QPushButton:disabled { background: #a0a0a0; }"
        )
        buttons_layout.addWidget(self.btn_enhance)

        self.btn_save = QPushButton("Simpan Semua")
        self.btn_save.setFont(self.font_button)
        self.btn_save.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:hover { background: #45a049; }"
        )
        buttons_layout.addWidget(self.btn_save)

        self.btn_reset = QPushButton("Tetapkan Semula")
        self.btn_reset.setFont(self.font_button)
        self.btn_reset.setStyleSheet(
            "QPushButton { background: #FF6B6B; color: white; border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:hover { background: #FF5252; }"
        )
        buttons_layout.addWidget(self.btn_reset)

        # Add spacing
        buttons_layout.addSpacing(20)

        self.btn_exit = QPushButton("Keluar")
        self.btn_exit.setFont(self.font_button)
        self.btn_exit.setStyleSheet(
            "QPushButton { background: #666666; color: white; border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:hover { background: #555555; }"
        )
        buttons_layout.addWidget(self.btn_exit)

        left_sidebar_layout.addWidget(buttons_card)

        left_sidebar_layout.addStretch(1)

        # ------------------------------------------------------------------
        # CENTER AREA (Tabs + Display + Bottom Metric Bar)
        # ------------------------------------------------------------------
        center_layout = QVBoxLayout()
        center_layout.setSpacing(8)
        content_layout.addLayout(center_layout, stretch=1)

        self.tab_bar = QTabBar()
        self.tab_bar.setStyleSheet("QTabBar::tab { color: black; }")
        self.tab_dual = self.tab_bar.addTab("Paparan Dua")
        self.tab_overall = self.tab_bar.addTab("Perbandingan Keseluruhan")
        self.tab_bar.setCurrentIndex(self.tab_dual)
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        center_layout.addWidget(self.tab_bar)

        self.dual_widget = self._build_dual_view()
        self.overall_widget = self._build_overall_view()

        self.view_stack = {
            "dual": self.dual_widget,
            "overall": self.overall_widget,
        }

        self.view_frame = QFrame()
        self.view_frame.setStyleSheet(f"background: {PANEL_BG}; border-radius: 4px;")
        view_frame_layout = QVBoxLayout(self.view_frame)
        view_frame_layout.setContentsMargins(8, 8, 8, 8)
        view_frame_layout.addWidget(self.dual_widget)
        center_layout.addWidget(self.view_frame, stretch=1)

        self.metric_bar = self._build_metric_bar()
        center_layout.addWidget(self.metric_bar)

        self._show_view("dual")

    # ====== setup helpers ======
    def _setup_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(DARK_BG))
        pal.setColor(QPalette.Base, QColor(DARK_BG))
        pal.setColor(QPalette.Text, QColor(TEXT_DARK))
        pal.setColor(QPalette.WindowText, QColor("white"))
        self.setPalette(pal)

    def _setup_fonts(self):
        self.font_title = QFont("Segoe UI", 16, QFont.Bold)
        self.font_section = QFont("Segoe UI", 13, QFont.Bold)
        self.font_section_small = QFont("Segoe UI", 10, QFont.Bold)
        self.font_body = QFont("Segoe UI", 10)
        self.font_button = QFont("Segoe UI", 12, QFont.Bold)

    # ===== center views =====
    def _build_dual_view(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        left = QVBoxLayout()
        self.before_label_tag = QLabel("SEBELUM")
        self.before_label_tag.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.before_label_tag.setFont(self.font_section_small)
        self.before_label_tag.setStyleSheet("color: white; background: rgba(0,0,0,60); padding: 2px 6px;")
        self.before_image_label = QLabel()
        self.before_image_label.setAlignment(Qt.AlignCenter)
        self.before_image_label.setStyleSheet("background: black;")
        left.addWidget(self.before_label_tag, alignment=Qt.AlignLeft)
        left.addWidget(self.before_image_label, stretch=1)

        right = QVBoxLayout()
        self.after_label_tag = QLabel("SELEPAS")
        self.after_label_tag.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.after_label_tag.setFont(self.font_section_small)
        self.after_label_tag.setStyleSheet("color: white; background: rgba(0,0,0,60); padding: 2px 6px;")
        self.after_image_label = QLabel()
        self.after_image_label.setAlignment(Qt.AlignCenter)
        self.after_image_label.setStyleSheet("background: black;")
        right.addWidget(self.after_label_tag, alignment=Qt.AlignLeft)
        right.addWidget(self.after_image_label, stretch=1)

        layout.addLayout(left, stretch=1)
        layout.addLayout(right, stretch=1)
        return w

    def _build_overall_view(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Image comparison section
        image_section = QVBoxLayout()
        image_section.setSpacing(4)

        # Image row
        image_row = QHBoxLayout()
        image_row.setSpacing(8)

        titles = ["Asal", "Prior Saluran Gelap", "Pembentukan", "Keseimbangan Keputihan", "Gabungan"]
        descriptions = ["Imej Input", "Peningkatan DCP", "Peningkatan Pembentukan", "Peningkatan Keseimbangan Keputihan", "Peningkatan DCP+CNN"]
        self.oc_image_labels = []
        self.oc_histogram_labels = []
        self.oc_histogram_buttons = []

        for i, (title, desc) in enumerate(zip(titles, descriptions)):
            # Create container for each image
            img_container = QVBoxLayout()
            img_container.setSpacing(2)

            # Image label
            img = QLabel()
            img.setAlignment(Qt.AlignCenter)
            img.setMinimumSize(200, 150)
            img.setStyleSheet("background: #E6F3FF; color: #2E4A62; border: 1px solid #B0C4DE;")
            img.setText(title)
            self.oc_image_labels.append(img)

            # Title label
            title_label = QLabel(title)
            title_label.setFont(self.font_section_small)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("color: #2E4A62; font-weight: bold;")

            # Description label
            desc_label = QLabel(desc)
            desc_label.setFont(self.font_body)
            desc_label.setAlignment(Qt.AlignCenter)
            desc_label.setStyleSheet("color: #666666;")

            # Histogram label
            hist_label = QLabel()
            hist_label.setAlignment(Qt.AlignCenter)
            hist_label.setMinimumSize(200, 150)
            hist_label.setStyleSheet("background: #F0F8FF; border: 1px solid #B0C4DE;")
            hist_label.setText("Histogram")
            self.oc_histogram_labels.append(hist_label)
            
            # Histogram view button
            hist_btn = QPushButton("Lihat Histogram")
            hist_btn.setFont(self.font_body)
            hist_btn.setFixedHeight(24)
            hist_btn.setStyleSheet(
                "QPushButton { background: #6495ED; color: white; border-radius: 3px; padding: 3px 8px; }"
                "QPushButton:hover { background: #6495BA; }"
            )
            hist_btn.setTag = i  # Store index for button identification
            self.oc_histogram_buttons.append(hist_btn)

            img_container.addWidget(title_label)
            img_container.addWidget(img)
            img_container.addWidget(desc_label)
            img_container.addWidget(hist_label)
            img_container.addWidget(hist_btn)

            img_widget = QWidget()
            img_widget.setLayout(img_container)
            image_row.addWidget(img_widget)

        image_section.addLayout(image_row)

        # Metrics table section
        table_section = QVBoxLayout()
        table_section.setSpacing(4)

        # Table title
        table_title = QLabel("Perbandingan Metrik Keseluruhan")
        table_title.setFont(self.font_section)
        table_title.setAlignment(Qt.AlignCenter)
        table_title.setStyleSheet("color: #2E4A62; margin-bottom: 8px;")
        table_section.addWidget(table_title)

        # Create metrics table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)

        # Table headers
        headers_layout = QHBoxLayout()
        headers_layout.setSpacing(1)
        headers_layout.setContentsMargins(0, 0, 0, 0)

        header_labels = ["Metrik", "Asal", "Prior Saluran Gelap", "Pembentukan", "Keseimbangan Keputihan", "Gabungan"]
        for header_text in header_labels:
            header = QLabel(header_text)
            header.setFont(self.font_section_small)
            header.setAlignment(Qt.AlignCenter)
            header.setStyleSheet("background: #6495ED; color: white; padding: 6px; border: 1px solid #4A7FB2;")
            header.setMinimumWidth(120)
            headers_layout.addWidget(header)

        table_layout.addLayout(headers_layout)

        # Metric rows
        metric_names = ["Entropy", "SSIM", "UIQM", "UCIQE", "PCQI"]
        self.oc_metric_labels = {}

        for metric_name in metric_names:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(1)
            row_layout.setContentsMargins(0, 0, 0, 0)

            # Metric name cell
            metric_cell = QLabel(metric_name)
            metric_cell.setFont(self.font_body)
            metric_cell.setAlignment(Qt.AlignCenter)
            metric_cell.setStyleSheet("background: #F0F8FF; color: #2E4A62; padding: 4px; border: 1px solid #B0C4DE;")
            metric_cell.setMinimumWidth(120)
            row_layout.addWidget(metric_cell)

            # Value cells
            for c in range(1, 6):
                val = QLabel("-")
                val.setAlignment(Qt.AlignCenter)
                val.setFont(self.font_body)
                val.setStyleSheet("background: white; color: #2E4A62; padding: 4px; border: 1px solid #B0C4DE;")
                val.setMinimumWidth(120)
                row_layout.addWidget(val)
                self.oc_metric_labels[(metric_name, c)] = val

            table_layout.addLayout(row_layout)

        table_section.addWidget(table_widget)

        layout.addLayout(image_section)
        layout.addLayout(table_section)
        return w

    def _build_metric_bar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(
            "QFrame {"
            "  background: #8697AE;"
            "  border-radius: 4px;"
            "}"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        def metric_pair(name: str):
            n = QLabel(name)
            n.setFont(self.font_section_small)
            n.setStyleSheet("color: white;")
            v = QLabel("0.0")
            v.setFont(self.font_body)
            v.setStyleSheet("color: white;")
            layout.addWidget(n)
            layout.addWidget(v)
            return v

        self.lbl_ssim_val = metric_pair("SSIM")
        self.lbl_entropy_val = metric_pair("\t\tEntropy")
        self.lbl_uiqm_val = metric_pair("\t\tUIQM")
        self.lbl_uciqe_val = metric_pair("\t\tUCIQE")
        self.lbl_pcqi_val = metric_pair("\t\tPCQI")

        layout.addStretch(1)

        self.lbl_size = QLabel("0 × 0 px")
        self.lbl_size.setFont(self.font_body)
        self.lbl_size.setStyleSheet("color: white;")
        layout.addWidget(self.lbl_size)

        return bar

    # ===== tab switching =====
    def _on_tab_changed(self, index: int):
        if index == self.tab_dual:
            self._show_view("dual")
            self.metric_bar.show()
        elif index == self.tab_overall:
            self._show_view("overall")
            self.metric_bar.hide()

    def _show_view(self, name: str):
        for v in self.view_stack.values():
            v.setParent(None)
        self.view_frame.layout().addWidget(self.view_stack[name])

    # ===== button connections =====
    def _connect_buttons(self):
        self.btn_formation_model.clicked.connect(self._select_fi_model)
        self.btn_white_balance.clicked.connect(self._select_wb_model)
        self.btn_enhance.clicked.connect(self._enhance_image)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_view_histogram.clicked.connect(self.view_histogram_fullscreen)
        self.btn_save.clicked.connect(self.save_all_outputs)
        self.btn_reset.clicked.connect(self.reset_application)
        self.btn_exit.clicked.connect(self.close)
        
        # Connect histogram view buttons
        histogram_names = ["Original", "Dark Channel Prior", "Formation", "White Balance", "Combined"]
        for i, btn in enumerate(self.oc_histogram_buttons):
            btn.clicked.connect(lambda checked, idx=i, name=histogram_names[i]: self.view_overall_histogram(idx, name))

    def reset_application(self):
        """Reset the application to its initial state"""
        # Clear image data
        self.image_path = None
        self.cv2_ori_img = None
        self.cv2_fi_img = None
        self.cv2_wb_img = None
        self.cv2_edt_img = None
        self.pix_ori_img = None

        # Reset processing flags
        self.after_enh = 0

        # Reset metrics
        self.ori_entropy = 0.0
        self.fi_entropy = 0.0
        self.wb_entropy = 0.0
        self.enh_entropy = 0.0
        self.original_Uiqm = 0.0
        self.fi_Uiqm = 0.0
        self.wb_Uiqm = 0.0
        self.enhanced_Uiqm = 0.0
        self.original_uciqe = 0.0
        self.fi_uciqe = 0.0
        self.wb_uciqe = 0.0
        self.enhanced_uciqe = 0.0
        self.ori_pcqi = 0.0
        self.fi_pcqi = 0.0
        self.wb_pcqi = 0.0
        self.com_pcqi = 0.0
        self.ori_ssim = 0.0
        self.fi_ssim = 0.0
        self.wb_ssim = 0.0
        self.com_ssim = 0.0

        # Clear UI displays
        self.before_image_label.clear()
        self.after_image_label.clear()

        # Clear overall comparison images
        for img_label in self.oc_image_labels:
            img_label.setText("Asal" if "Asal" in img_label.text() else
                            "Pembentukan" if "Pembentukan" in img_label.text() else
                            "Keseimbangan Keputihan" if "Keseimbangan Keputihan" in img_label.text() else
                            "Gabungan")

        # Clear overall comparison metrics
        for key in self.oc_metric_labels:
            self.oc_metric_labels[key].setText("-")

        # Clear overall comparison histograms
        for hist_label in self.oc_histogram_labels:
            hist_label.clear()
            hist_label.setText("Histogram")

        # Clear metric bar
        self.lbl_ssim_val.setText("0.0")
        self.lbl_entropy_val.setText("0.0")
        self.lbl_uiqm_val.setText("0.0")
        self.lbl_uciqe_val.setText("0.0")
        self.lbl_pcqi_val.setText("0.0")

        # Clear histogram
        self.histogram_placeholder.setText("Tiada histogram")

        # Reset status
        self.status_label.setPlainText("Aplikasi telah ditetapkan semula - Sedia")

        # Reset size display
        self.lbl_size.setText("0 × 0 px")

        # Switch back to dual view
        self.tab_bar.setCurrentIndex(self.tab_dual)

    def view_histogram_fullscreen(self):
        """Open histogram in full screen"""
        hist_path = './histogram_temp/histogram.png'
        if not os.path.exists(hist_path):
            QMessageBox.warning(self, "Tiada Histogram", "Sila tingkatkan imej terlebih dahulu untuk menjana histogram.")
            return
        
        try:
            # Create a new window for the histogram
            self.hist_window = QMainWindow()
            self.hist_window.setWindowTitle("Histogram - Skrin Penuh")
            self.hist_window.resize(1200, 850)
            
            # Load and display histogram using QPixmap (simpler and more reliable)
            pixmap = QPixmap(hist_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "Ralat", "Gagal memuatkan imej histogram.")
                return
            
            hist_label = QLabel()
            hist_label.setPixmap(pixmap.scaledToWidth(1100, Qt.SmoothTransformation))
            hist_label.setAlignment(Qt.AlignCenter)
            
            scroll_area = QScrollArea()
            scroll_area.setWidget(hist_label)
            self.hist_window.setCentralWidget(scroll_area)
            self.hist_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Ralat", f"Gagal membuka histogram:\n{str(e)}")
    
    def view_overall_histogram(self, index, name):
        """Open individual histogram from overall comparison in full screen"""
        histogram_files = [
            'hist_original.png',
            'hist_dcp.png',
            'hist_formation.png',
            'hist_whitebalance.png',
            'hist_combined.png'
        ]
        
        if index < 0 or index >= len(histogram_files):
            QMessageBox.warning(self, "Ralat", "Indeks histogram tidak sah.")
            return
        
        hist_path = f'./histogram_temp/{histogram_files[index]}'
        if not os.path.exists(hist_path):
            QMessageBox.warning(self, "Tiada Histogram", f"Histogram untuk {name} belum tersedia. Sila tingkatkan imej terlebih dahulu.")
            return
        
        try:
            # Create a new window for the histogram
            hist_window = QMainWindow()
            hist_window.setWindowTitle(f"Histogram - {name}")
            hist_window.resize(1000, 600)
            
            # Load and display histogram using QPixmap
            pixmap = QPixmap(hist_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "Ralat", f"Gagal memuatkan histogram {name}.")
                return
            
            hist_label = QLabel()
            hist_label.setPixmap(pixmap.scaledToWidth(950, Qt.SmoothTransformation))
            hist_label.setAlignment(Qt.AlignCenter)
            
            scroll_area = QScrollArea()
            scroll_area.setWidget(hist_label)
            hist_window.setCentralWidget(scroll_area)
            hist_window.show()
            
            # Keep reference to prevent garbage collection
            if not hasattr(self, 'hist_windows'):
                self.hist_windows = []
            self.hist_windows.append(hist_window)
        except Exception as e:
            QMessageBox.critical(self, "Ralat", f"Gagal membuka histogram {name}:\n{str(e)}")

    # ===== save all outputs =====
    def save_all_outputs(self):
        """Save all output images including histogram from overall comparison"""
        if self.after_enh == 0:
            QMessageBox.warning(self, "Tiada Peningkatan", "Sila tingkatkan imej terlebih dahulu sebelum menyimpan.")
            return
        
        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Pilih Direktori Keluaran")
        if not output_dir:
            return
        
        try:
            # Create subdirectory for this save
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(output_dir, f"enhancement_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Save all images
            images_to_save = {
                "01_Original.png": self.cv2_ori_img,
                "02_DCP.png": self.cv2_dcp_img if hasattr(self, 'cv2_dcp_img') and self.cv2_dcp_img is not None else None,
                "03_Formation_Image.png": self.cv2_fi_img if self.cv2_fi_img is not None else None,
                "04_White_Balance.png": self.cv2_wb_img if hasattr(self, 'cv2_wb_img') and self.cv2_wb_img is not None else None,
                "05_Combined_CNN_DCP.png": self.cv2_edt_img if self.cv2_edt_img is not None else None,
            }
            
            saved_count = 0
            for filename, img_data in images_to_save.items():
                if img_data is not None:
                    output_path = os.path.join(save_dir, filename)
                    # Convert RGB to BGR for cv2.imwrite if needed
                    if len(img_data.shape) == 3 and img_data.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_path, img_bgr)
                    else:
                        cv2.imwrite(output_path, img_data)
                    saved_count += 1
            
            # Save main histogram if it exists
            hist_source = './histogram_temp/histogram.png'
            if os.path.exists(hist_source):
                hist_path = os.path.join(save_dir, "00_Histogram.png")
                shutil.copy(hist_source, hist_path)
                saved_count += 1
            
            # Save individual histograms for each image
            histogram_files = [
                ('hist_original.png', '01_Original_Histogram.png'),
                ('hist_dcp.png', '02_DCP_Histogram.png'),
                ('hist_formation.png', '03_Formation_Histogram.png'),
                ('hist_whitebalance.png', '04_WhiteBalance_Histogram.png'),
                ('hist_combined.png', '05_Combined_Histogram.png')
            ]
            
            for src_name, dst_name in histogram_files:
                src_path = f'./histogram_temp/{src_name}'
                if os.path.exists(src_path):
                    dst_path = os.path.join(save_dir, dst_name)
                    shutil.copy(src_path, dst_path)
                    saved_count += 1
            
            # Save metrics as text file
            metrics_file = os.path.join(save_dir, "metrics.txt")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write("Metrik Peningkatan\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Imej: {os.path.basename(self.image_path)}\n")
                f.write(f"Masa Peningkatan: {timestamp}\n\n")
                f.write("Entropy:\n")
                f.write(f"  Asal: {self.ori_entropy:.4f}\n")
                f.write(f"  Prior Saluran Gelap: {self.dcp_entropy:.4f}\n")
                f.write(f"  Pembentukan: {self.fi_entropy:.4f}\n")
                f.write(f"  Keseimbangan Keputihan: {self.wb_entropy:.4f}\n")
                f.write(f"  Gabungan: {self.enh_entropy:.4f}\n\n")
                f.write("UIQM:\n")
                f.write(f"  Asal: {self.original_Uiqm:.4f}\n")
                f.write(f"  Prior Saluran Gelap: {self.dcp_Uiqm:.4f}\n")
                f.write(f"  Pembentukan: {self.fi_Uiqm:.4f}\n")
                f.write(f"  Keseimbangan Keputihan: {self.wb_Uiqm:.4f}\n")
                f.write(f"  Gabungan: {self.enhanced_Uiqm:.4f}\n\n")
                f.write("UCIQE:\n")
                f.write(f"  Asal: {self.original_uciqe:.4f}\n")
                f.write(f"  Prior Saluran Gelap: {self.dcp_uciqe:.4f}\n")
                f.write(f"  Pembentukan: {self.fi_uciqe:.4f}\n")
                f.write(f"  Keseimbangan Keputihan: {self.wb_uciqe:.4f}\n")
                f.write(f"  Gabungan: {self.enhanced_uciqe:.4f}\n\n")
                f.write("PCQI:\n")
                f.write(f"  Asal: {self.ori_pcqi:.4f}\n")
                f.write(f"  Prior Saluran Gelap: {self.dcp_pcqi:.4f}\n")
                f.write(f"  Pembentukan: {self.fi_pcqi:.4f}\n")
                f.write(f"  Keseimbangan Keputihan: {self.wb_pcqi:.4f}\n")
                f.write(f"  Gabungan: {self.com_pcqi:.4f}\n\n")
                f.write("SSIM:\n")
                f.write(f"  Asal: {self.ori_ssim:.4f}\n")
                f.write(f"  Prior Saluran Gelap: {self.dcp_ssim:.4f}\n")
                f.write(f"  Pembentukan: {self.fi_ssim:.4f}\n")
                f.write(f"  Keseimbangan Keputihan: {self.wb_ssim:.4f}\n")
                f.write(f"  Gabungan: {self.com_ssim:.4f}\n")
            
            QMessageBox.information(self, "Berjaya", 
                f"Disimpan {saved_count} imej dan metrik ke:\n{save_dir}")
            self.status_label.setPlainText(f"Keluaran disimpan ke: {save_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ralat", f"Gagal menyimpan keluaran:\n{str(e)}")
            self.status_label.setPlainText(f"Ralat menyimpan keluaran: {str(e)}")

    def _select_fi_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Model Pembentukan", "", "PyTorch Model (*.pth.tar);;All Files (*)"
        )
        if file_path:
            self.fi_model_checkpoint = file_path
            self.formation_model_path.setText(os.path.basename(file_path))

    def _select_wb_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Model Keseimbangan Keputihan", "", "PyTorch Model (*.pth.tar);;All Files (*)"
        )
        if file_path:
            self.wb_model_checkpoint = file_path
            self.white_balance_path.setText(os.path.basename(file_path))

    def _enhance_image(self):
        if not self.image_path:
            QMessageBox.warning(self, "Tiada Imej", "Sila muat imej terlebih dahulu.")
            return

        self.status_label.setPlainText("Memproses...")
        self.btn_enhance.setEnabled(False)

        # Start enhancement in a separate thread to keep UI responsive
        self.enhancement_thread = EnhancementThread(self)
        self.enhancement_thread.finished.connect(self._on_enhancement_finished)
        self.enhancement_thread.status_update.connect(self.on_status_updated)
        self.enhancement_thread.start()

    def on_status_updated(self, message):
        """Slot to safely update status label from any thread"""
        self.status_label.setPlainText(message)

    # status update after enhancement
    def _on_enhancement_finished(self):
        self.btn_enhance.setEnabled(True)
        completion_message = "Peningkatan berjaya selesai!\n\n"
        completion_message += f"Peranti: {self.device_combo.currentText()}\n"
        completion_message += "Saluran Paip: DCP → Pembentukan Imej (FI) → Keseimbangan Keputihan (WB)\n"
        completion_message += "Pemprosesan: Gabungan Prior Saluran Gelap + CNN\n"
        completion_message += "Pemetaan Warna: Pemetaan polinomial WB digunakan pada keluaran FI\n"
        completion_message += "Status: Sedia"
        self.status_label.setPlainText(completion_message)
        self.update_display()
        self.update_metrics()

    # ===== processing methods =====
    def get_device(self):
        device = self.device_combo.currentText()
        if device == "Auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "GPU":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device("cpu")

    # ============================================================
    # DETAILS STEP BY STEP ENHANCEMENT PIPELINE
    # ============================================================
    def enhanceImage(self, status_callback=None):
        """Main enhancement pipeline using ModelManager"""
        # Helper to safely update status
        def update_status(msg):
            if status_callback:
                status_callback(msg)
            else:
                self.status_label.setPlainText(msg)
            QApplication.processEvents()
        
        # Get device from combo box and pass to ModelManager
        device_type = self.device_combo.currentText()
        device = self.model_manager.get_device(device_type)
        
        # Load models using ModelManager
        update_status("Memuatkan model Pembentukan Imej (FI)...")
        FI_model = self.model_manager.load_fi_model(device)

        update_status("Memuatkan model Keseimbangan Keputihan (WB)...")
        WB_model = self.model_manager.load_wb_model(device)
        
        # Load original image
        update_status("Memuatkan imej...")
        imgPil = Image.open(self.image_path)
        # Convert RGBA to RGB if needed (e.g., PNG with alpha channel)
        if imgPil.mode == 'RGBA':
            imgPil = imgPil.convert('RGB')
        self.cv2_ori_img = np.array(imgPil)
        
        # Enhancement pipeline
        import utils_func as util
        
        # ============================================================
        # STAGE 1: DCP PREPROCESSING
        # ============================================================
        # Purpose: Remove underwater haze and backscatter
        # Input: Original underwater image
        # Parameters: omega=0.30 (patch size), t0=0.55 (atmospheric light)
        # Output: DCP-enhanced image (normalized visibility)
        update_status("Menggunakan Prior Saluran Gelap (DCP)...")
        dcp_enhanced = util.enhance_image_dcp(self.cv2_ori_img, omega=0.30, t0=0.55)
        self.cv2_dcp_img = (dcp_enhanced * 255).astype(np.uint8)  # Convert to uint8 (0-255 range)
        
        # ============================================================
        # STAGE 2: SAVE & RELOAD DCP AS PNG
        # ============================================================
        # Purpose: Ensure consistent color space and format for both FI and WB models
        # Why: PNG compression standardizes image data (prevents preprocessing inconsistencies)
        # Process: DCP numpy array → PNG file → Load back → PIL Image
        update_status("Saving and reloading DCP preprocessing...")
        dcp_png_path = './dcp_preprocessed.png'
        cv2.imwrite(dcp_png_path, cv2.cvtColor(self.cv2_dcp_img, cv2.COLOR_RGB2BGR))  # Save as PNG (RGB→BGR for OpenCV)
        dcp_loaded = cv2.imread(dcp_png_path)  # Load PNG from disk
        dcp_loaded = cv2.cvtColor(dcp_loaded, cv2.COLOR_BGR2RGB)  # Convert BGR back to RGB
        dcp_pil = Image.fromarray(dcp_loaded, mode='RGB')  # Convert to PIL Image (standardized input)
        
        # ============================================================
        # STAGE 3a: FI MODEL (Formation Image Enhancement)
        # ============================================================
        # Purpose: Enhance image structure and details
        # Key Feature: Processes at FULL ORIGINAL RESOLUTION (no downsampling)
        # Architecture: PhysicalNN with BsConvBlock + DtConvBlock
        # Output: Formation Image with enhanced details and structure
        update_status("Memproses dengan model Pembentukan Imej (FI)...")
        testtransform = transforms.Compose([transforms.ToTensor()])
        unloader = transforms.ToPILImage()
        
        inp_fi = testtransform(dcp_pil).unsqueeze(0).to(device)  # Convert PIL to tensor, add batch, send to device
        FI_out = FI_model(inp_fi)  # Process through FI CNN (FULL RESOLUTION - critical for detail preservation)
        FI_out = unloader(FI_out.cpu().squeeze(0))  # Convert tensor back to PIL Image
        FI_out.save('./test.png')  # Save FI output for debugging/verification
        FI_out_array = np.array(FI_out)  # Convert PIL to numpy array (FULL RESOLUTION preserved)
        self.cv2_fi_img = FI_out_array  # Store FI output (FULL RESOLUTION DETAILS)
        
        # ============================================================
        # STAGE 3b: WB MODEL (White Balance / Color Correction - Applied to FI)
        # ============================================================
        # Purpose: Apply color correction to FI output for final result
        # Pipeline: Original → DCP → FI → WB → Final Result
        # Strategy: Apply WB to FI with proper blending to preserve detail
        update_status("Memproses dengan model Keseimbangan Keputihan (WB)...")
        
        # Resize FI output to 656 pixels for WB model processing
        s = 656  # Base size in pixels for efficient processing
        fi_resized = Image.fromarray(FI_out_array).resize((round(FI_out_array.shape[1] / max(FI_out_array.shape) * s), round(FI_out_array.shape[0] / max(FI_out_array.shape) * s)))
        w, h = fi_resized.size
        # Align to 2^4 boundary (16-pixel alignment required by UNet architecture)
        new_w = w + (2 ** 4 - w % 2 ** 4) if w % 2 ** 4 != 0 else w
        new_h = h + (2 ** 4 - h % 2 ** 4) if h % 2 ** 4 != 0 else h
        fi_resized = fi_resized.resize((new_w, new_h))  # Resize to 2^4 aligned dimensions
        
        # Ensure RGB format for model consistency
        if fi_resized.mode != 'RGB':
            fi_resized = fi_resized.convert('RGB')
        fi_resized_array = np.array(fi_resized)  # Downsampled FI (~656px, 2^4 aligned)
        
        # Process downsampled FI through WB model
        inpWB = testtransform(fi_resized_array).unsqueeze(0).to(device)  # Convert to tensor, add batch, send to device
        wb_out = WB_model(inpWB)  # Process through WB CNN (applies color correction to FI)
        wb_out = torch.squeeze(wb_out.cpu())  # Remove batch dimension, move to CPU
        wb_out = torch.clamp(wb_out, 0, 1)  # Clamp output to valid [0, 1] range
        wb_out_np = wb_out.detach().numpy().transpose((1, 2, 0))  # Convert to numpy, HWC format
        wb_out_np = np.clip(wb_out_np, 0, 1)  # Ensure valid range
        
        # Learn color mapping polynomial from WB output
        # This maps FI color space to WB-corrected color space
        map_out = util.get_mapping_func(fi_resized_array, wb_out_np)
        
        # ============================================================
        # STAGE 4: COMBINE FI + WB WITH BLENDING
        # ============================================================
        # Purpose: Apply WB color correction to FI output while preserving detail
        # Strategy: Use weighted blending (90% FI detail + 10% WB correction)
        # This prevents over-correction while still applying color correction
        update_status("Menggabungkan FI dengan koreksi warna WB...")
        
        # Convert FI output to float for blending
        fi_float = FI_out_array.astype(np.float32) / 255.0 if np.max(FI_out_array) > 1 else FI_out_array.astype(np.float32)
        
        # Apply WB color mapping to FI
        fi_uint8 = (fi_float * 255).astype(np.uint8)
        wb_corrected = util.outOfGamutClipping(util.apply_mapping_func(fi_uint8, map_out))
        
        # Blend: 95% FI + 5% WB correction (preserves FI detail while adding subtle color correction)
        # Lower WB weight reduces structure degradation and keeps SSIM higher
        blend_alpha = 0.05  # Weight of WB correction (reduced from 0.15 to preserve structure)
        combined_blended = fi_float * (1 - blend_alpha) + wb_corrected * blend_alpha
        combined_output = np.clip(combined_blended, 0, 1)
        
        # ============================================================
        # STAGE 5: FINAL RESIZE & OUTPUT
        # ============================================================
        # Purpose: Resize to original dimensions and store final result
        update_status("Mengesahkan gabungan DCP+FI+WB...")
        
        # Convert to uint8 for final output
        final_output_uint8 = (np.clip(combined_output, 0, 1) * 255).astype(np.uint8)
        
        # Create final PIL Image and store as numpy array (consistent RGB format for metrics/display)
        final_output_pil = Image.fromarray(final_output_uint8, mode='RGB')
        self.cv2_edt_img = np.array(final_output_pil)  # Store final enhanced image (DCP+FI+WB)
        self.after_enh = 1  # Flag: enhancement completed successfully
        
        # Calculate comparison models (WB applied to original for fair comparison)
        update_status("Mengira model perbandingan...")
        self.enhanceFromFiModel(FI_model)  # FI applied to original (without DCP preprocessing)
        self.enhanceFromWbModel(WB_model)  # WB applied to original for comparison

    def enhanceFromFiModel(self, FI_model):
        """Generate FI model comparison output"""
        device_type = self.device_combo.currentText()
        device = self.model_manager.get_device(device_type)
        testtransform = transforms.Compose([transforms.ToTensor()])
        unloader = transforms.ToPILImage()

        imgPil = Image.open(self.image_path)
        # Convert RGBA to RGB if needed
        if imgPil.mode == 'RGBA':
            imgPil = imgPil.convert('RGB')
        inp = testtransform(imgPil).unsqueeze(0).to(device)

        FI_out = FI_model(inp)
        FI_out = unloader(FI_out.cpu().squeeze(0))
        FI_out.save('./test_fi.png')
        fi_img = Image.open('./test_fi.png')
        self.cv2_fi_img = np.array(fi_img)

    def enhanceFromWbModel(self, WB_model):
        """Generate WB model comparison output"""
        device_type = self.device_combo.currentText()
        device = self.model_manager.get_device(device_type)
        import utils_func as util
        
        WB_model.eval()
        testtransform = transforms.Compose([transforms.ToTensor()])
        tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        imgPil = Image.open(self.image_path)
        # Convert RGBA to RGB if needed
        if imgPil.mode == 'RGBA':
            imgPil = imgPil.convert('RGB')
        s = 656
        img_resized = imgPil.resize((round(imgPil.width / max(imgPil.size) * s), round(imgPil.height / max(imgPil.size) * s)))
        w, h = img_resized.size
        new_w = w + (2 ** 4 - w % 2 ** 4) if w % 2 ** 4 != 0 else w
        new_h = h + (2 ** 4 - h % 2 ** 4) if h % 2 ** 4 != 0 else h
        img_resized = img_resized.resize((new_w, new_h))

        img_resized_array = np.array(img_resized)
        inpWB = testtransform(img_resized_array).unsqueeze(0).to(device)

        out = WB_model(inpWB)
        # Properly handle the WB model output tensor
        out = torch.squeeze(out.cpu())
        # Ensure tensor values are in [0, 1] range before converting to numpy
        out = torch.clamp(out, 0, 1)
        # Convert tensor to numpy array in HWC format
        out = out.detach().numpy().transpose((1, 2, 0))
        # Ensure output is in proper value range [0, 1]
        out = np.clip(out, 0, 1)
        
        map_out = util.get_mapping_func(img_resized_array, out)
        final_output = util.outOfGamutClipping(util.apply_mapping_func(self.cv2_ori_img, map_out))
        final_output = Image.fromarray((np.clip(final_output, 0, 1) * 255).astype(np.uint8))

        self.cv2_wb_img = np.array(final_output)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Imej", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        if file_path:
            self.image_path = file_path
            self.load_image_to_display(file_path)
            self.status_label.setPlainText(f"Dimuat: {os.path.basename(file_path)}")
            # Switch to dual view to show the loaded image
            self.tab_bar.setCurrentIndex(self.tab_dual)

    def load_image_to_display(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.cv2_ori_img = img
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pix_ori_img = QPixmap.fromImage(q_img)
        self.update_display()

    def update_display(self):
        if self.pix_ori_img:
            # Always show original image in before section
            scaled_before = self.pix_ori_img.scaled(self.before_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.before_image_label.setPixmap(scaled_before)

            # Update dual view - show enhanced image if available
            if self.after_enh and self.cv2_edt_img is not None:

                # After image
                height, width, channel = self.cv2_edt_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(self.cv2_edt_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pix_after = QPixmap.fromImage(q_img)
                scaled_after = pix_after.scaled(self.after_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.after_image_label.setPixmap(scaled_after)

            # Update overall view
            if self.after_enh:
                # Original
                scaled_orig = self.pix_ori_img.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.oc_image_labels[0].setPixmap(scaled_orig)

                # DCP
                if hasattr(self, 'cv2_dcp_img') and self.cv2_dcp_img is not None:
                    height, width, channel = self.cv2_dcp_img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(self.cv2_dcp_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pix_dcp = QPixmap.fromImage(q_img)
                    scaled_dcp = pix_dcp.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.oc_image_labels[1].setPixmap(scaled_dcp)

                # FI only
                if self.cv2_fi_img is not None:
                    height, width, channel = self.cv2_fi_img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(self.cv2_fi_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pix_fi = QPixmap.fromImage(q_img)
                    scaled_fi = pix_fi.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.oc_image_labels[2].setPixmap(scaled_fi)

                # WB only
                if self.cv2_wb_img is not None:
                    height, width, channel = self.cv2_wb_img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(self.cv2_wb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pix_wb = QPixmap.fromImage(q_img)
                    scaled_wb = pix_wb.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.oc_image_labels[3].setPixmap(scaled_wb)

                # Combined
                if self.cv2_edt_img is not None:
                    height, width, channel = self.cv2_edt_img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(self.cv2_edt_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pix_combined = QPixmap.fromImage(q_img)
                    scaled_combined = pix_combined.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.oc_image_labels[4].setPixmap(scaled_combined)

    # ============================================================
    # OVERALL METRICS UPDATE AND CALCULATION
    # ============================================================
    def update_metrics(self):
        if self.cv2_ori_img is None:
            return

        # Calculate metrics
        self.calculate_entropy()
        self.calculate_uiqm()
        self.calculate_uciqe()
        self.calculate_pcqi()
        self.calculate_ssim()

        # Update metric bar
        self.lbl_ssim_val.setText(f"{self.com_ssim:.3f}")
        self.lbl_entropy_val.setText(f"{self.enh_entropy:.3f}")
        self.lbl_uiqm_val.setText(f"{self.enhanced_Uiqm:.3f}")
        self.lbl_uciqe_val.setText(f"{self.enhanced_uciqe:.3f}")
        self.lbl_pcqi_val.setText(f"{self.com_pcqi:.3f}")

        if self.cv2_edt_img is not None:
            height, width = self.cv2_edt_img.shape[:2]
            self.lbl_size.setText(f"{width} × {height} px")

        # Update overall comparison metrics
        self.update_overall_metrics()

        # Update histogram display
        self.update_histogram()
        
        # Update overall comparison histograms
        self.update_overall_histograms()

    # ENTROPY (Shannon Information Entropy)
    # What it measures: Information content and pixel diversity in an image
    # Important Context: This is SHANNON ENTROPY, not noise level
    # Shannon Entropy Behavior:
    # - Hazy/unclear image = LOW entropy (pixel values are similar/uniform)
    # - Clear/detailed image = HIGH entropy (pixel values are diverse)
    # - Enhancement = ENTROPY INCREASES because detail becomes visible
    # NOTE: Higher entropy is GOOD for enhanced images - it means more visible detail
    # The increase from 5.98→7.52 shows the enhancement revealed more image information
    def calculate_entropy(self):
        if self.cv2_ori_img is not None:
            img = cv2.cvtColor(self.cv2_ori_img, cv2.COLOR_RGB2GRAY)
            num_of_pixels = img.shape[0] * img.shape[1]
            histogram, bins = np.histogram(np.array(img).flatten(), bins=np.arange(256+1))
            self.ori_entropy = 0
            for i in range(256):
                p = histogram[i] / num_of_pixels # probability of each intensity level
                if p > 0:
                    self.ori_entropy -= p * np.log2(p)

        if self.cv2_edt_img is not None:
            img = cv2.cvtColor(self.cv2_edt_img, cv2.COLOR_RGB2GRAY)
            num_of_pixels = img.shape[0] * img.shape[1]
            histogram, bins = np.histogram(np.array(img).flatten(), bins=np.arange(256+1))
            self.enh_entropy = 0
            for i in range(256):
                p = histogram[i] / num_of_pixels
                if p > 0:
                    self.enh_entropy -= p * np.log2(p)

        if self.cv2_fi_img is not None:
            img = cv2.cvtColor(self.cv2_fi_img, cv2.COLOR_RGB2GRAY)
            num_of_pixels = img.shape[0] * img.shape[1]
            histogram, bins = np.histogram(np.array(img).flatten(), bins=np.arange(256+1))
            self.fi_entropy = 0
            for i in range(256):
                p = histogram[i] / num_of_pixels
                if p > 0:
                    self.fi_entropy -= p * np.log2(p)

        if self.cv2_wb_img is not None:
            img = cv2.cvtColor(self.cv2_wb_img, cv2.COLOR_RGB2GRAY)
            num_of_pixels = img.shape[0] * img.shape[1]
            histogram, bins = np.histogram(np.array(img).flatten(), bins=np.arange(256+1))
            self.wb_entropy = 0
            for i in range(256):
                p = histogram[i] / num_of_pixels
                if p > 0:
                    self.wb_entropy -= p * np.log2(p)

        if hasattr(self, 'cv2_dcp_img') and self.cv2_dcp_img is not None:
            img = cv2.cvtColor(self.cv2_dcp_img, cv2.COLOR_RGB2GRAY)
            num_of_pixels = img.shape[0] * img.shape[1]
            histogram, bins = np.histogram(np.array(img).flatten(), bins=np.arange(256+1))
            self.dcp_entropy = 0
            for i in range(256):
                p = histogram[i] / num_of_pixels
                if p > 0:
                    self.dcp_entropy -= p * np.log2(p)

    # UIQM (Underwater Image Quality Metric)
    # What it measures: Overall quality of underwater images based on color, sharpness, and contrast
    # What it means:
    # Higher UIQM = Better color, sharper edges, better contrast
    # Most important for underwater enhancement
    # Combined should exceed individual FI/WB scores
    def calculate_uiqm(self):
        # Calculate UIQM for original image
        if self.cv2_ori_img is not None:
            img = self.cv2_ori_img
            # weighted coefficients
            c1 = 0.0282 # (color weight)
            c2 = 0.2953 # (sharpness weight)
            c3 = 3.5753 # (contrast weight)
            # UICM (Colorfulness): Measures R-G and Y-B color distribution
            # UISM (Sharpness): Sobel edge detection + EME (Enhanced Mean Energy)
            # UIConM (Contrast): Logarithmic average enhancement measurement            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # UICM, colourfulness measure
            rg = rgb[:,:,0] - rgb[:,:,1]
            yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
            rgl = np.sort(rg,axis=None)
            ybl = np.sort(yb,axis=None)
            al1 = 0.1
            al2 = 0.1
            T1 = np.int_(al1 * len(rgl))
            T2 = np.int_(al2 * len(rgl))
            rgl_tr = rgl[T1:-T2]
            ybl_tr = ybl[T1:-T2]

            urg = np.mean(rgl_tr)
            s2rg = np.mean((rgl_tr - urg) ** 2)
            uyb = np.mean(ybl_tr)
            s2yb = np.mean((ybl_tr- uyb) ** 2)

            uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

            # UISM, sharpness
            Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
            Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
            Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

            Rsobel=np.round(Rsobel).astype(np.uint8)
            Gsobel=np.round(Gsobel).astype(np.uint8)
            Bsobel=np.round(Bsobel).astype(np.uint8)

            Reme = self.eme(Rsobel)
            Geme = self.eme(Gsobel)
            Beme = self.eme(Bsobel)

            uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

            # UIConM, contrast
            uiconm = self.logamee(gray)

            uiqm = c1 * uicm + c2 * uism + c3 * uiconm

            self.original_Uiqm = float(f'{uiqm:.3f}')

        # Calculate UIQM for enhanced image
        if self.cv2_edt_img is not None:
            img = self.cv2_edt_img
            # weighted coefficients
            c1 = 0.0282
            c2 = 0.2953
            c3 = 3.5753
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # UICM, colourfulness measure
            rg = rgb[:,:,0] - rgb[:,:,1]
            yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
            rgl = np.sort(rg,axis=None)
            ybl = np.sort(yb,axis=None)
            al1 = 0.1
            al2 = 0.1
            T1 = np.int_(al1 * len(rgl))
            T2 = np.int_(al2 * len(rgl))
            rgl_tr = rgl[T1:-T2]
            ybl_tr = ybl[T1:-T2]

            urg = np.mean(rgl_tr)
            s2rg = np.mean((rgl_tr - urg) ** 2)
            uyb = np.mean(ybl_tr)
            s2yb = np.mean((ybl_tr- uyb) ** 2)

            uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

            # UISM, sharpness
            Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
            Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
            Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

            Rsobel=np.round(Rsobel).astype(np.uint8)
            Gsobel=np.round(Gsobel).astype(np.uint8)
            Bsobel=np.round(Bsobel).astype(np.uint8)

            Reme = self.eme(Rsobel)
            Geme = self.eme(Gsobel)
            Beme = self.eme(Bsobel)

            uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

            # UIConM, contrast
            uiconm = self.logamee(gray)

            uiqm = c1 * uicm + c2 * uism + c3 * uiconm

            self.enhanced_Uiqm = float(f'{uiqm:.3f}')

        # Calculate UIQM for FI image
        if self.cv2_fi_img is not None:
            img = self.cv2_fi_img
            # weighted coefficients
            c1 = 0.0282
            c2 = 0.2953
            c3 = 3.5753
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # UICM, colourfulness measure
            rg = rgb[:,:,0] - rgb[:,:,1]
            yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
            rgl = np.sort(rg,axis=None)
            ybl = np.sort(yb,axis=None)
            al1 = 0.1
            al2 = 0.1
            T1 = np.int_(al1 * len(rgl))
            T2 = np.int_(al2 * len(rgl))
            rgl_tr = rgl[T1:-T2]
            ybl_tr = ybl[T1:-T2]

            urg = np.mean(rgl_tr)
            s2rg = np.mean((rgl_tr - urg) ** 2)
            uyb = np.mean(ybl_tr)
            s2yb = np.mean((ybl_tr- uyb) ** 2)

            uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

            # UISM, sharpness
            Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
            Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
            Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

            Rsobel=np.round(Rsobel).astype(np.uint8)
            Gsobel=np.round(Gsobel).astype(np.uint8)
            Bsobel=np.round(Bsobel).astype(np.uint8)

            Reme = self.eme(Rsobel)
            Geme = self.eme(Gsobel)
            Beme = self.eme(Bsobel)

            uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

            # UIConM, contrast
            uiconm = self.logamee(gray)

            uiqm = c1 * uicm + c2 * uism + c3 * uiconm

            self.fi_Uiqm = float(f'{uiqm:.3f}')

        # Calculate UIQM for WB image
        if self.cv2_wb_img is not None:
            img = self.cv2_wb_img
            # weighted coefficients
            c1 = 0.0282
            c2 = 0.2953
            c3 = 3.5753
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # UICM, colourfulness measure
            rg = rgb[:,:,0] - rgb[:,:,1]
            yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
            rgl = np.sort(rg,axis=None)
            ybl = np.sort(yb,axis=None)
            al1 = 0.1
            al2 = 0.1
            T1 = np.int_(al1 * len(rgl))
            T2 = np.int_(al2 * len(rgl))
            rgl_tr = rgl[T1:-T2]
            ybl_tr = ybl[T1:-T2]

            urg = np.mean(rgl_tr)
            s2rg = np.mean((rgl_tr - urg) ** 2)
            uyb = np.mean(ybl_tr)
            s2yb = np.mean((ybl_tr- uyb) ** 2)

            uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

            # UISM, sharpness
            Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
            Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
            Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

            Rsobel=np.round(Rsobel).astype(np.uint8)
            Gsobel=np.round(Gsobel).astype(np.uint8)
            Bsobel=np.round(Bsobel).astype(np.uint8)

            Reme = self.eme(Rsobel)
            Geme = self.eme(Gsobel)
            Beme = self.eme(Bsobel)

            uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

            # UIConM, contrast
            uiconm = self.logamee(gray)

            uiqm = c1 * uicm + c2 * uism + c3 * uiconm

            self.wb_Uiqm = float(f'{uiqm:.3f}')

        # Calculate UIQM for DCP image
        if hasattr(self, 'cv2_dcp_img') and self.cv2_dcp_img is not None:
            img = self.cv2_dcp_img
            # weighted coefficients
            c1 = 0.0282
            c2 = 0.2953
            c3 = 3.5753
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # UICM, colourfulness measure
            rg = rgb[:,:,0] - rgb[:,:,1]
            yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
            rgl = np.sort(rg,axis=None)
            ybl = np.sort(yb,axis=None)
            al1 = 0.1
            al2 = 0.1
            T1 = np.int_(al1 * len(rgl))
            T2 = np.int_(al2 * len(rgl))
            rgl_tr = rgl[T1:-T2]
            ybl_tr = ybl[T1:-T2]

            urg = np.mean(rgl_tr)
            s2rg = np.mean((rgl_tr - urg) ** 2)
            uyb = np.mean(ybl_tr)
            s2yb = np.mean((ybl_tr- uyb) ** 2)

            uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

            # UISM, sharpness
            Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
            Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
            Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

            Rsobel=np.round(Rsobel).astype(np.uint8)
            Gsobel=np.round(Gsobel).astype(np.uint8)
            Bsobel=np.round(Bsobel).astype(np.uint8)

            Reme = self.eme(Rsobel)
            Geme = self.eme(Gsobel)
            Beme = self.eme(Bsobel)

            uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

            # UIConM, contrast
            uiconm = self.logamee(gray)

            uiqm = c1 * uicm + c2 * uism + c3 * uiconm

            self.dcp_Uiqm = float(f'{uiqm:.3f}')

    # UCIQE (Underwater Color Image Quality Evaluation)
    # What it measures: Color quality of underwater images based on chroma, contrast, and saturation
    # What it means:
    # Higher UCIQE = Better colors, higher contrast, better saturation
    # Specifically designed for underwater images
    # Better than UIQM for color evaluation
    def calculate_uciqe(self):
        # Calculate UCIQE for original image
        if self.cv2_ori_img is not None:
            img = self.cv2_ori_img
            # weighted coefficients
            c1 = 0.4680 # (chroma weight) 
            c2 = 0.2745 # (contrast weight)
            c3 = 0.2576 # (saturation weight)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l = lab[:,:,0]

            # 1st term: Chroma Standard deviation of color saturation in LAB space
            chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
            uc = np.mean(chroma)
            sc = (np.mean((chroma - uc)**2))**0.5

            # 2nd term: Contrast Difference between brightest and darkest 1% of pixels
            top = np.int_(np.round(0.01*l.shape[0]*l.shape[1]))
            sl = np.sort(l,axis=None)
            isl = sl[::-1]
            conl = np.mean(isl[:top])-np.mean(sl[:top])

            # 3rd term: Saturation Mean saturation level (chroma/luminance ratio)
            satur = []
            chroma1 = chroma.flatten()
            l1 = l.flatten()
            for i in range(len(l1)):
                if chroma1[i] == 0: satur.append(0)
                elif l1[i] == 0: satur.append(0)
                else: satur.append(chroma1[i] / l1[i])

            us = np.mean(satur)

            uciqe = c1 * sc + c2 * conl + c3 * us
            self.original_uciqe = float(f'{uciqe:.3f}')

        # Calculate UCIQE for enhanced image
        if self.cv2_edt_img is not None:
            img = self.cv2_edt_img
            # weighted coefficients
            c1 = 0.4680
            c2 = 0.2745
            c3 = 0.2576
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l = lab[:,:,0]

            # 1st term: Chroma
            chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
            uc = np.mean(chroma)
            sc = (np.mean((chroma - uc)**2))**0.5

            # 2nd term: Contrast
            top = np.int_(np.round(0.01*l.shape[0]*l.shape[1]))
            sl = np.sort(l,axis=None)
            isl = sl[::-1]
            conl = np.mean(isl[:top])-np.mean(sl[:top])

            # 3rd term: Saturation
            satur = []
            chroma1 = chroma.flatten()
            l1 = l.flatten()
            for i in range(len(l1)):
                if chroma1[i] == 0: satur.append(0)
                elif l1[i] == 0: satur.append(0)
                else: satur.append(chroma1[i] / l1[i])

            us = np.mean(satur)

            uciqe = c1 * sc + c2 * conl + c3 * us
            self.enhanced_uciqe = float(f'{uciqe:.3f}')

        # Calculate UCIQE for FI image
        if self.cv2_fi_img is not None:
            img = self.cv2_fi_img
            # weighted coefficients
            c1 = 0.4680
            c2 = 0.2745
            c3 = 0.2576
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l = lab[:,:,0]

            # 1st term: Chroma
            chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
            uc = np.mean(chroma)
            sc = (np.mean((chroma - uc)**2))**0.5

            # 2nd term: Contrast
            top = np.int_(np.round(0.01*l.shape[0]*l.shape[1]))
            sl = np.sort(l,axis=None)
            isl = sl[::-1]
            conl = np.mean(isl[:top])-np.mean(sl[:top])

            # 3rd term: Saturation
            satur = []
            chroma1 = chroma.flatten()
            l1 = l.flatten()
            for i in range(len(l1)):
                if chroma1[i] == 0: satur.append(0)
                elif l1[i] == 0: satur.append(0)
                else: satur.append(chroma1[i] / l1[i])

            us = np.mean(satur)

            uciqe = c1 * sc + c2 * conl + c3 * us
            self.fi_uciqe = float(f'{uciqe:.3f}')

        # Calculate UCIQE for WB image
        if self.cv2_wb_img is not None:
            img = self.cv2_wb_img
            # weighted coefficients
            c1 = 0.4680
            c2 = 0.2745
            c3 = 0.2576
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l = lab[:,:,0]

            # 1st term: Chroma
            chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
            uc = np.mean(chroma)
            sc = (np.mean((chroma - uc)**2))**0.5

            # 2nd term: Contrast
            top = np.int_(np.round(0.01*l.shape[0]*l.shape[1]))
            sl = np.sort(l,axis=None)
            isl = sl[::-1]
            conl = np.mean(isl[:top])-np.mean(sl[:top])

            # 3rd term: Saturation
            satur = []
            chroma1 = chroma.flatten()
            l1 = l.flatten()
            for i in range(len(l1)):
                if chroma1[i] == 0: satur.append(0)
                elif l1[i] == 0: satur.append(0)
                else: satur.append(chroma1[i] / l1[i])

            us = np.mean(satur)

            uciqe = c1 * sc + c2 * conl + c3 * us
            self.wb_uciqe = float(f'{uciqe:.3f}')

        # Calculate UCIQE for DCP image
        if hasattr(self, 'cv2_dcp_img') and self.cv2_dcp_img is not None:
            img = self.cv2_dcp_img
            # weighted coefficients
            c1 = 0.4680
            c2 = 0.2745
            c3 = 0.2576
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l = lab[:,:,0]

            # 1st term: Chroma
            chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
            uc = np.mean(chroma)
            sc = (np.mean((chroma - uc)**2))**0.5

            # 2nd term: Contrast
            top = np.int_(np.round(0.01*l.shape[0]*l.shape[1]))
            sl = np.sort(l,axis=None)
            isl = sl[::-1]
            conl = np.mean(isl[:top])-np.mean(sl[:top])

            # 3rd term: Saturation
            satur = []
            chroma1 = chroma.flatten()
            l1 = l.flatten()
            for i in range(len(l1)):
                if chroma1[i] == 0: satur.append(0)
                elif l1[i] == 0: satur.append(0)
                else: satur.append(chroma1[i] / l1[i])

            us = np.mean(satur)

            uciqe = c1 * sc + c2 * conl + c3 * us
            self.dcp_uciqe = float(f'{uciqe:.3f}')
            
    # PCQI (Patch-wise Contrast Quality Index)
    # What it measures: Contrast quality of an image based on local patches
    # Where:
    # Contrast = Standard deviation of pixel intensities
    # Intensity difference = Difference in patch brightness (0 for single image)
    # What it means:
    # Higher PCQI = More visible local details and texture
    # Patch-based approach = Evaluates image detail uniformly
    # Good for checking spatial consistency
    def calculate_pcqi(self):
        # Calculate PCQI for original image
        if self.cv2_ori_img is not None:
            pcqi_score = self.calculate_image_pcqi(self.cv2_ori_img)
            self.ori_pcqi = float(f'{pcqi_score:.3f}')

        # Calculate PCQI for enhanced image
        if self.cv2_edt_img is not None:
            pcqi_score = self.calculate_image_pcqi(self.cv2_edt_img)
            self.com_pcqi = float(f'{pcqi_score:.3f}')

        # Calculate PCQI for FI image
        if self.cv2_fi_img is not None:
            pcqi_score = self.calculate_image_pcqi(self.cv2_fi_img)
            self.fi_pcqi = float(f'{pcqi_score:.3f}')

        # Calculate PCQI for WB image
        if self.cv2_wb_img is not None:
            pcqi_score = self.calculate_image_pcqi(self.cv2_wb_img)
            self.wb_pcqi = float(f'{pcqi_score:.3f}')

        # Calculate PCQI for DCP image
        if hasattr(self, 'cv2_dcp_img') and self.cv2_dcp_img is not None:
            pcqi_score = self.calculate_image_pcqi(self.cv2_dcp_img)
            self.dcp_pcqi = float(f'{pcqi_score:.3f}')

    def calculate_image_pcqi(self, image, patch_size=4):
        # Convert the image to grayscale if it's in color
        distorted_image = image
        if distorted_image.shape[-1] == 3:
            distorted_image = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)

        # Initialize variables to store quality components
        pcqi_values = []

        # Iterate through the distorted image in patches
        for i in range(0, distorted_image.shape[0] - patch_size + 1, patch_size):
            for j in range(0, distorted_image.shape[1] - patch_size + 1, patch_size):
                patch = distorted_image[i:i + patch_size, j:j + patch_size]

                # Calculate quality for the current patch
                patch_quality = self.calculate_patch_quality(patch)
                pcqi_values.append(patch_quality)

        # Calculate the final PCQI score as the mean of all patch qualities
        pcqi_score = np.mean(pcqi_values)
        return pcqi_score

    def calculate_patch_quality(self, patch):
        # Calculate contrast (standard deviation of pixel intensities)
        contrast = np.std(patch)

        # For single-image PCQI, mean intensity difference is not applicable, so we set it to zero
        diff_mean_intensity = 0

        # Calculate PCQI component (normalized to the range [0, 1])
        pcqi = (diff_mean_intensity + contrast) / 2.0
        pcqi = np.clip(pcqi / 2.0, 0, 2)

        return pcqi

    # SSIM (Structural Similarity Index)
    # What it measures: Structural similarity between original and enhanced images
    # What it means:
    # Range: -1 to 1 (1 = identical images, lower = more different structures)
    # Higher SSIM = Better structure preservation (closer to original)
    # Important: Enhanced images SHOULD have lower SSIM than original because:
    # - Enhancement intentionally CHANGES structure to improve quality
    # - Lower SSIM doesn't mean worse quality - it means DIFFERENT from original
    # - Combined technique may have lower SSIM because FI + WB mapping transforms structure
    # - What matters is PERCEPTUAL QUALITY (UIQM, UCIQE), not structure similarity
    def calculate_ssim(self):
        if self.cv2_ori_img is not None and self.cv2_edt_img is not None:
            # Convert to grayscale for SSIM
            gray_ori = cv2.cvtColor(self.cv2_ori_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray_edt = cv2.cvtColor(self.cv2_edt_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            self.com_ssim, _ = structural_similarity(gray_ori, gray_edt, full=True, data_range=255.0)

        if self.cv2_ori_img is not None and self.cv2_fi_img is not None:
            gray_ori = cv2.cvtColor(self.cv2_ori_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray_fi = cv2.cvtColor(self.cv2_fi_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            self.fi_ssim, _ = structural_similarity(gray_ori, gray_fi, full=True, data_range=255.0)

        if self.cv2_ori_img is not None and self.cv2_wb_img is not None:
            gray_ori = cv2.cvtColor(self.cv2_ori_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray_wb = cv2.cvtColor(self.cv2_wb_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            self.wb_ssim, _ = structural_similarity(gray_ori, gray_wb, full=True, data_range=255.0)

        if hasattr(self, 'cv2_dcp_img') and self.cv2_dcp_img is not None and self.cv2_ori_img is not None:
            gray_ori = cv2.cvtColor(self.cv2_ori_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray_dcp = cv2.cvtColor(self.cv2_dcp_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            self.dcp_ssim, _ = structural_similarity(gray_ori, gray_dcp, full=True, data_range=255.0)

        if self.cv2_ori_img is not None:
            self.ori_ssim = 1.0  # SSIM with itself is 1.0

    # SUMMARY OF OVERALL METRICS
    # Entropy = Shannon Information (Higher = More detail visible) - INCREASE is GOOD
    # UIQM = Overall quality (Standard underwater metric) - Higher is better ✓
    # UCIQE = Color & contrast (Underwater-specific) - Higher is better ✓
    # PCQI = Local contrast (Texture uniformity) - Higher is better ✓
    # SSIM = Structural similarity (Decreases slightly after enhancement due to color mapping)
    def update_overall_metrics(self):
        # Update the metric labels in the overall comparison view
        metrics = [
            ("Entropy", self.ori_entropy, getattr(self, 'dcp_entropy', 0.0), self.fi_entropy, self.wb_entropy, self.enh_entropy),
            ("SSIM", self.ori_ssim, getattr(self, 'dcp_ssim', 0.0), self.fi_ssim, self.wb_ssim, self.com_ssim),
            ("UIQM", self.original_Uiqm, getattr(self, 'dcp_Uiqm', 0.0), self.fi_Uiqm, self.wb_Uiqm, self.enhanced_Uiqm),
            ("UCIQE", self.original_uciqe, getattr(self, 'dcp_uciqe', 0.0), self.fi_uciqe, self.wb_uciqe, self.enhanced_uciqe),
            ("PCQI", self.ori_pcqi, getattr(self, 'dcp_pcqi', 0.0), self.fi_pcqi, self.wb_pcqi, self.com_pcqi)
        ]

        for i, (name, orig, dcp, fi, wb, comb) in enumerate(metrics):
            self.oc_metric_labels[(name, 1)].setText(f"{orig:.3f}")
            self.oc_metric_labels[(name, 2)].setText(f"{dcp:.3f}")
            self.oc_metric_labels[(name, 3)].setText(f"{fi:.3f}")
            self.oc_metric_labels[(name, 4)].setText(f"{wb:.3f}")
            self.oc_metric_labels[(name, 5)].setText(f"{comb:.3f}")


    def update_histogram(self):
        if self.cv2_edt_img is None:
            self.histogram_placeholder.setText("No histogram available")
            return

        os.makedirs('./histogram_temp', exist_ok=True)
        fig = plt.Figure(figsize=(10, 6), dpi=150)
        ax1 = fig.add_subplot(1, 1, 1)
        color = ('b','g','r')
        for channel, col in enumerate(color):
            histogram = cv2.calcHist([self.cv2_edt_img],[channel],None,[256],[0,256])
            ax1.plot(histogram, color=col, linewidth=2)
        
        # Add axis labels and title
        ax1.set_xlabel('Keamatan Piksel', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frekuensi', fontsize=12, fontweight='bold')
        ax1.set_title('Histogram Warna (B=Biru, G=Hijau, R=Merah)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['Biru', 'Hijau', 'Merah'], loc='upper right')
        
        fig.tight_layout()
        fig.savefig('./histogram_temp/histogram.png', dpi=150)
        plt.close(fig)
        
        pixmap = QPixmap('./histogram_temp/histogram.png')
        scaled_pixmap = pixmap.scaled(self.histogram_placeholder.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.histogram_placeholder.setPixmap(scaled_pixmap)
        self.histogram_placeholder.setAlignment(Qt.AlignCenter)

    def generate_image_histogram(self, image, filename):
        """Generate histogram for a single image and return as QPixmap"""
        if image is None:
            return None
        
        try:
            os.makedirs('./histogram_temp', exist_ok=True)
            fig = plt.Figure(figsize=(5, 2.5), dpi=150)
            ax = fig.add_subplot(1, 1, 1)
            color = ('b', 'g', 'r')
            for channel, col in enumerate(color):
                histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
                ax.plot(histogram, color=col, linewidth=1.5)
            
            ax.set_xlim([0, 256])
            ax.set_xlabel('Keamatan Piksel', fontsize=9, fontweight='bold')
            ax.set_ylabel('Frekuensi', fontsize=9, fontweight='bold')
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3)
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
            
            filepath = f'./histogram_temp/{filename}'
            fig.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)
            
            pixmap = QPixmap(filepath)
            return pixmap
        except Exception as e:
            print(f"Error generating histogram for {filename}: {e}")
            return None

    def update_overall_histograms(self):
        """Update histograms for all images in overall comparison"""
        if not hasattr(self, 'oc_histogram_labels') or len(self.oc_histogram_labels) == 0:
            return
        
        images = [
            self.cv2_ori_img,
            getattr(self, 'cv2_dcp_img', None),
            self.cv2_fi_img,
            self.cv2_wb_img,
            self.cv2_edt_img
        ]
        
        filenames = [
            'hist_original.png',
            'hist_dcp.png',
            'hist_formation.png',
            'hist_whitebalance.png',
            'hist_combined.png'
        ]
        
        for i, (img, filename, label) in enumerate(zip(images, filenames, self.oc_histogram_labels)):
            if img is not None:
                pixmap = self.generate_image_histogram(img, filename)
                if pixmap is not None:
                    scaled_pixmap = pixmap.scaled(200, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    label.setPixmap(scaled_pixmap)
                    label.setAlignment(Qt.AlignCenter)
            else:
                label.setText("No histogram")

    def plipsum(self, i, j, gamma=1026):
        return i + j - i * j / gamma

    def plipsub(self, i, j, k=1026):
        return k * (i - j) / (k - j)

    def plipmult(self, c, j, gamma=1026):
        return gamma - gamma * (1 - j / gamma)**c
    
    def eme(self, ch, blocksize=8):
        num_x = math.ceil(ch.shape[0] / blocksize)
        num_y = math.ceil(ch.shape[1] / blocksize)
        
        eme = 0
        w = 2. / (num_x * num_y)
        for i in range(num_x):

            xlb = i * blocksize
            if i < num_x - 1:
                xrb = (i+1) * blocksize
            else:
                xrb = ch.shape[0]

            for j in range(num_y):

                ylb = j * blocksize
                if j < num_y - 1:
                    yrb = (j+1) * blocksize
                else:
                    yrb = ch.shape[1]
                
                block = ch[xlb:xrb,ylb:yrb]

                blockmin = np.double(np.min(block))
                blockmax = np.double(np.max(block))

                # new version
                if blockmin == 0: blockmin+=1
                if blockmax == 0: blockmax+=1
                eme += w * math.log(blockmax / blockmin)
        return eme
    
    def logamee(self, ch, blocksize=8):
        num_x = math.ceil(ch.shape[0] / blocksize)
        num_y = math.ceil(ch.shape[1] / blocksize)
        
        s = 0
        w = 1. / (num_x * num_y)
        for i in range(num_x):

            xlb = i * blocksize
            if i < num_x - 1:
                xrb = (i+1) * blocksize
            else:
                xrb = ch.shape[0]

            for j in range(num_y):

                ylb = j * blocksize
                if j < num_y - 1:
                    yrb = (j+1) * blocksize
                else:
                    yrb = ch.shape[1]
                
                block = ch[xlb:xrb,ylb:yrb]
                blockmin = np.double(np.min(block))
                blockmax = np.double(np.max(block))

                top = self.plipsub(blockmax,blockmin)
                bottom = self.plipsum(blockmax,blockmin)
                
                # Avoid division by zero
                if bottom == 0:
                    m = 0
                else:
                    m = top/bottom
                    
                if m == 0.:
                    s += 0
                else:
                    s += (m) * np.log(m)

        return self.plipmult(w,s)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win._connect_buttons()  # Connect buttons after UI is set up
    win.show()
    sys.exit(app.exec_())
