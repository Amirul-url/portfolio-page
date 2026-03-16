# ============================================================================
# model_blocks.py - Reusable Neural Network Building Blocks
# ============================================================================
# Purpose: Define modular building blocks used in FI_Model and WB_Model
#
# Contains:
#   - BsConvBlock: Backscatter Subtraction convolution block
#   - DtConvBlock: Detail Transfer dilated convolution block
#   - Standard UNet components for encoder-decoder architectures
#
# ============================================================================

import torch
import torch.nn as nn

# ============================================================================
# FORMATION IMAGE (FI) SPECIFIC BLOCKS
# ============================================================================

class BsConvBlock(nn.Module):
    """
    Backscatter Subtraction Convolution Block
    
    Purpose: Estimate and model the backscatter/atmospheric scattering component
    
    Architecture:
    1. Two 3×3 convolutions with PReLU (extract basic features)
    2. AdaptiveAvgPool2d (capture global backscatter information)
    3. Two 1×1 convolutions with PReLU (refine backscatter estimate)
    
    Process Flow:
    Input (3-channel) → Conv3×3 → PReLU → Conv3×3 → PReLU
                      → AdaptiveAvgPool → Conv1×1 → PReLU → Conv1×1 → PReLU
                      → Output (3-channel)
    
    Key Features:
    - Adaptive average pooling: Reduces spatial dimensions, captures global statistics
    - PReLU activation: Allows negative values (better for residual learning)
    - Output represents the backscatter component to be removed
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize BsConvBlock
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            out_channels: Number of output channels (typically 3 for backscatter estimate)
        """
        super().__init__()
        self.BsConv_Block = nn.Sequential(
            # Step 1: Extract local features with 3×3 convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(),  # Parametric ReLU: f(x) = max(ax, x), allows negative values
            
            # Step 2: Further refine local features
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            
            # Step 3: Capture global backscatter statistics
            nn.AdaptiveAvgPool2d((1, 1)),  # Pool to 1×1 spatial dimensions
            
            # Step 4: Process global information to refine estimate
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 1×1 conv for channel-wise processing
            nn.PReLU(),
            
            # Step 5: Final refinement
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, in_channels, height, width)
        Returns:
            Backscatter estimate (batch, out_channels, height, width)
        """
        return self.BsConv_Block(x)


class DtConvBlock(nn.Module):
    """
    Detail Transfer Dilated Convolution Block
    
    Purpose: Extract detail enhancement information using multi-scale dilated convolutions
    
    Architecture:
    - Multi-scale dilated convolutions (dilation rates: 1, 2, 5)
    - Progressively extracts details at different scales
    
    Process Flow:
    Input (6-channel: backscatter + original)
    → Conv3×3 (dilation=1, receptive field=3) → PReLU
    → Conv3×3 (dilation=2, receptive field=7) → PReLU
    → Conv3×3 (dilation=5, receptive field=11) → PReLU
    → Conv3×3 (dilation=1, receptive field=3) → PReLU
    → Output (in_channels: 3-channel)
    
    Key Features:
    - Dilated Convolutions: Increase receptive field without downsampling
    - Multi-scale: Captures details at different semantic levels
    - Progressive dilation: Gradually increases context awareness
    - Maintains spatial resolution (critical for preserving details)
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize DtConvBlock
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            out_channels: Number of output channels (typically 8 for internal features)
        """
        super().__init__()
        self.DtConv_Block = nn.Sequential(
            # Step 1: Multi-scale detail extraction - Local scale (receptive field ~3)
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.PReLU(),
            
            # Step 2: Multi-scale detail extraction - Medium scale (receptive field ~7)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            
            # Step 3: Multi-scale detail extraction - Large scale (receptive field ~11)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5),
            nn.PReLU(),
            
            # Step 4: Combine and refine multi-scale information
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, in_channels*2, height, width)
               Concatenated [backscatter_estimate, original_image]
        Returns:
            Detail transfer weights (batch, in_channels, height, width)
        """
        return self.DtConv_Block(x)


# ============================================================================
# WHITE BALANCE (WB) / UNET SPECIFIC BLOCKS
# ============================================================================

class DoubleConvBlock(nn.Module):
    """
    Double Convolution Block
    
    Purpose: Extract features using two consecutive 3×3 convolutions
    
    Used in: UNet encoder/decoder paths
    
    Process:
    Input → Conv3×3 → ReLU → Conv3×3 → ReLU → Output
    
    Standard building block for feature extraction in segmentation networks
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling Block (Encoder)
    
    Purpose: Reduce spatial dimensions while extracting features
    
    Process:
    Input → MaxPool2d (2×2) → DoubleConv → Output
    
    Used in: UNet encoder to progressively downsample and extract features
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Reduce spatial size by 2× (e.g., 512→256)
            DoubleConvBlock(in_channels, out_channels)  # Extract features at lower resolution
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeDown(nn.Module):
    """
    Bridge Downsampling Block (Bottleneck Encoder)
    
    Purpose: Final downsampling step connecting encoder to bottleneck
    
    Process:
    Input → MaxPool2d (2×2) → Conv3×3 → ReLU → Output
    
    Simpler than DownBlock (single conv instead of double) for efficient bottleneck processing
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeUP(nn.Module):
    """
    Bridge Upsampling Block (Bottleneck Decoder)
    
    Purpose: Initial upsampling step from bottleneck to decoder
    
    Process:
    Input → Conv3×3 → ReLU → TransposeConv (2×2 stride) → Output
    
    Expands spatial dimensions (e.g., 64→128) for decoder path
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # 2× upsampling
        )

    def forward(self, x):
        return self.conv_up(x)


class UpBlock(nn.Module):
    """
    Upsampling Block (Decoder) with Skip Connection
    
    Purpose: Restore spatial dimensions and combine with corresponding encoder features
    
    Process:
    Encoder path (x2) + Decoder path (x1) 
    → Concatenate [x2, x1] (double channels)
    → DoubleConv → ReLU
    → TransposeConv (2×2 stride) for 2× upsampling
    → Output
    
    Key Features:
    - Skip connections: Preserves fine-grained features from encoder
    - Concatenation: Combines low-level and high-level features
    - Upsampling: Restores spatial resolution toward original
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)  # Process concatenated features
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        """
        Args:
            x1: Features from decoder path (lower resolution)
            x2: Features from encoder path (higher resolution, same spatial size after upsampling)
        Returns:
            Upsampled and fused features
        """
        x = torch.cat([x2, x1], dim=1)  # Concatenate along channel dimension
        x = self.conv(x)  # Process concatenated features
        return torch.relu(self.up(x))  # Upsample and apply activation


class OutputBlock(nn.Module):
    """
    Final Output Block (Output Layer)
    
    Purpose: Combine final features and generate output
    
    Process:
    Encoder output + Decoder output
    → Concatenate [encoder, decoder]
    → DoubleConv (combine features)
    → Conv1×1 (generate output channels)
    → Output
    
    Used in: Final layer of UNet to produce color-corrected image
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),  # Combine encoder/decoder features
            nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Generate output (typically 3 for RGB)
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: Features from decoder final layer
            x2: Features from encoder first layer (skip connection)
        Returns:
            Output image (same spatial size as input)
        """
        x = torch.cat([x2, x1], dim=1)  # Concatenate along channel dimension
        return self.out_conv(x)  # Generate final output


# ============================================================================
# COMPONENT SUMMARY:
# ============================================================================
# FI-Specific:
#   - BsConvBlock: Estimates backscatter for scattering removal
#   - DtConvBlock: Extracts multi-scale detail enhancement
#
# UNet Components:
#   - DoubleConvBlock: Basic feature extraction
#   - DownBlock: Encoder downsampling
#   - UpBlock: Decoder upsampling with skip connections
#   - BridgeDown/UP: Bottleneck transition layers
#   - OutputBlock: Final output generation
#
# ============================================================================

    
class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_up(x)

class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))
    
class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)