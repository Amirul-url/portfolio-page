# ============================================================================
# WB_Model.py - White Balance Color Correction CNN Model
# ============================================================================
# Purpose: Correct color cast and balance color channels in underwater images
#          using a U-Net encoder-decoder architecture with 384 bottleneck
#
# Architecture: WBNet (White Balance Network)
#   - Encoder: Progressive downsampling to extract color correction features
#   - Bottleneck: 384-channel bottleneck for high-capacity feature processing
#   - Decoder: Progressive upsampling with skip connections to restore resolution
#   - Strategy: Learn color mapping from FI-enhanced images
#
# Key Characteristics:
#   - Downsamples input to 656px (2^4 aligned for UNet compatibility)
#   - Uses skip connections to preserve fine details
#   - Outputs: 3-channel color-corrected image
#   - Input: 3-channel RGB image (0-1 normalized)
#   - Output: 3-channel RGB color-corrected image (0-1 normalized)
#
# ============================================================================

from model_blocks import *

import torch
import numpy as np
import torch.nn as nn


class WBNet(nn.Module):
    """
    White Balance Correction Network (U-Net Architecture)
    
    Purpose: Learn and apply color correction to underwater images
    
    Architecture Breakdown:
    ┌─────────────────────────────────────────────────────────────────┐
    │ ENCODER (Extract Color Correction Features)                    │
    ├─────────────────────────────────────────────────────────────────┤
    │ Input (3 channels) ─→ IncConv (→24) ─→ Down1 (→48)            │
    │                      ↓                   ↓                      │
    │                    Down2 (→96)  ←────────┘                      │
    │                      ↓                                          │
    │                    Down3 (→192) ─→ BridgeDown (→384)           │
    │                                  [BOTTLENECK]                  │
    ├─────────────────────────────────────────────────────────────────┤
    │ DECODER (Apply Color Correction with Skip Connections)        │
    ├─────────────────────────────────────────────────────────────────┤
    │ BridgeUP (→192) ─→ UpBlock1 + skip(x4, 192) ─→ (→96)           │
    │                                   ↓                            │
    │                           UpBlock2 + skip(x3, 96) ─→ (→48)    │
    │                                   ↓                            │
    │                           UpBlock3 + skip(x2, 48) ─→ (→24)    │
    │                                   ↓                            │
    │                         OutputBlock + skip(x1, 24) ─→ (→3)    │
    │                                   ↓                            │
    │                          Output: Color-corrected image         │
    └─────────────────────────────────────────────────────────────────┘
    
    Channel Progression:
    - Input: 3 → 24 → 48 → 96 → 192 → [384 bottleneck] → 192 → 96 → 48 → 24 → 3
    """
    
    def __init__(self):
        """
        Initialize WBNet U-Net architecture
        
        Structure:
        - Encoder: 5 layers (IncConv + 3 DownBlocks + BridgeDown)
        - Bottleneck: 384-channel feature space (high-capacity)
        - Decoder: 5 layers (BridgeUP + 3 UpBlocks + OutputBlock)
        - Skip Connections: All decoder layers receive encoder features
        """
        super(WBNet, self).__init__()
        
        # ============================================================
        # ENCODER PATH (Feature Extraction & Downsampling)
        # ============================================================
        # Purpose: Extract color correction patterns at multiple scales
        
        # Layer 0: Initial Feature Extraction (No downsampling)
        # Input: 3 channels (RGB) → Output: 24 channels
        # Process: Extract basic color features at full resolution
        self.encoder_inc = DoubleConvBlock(3, 24)
        
        # Layer 1: First Downsampling (Resolution: 1/2)
        # Input: 24 channels → Output: 48 channels
        # Process: Downsample 2× and extract color patterns at medium scale
        self.encoder_down1 = DownBlock(24, 48)
        
        # Layer 2: Second Downsampling (Resolution: 1/4)
        # Input: 48 channels → Output: 96 channels
        # Process: Downsample 2× and extract larger-scale color patterns
        self.encoder_down2 = DownBlock(48, 96)
        
        # Layer 3: Third Downsampling (Resolution: 1/8)
        # Input: 96 channels → Output: 192 channels
        # Process: Downsample 2× and extract global color information
        self.encoder_down3 = DownBlock(96, 192)
        
        # Bottleneck: Bridge Downsampling (Resolution: 1/16)
        # Input: 192 channels → Output: 384 channels (HIGH CAPACITY)
        # Purpose: Final compression before decoder
        # Why 384 channels: Large feature space to learn complex color mappings
        # The 384-channel bottleneck allows model to capture subtle color corrections
        self.encoder_bridge_down = BridgeDown(192, 384)
        
        # ============================================================
        # DECODER PATH (Color Correction & Upsampling)
        # ============================================================
        # Purpose: Reconstruct full-resolution color-corrected image
        # Strategy: Progressively upsample while combining with encoder features
        
        # Bottleneck Bridge Upsampling (Resolution: 1/8)
        # Input: 384 channels (bottleneck) → Output: 192 channels
        # Process: Start decoder path with upsampling from bottleneck
        self.awb_decoder_bridge_up = BridgeUP(384, 192)
        
        # Decoder Layer 1 (Resolution: 1/4)
        # Input: 192 from decoder + 192 from skip(x4) → Output: 96 channels
        # Process: Combine bottleneck features with encoder layer 3 features
        # Skip connection (x4): Brings fine-grained color details from encoder
        self.awb_decoder_up1 = UpBlock(192, 96)
        
        # Decoder Layer 2 (Resolution: 1/2)
        # Input: 96 from decoder + 96 from skip(x3) → Output: 48 channels
        # Process: Combine upsampled features with encoder layer 2 features
        # Skip connection (x3): Adds medium-scale color correction patterns
        self.awb_decoder_up2 = UpBlock(96, 48)
        
        # Decoder Layer 3 (Resolution: 1/1 - Full)
        # Input: 48 from decoder + 48 from skip(x2) → Output: 24 channels
        # Process: Combine upsampled features with encoder layer 1 features
        # Skip connection (x2): Adds detailed color features at full scale
        self.awb_decoder_up3 = UpBlock(48, 24)
        
        # Final Output Layer (Resolution: Full - Same as input)
        # Input: 24 from decoder + 24 from skip(x1) → Output: 3 channels (RGB)
        # Purpose: Generate final color-corrected image
        # Skip connection (x1): Combines final features with initial extraction
        self.awb_decoder_out = OutputBlock(24, 3)

    def forward(self, x):
        """
        Forward pass through WBNet for color correction
        
        Args:
            x: Input tensor of shape (batch, 3, height, width)
               Values in range [0, 1]
               Note: Input should be pre-resized to 656px for this model
        
        Returns:
            Color-corrected image tensor (batch, 3, height, width)
            Values in range [0, 1]
        
        Process Flow:
        
        ENCODING PHASE (Extract Color Features):
        ────────────────────────────────────────
        x (input) ──↓
        x1 = IncConv(x)              [Full res,   24 channels]
          ──↓
        x2 = Down1(x1)               [1/2 res,    48 channels]
          ──↓
        x3 = Down2(x2)               [1/4 res,    96 channels]
          ──↓
        x4 = Down3(x3)               [1/8 res,   192 channels]
          ──↓
        x5 = BridgeDown(x4)          [1/16 res,  384 channels] ← BOTTLENECK
        
        DECODING PHASE (Apply Color Correction with Skip Connections):
        ──────────────────────────────────────────────────────────────
        x5 (384 ch) ──↓
        x_awb = BridgeUP(x5)         [1/8 res,   192 channels]
                  ↓ + skip from x4 (encoder features at 1/8 res)
        x_awb = UpBlock1(x_awb, x4)  [1/4 res,    96 channels]
                  ↓ + skip from x3 (encoder features at 1/4 res)
        x_awb = UpBlock2(x_awb, x3)  [1/2 res,    48 channels]
                  ↓ + skip from x2 (encoder features at 1/2 res)
        x_awb = UpBlock3(x_awb, x2)  [Full res,   24 channels]
                  ↓ + skip from x1 (initial features at full res)
        awb = OutputBlock(x_awb, x1) [Full res,    3 channels] ← OUTPUT
        
        SKIP CONNECTIONS ROLE:
        ──────────────────────
        Why skip connections matter for color correction:
        1. x4 skip: Brings encoder's 1/8 resolution features (global color info)
        2. x3 skip: Brings encoder's 1/4 resolution features (medium-scale colors)
        3. x2 skip: Brings encoder's 1/2 resolution features (detailed colors)
        4. x1 skip: Brings encoder's initial features (fine-grained RGB patterns)
        
        Together they enable the decoder to:
        - Preserve fine color details (from higher resolution skips)
        - Apply global color correction (from lower resolution features)
        - Maintain structural integrity while correcting colors
        
        WHAT THIS ACHIEVES:
        ───────────────────
        - Input: FI-enhanced image with potential color cast (e.g., too blue/green)
        - Process: Learn color mapping through 384-channel bottleneck
        - Output: Color-corrected version with balanced RGB channels
        - Result: Image ready for final pipeline combination
        """
        
        # ============================================================
        # ENCODING: Extract color correction features
        # ============================================================
        
        # Layer 0: Initial feature extraction at full resolution
        x1 = self.encoder_inc(x)  # 3ch → 24ch, preserve spatial dims
        
        # Layer 1: Downsample 2×, extract 48-channel features
        x2 = self.encoder_down1(x1)  # 24ch → 48ch, spatial: 1/2
        
        # Layer 2: Downsample 2×, extract 96-channel features
        x3 = self.encoder_down2(x2)  # 48ch → 96ch, spatial: 1/4
        
        # Layer 3: Downsample 2×, extract 192-channel features
        x4 = self.encoder_down3(x3)  # 96ch → 192ch, spatial: 1/8
        
        # Bottleneck: Final downsampling to 384-channel feature space
        x5 = self.encoder_bridge_down(x4)  # 192ch → 384ch, spatial: 1/16
        
        # ============================================================
        # DECODING: Reconstruct color-corrected image with skip connections
        # ============================================================
        
        # Bridge upsampling from bottleneck
        x_awb = self.awb_decoder_bridge_up(x5)  # 384ch → 192ch, spatial: 1/8
        
        # Decoder Layer 1: Upsample + skip from encoder layer 3
        # Combines: upsampled features + encoder features at 1/8 resolution
        x_awb = self.awb_decoder_up1(x_awb, x4)  # [192ch+192ch] → 96ch, spatial: 1/4
        
        # Decoder Layer 2: Upsample + skip from encoder layer 2
        # Combines: upsampled features + encoder features at 1/4 resolution
        x_awb = self.awb_decoder_up2(x_awb, x3)  # [96ch+96ch] → 48ch, spatial: 1/2
        
        # Decoder Layer 3: Upsample + skip from encoder layer 1
        # Combines: upsampled features + encoder features at 1/2 resolution
        x_awb = self.awb_decoder_up3(x_awb, x2)  # [48ch+48ch] → 24ch, spatial: full
        
        # Final Output: Combine with initial encoder features
        # Combines: final upsampled features + initial extraction at full res
        awb = self.awb_decoder_out(x_awb, x1)  # [24ch+24ch] → 3ch (RGB output)
        
        # Return color-corrected image
        return awb


# ============================================================================
# ARCHITECTURE SUMMARY
# ============================================================================
# U-Net Structure:
#   - Symmetric encoder-decoder with skip connections
#   - 384-channel bottleneck for high-capacity color learning
#   - 4 levels of resolution: full → 1/2 → 1/4 → 1/8 → 1/16 (bottleneck)
#
# Purpose in Pipeline:
#   1. Takes FI-enhanced image as input (downsampled to 656px)
#   2. Learns color correction patterns in 384-dim feature space
#   3. Outputs color-corrected mapping applied back to full-resolution FI
#
# Channel Flow:
#   Encoder:  3 → 24 → 48 → 96 → 192 → [384 bottleneck]
#   Decoder: 192 → 96 → 48 → 24 → 3
#
# Skip Connections:
#   - Level 3→1: 192ch + 192ch
#   - Level 2→2: 96ch + 96ch
#   - Level 1→3: 48ch + 48ch
#   - Level 0→4: 24ch + 24ch
#
# Why 384 Bottleneck:
#   - Larger feature space captures subtle color mappings
#   - Balances between model capacity and computational efficiency
#   - Sufficient for learning complex underwater color corrections
#
# ============================================================================
 