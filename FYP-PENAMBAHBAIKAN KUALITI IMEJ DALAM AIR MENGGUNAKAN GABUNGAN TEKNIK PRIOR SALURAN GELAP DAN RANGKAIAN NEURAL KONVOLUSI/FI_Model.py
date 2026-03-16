# ============================================================================
# FI_Model.py - Formation Image Enhancement CNN Model
# ============================================================================
# Purpose: Enhance underwater image structure and details using Formation Image
#          formation principles with Backscatter Subtraction and Detail Transfer
#
# Architecture: FINet (Formation Image Network)
#   - Two parallel processing branches:
#     1. BsConvBlock: Backscatter Subtraction (estimates background scattering)
#     2. DtConvBlock: Detail Transfer (extracts and enhances details)
#   - Combines both to create enhanced output
#
# Key Characteristics:
#   - Preserves FULL RESOLUTION (no downsampling in this model)
#   - Outputs: Enhanced image with visible details and structure
#   - Input: 3-channel RGB image (0-1 normalized)
#   - Output: 3-channel RGB enhanced image (0-1 normalized)
#
# ============================================================================

from model_blocks import *

import torch
import numpy as np
import torch.nn as nn


class FINet(nn.Module):
    """
    Formation Image Enhancement Network
    
    Architecture:
    - BsConvBlock: Estimates and removes backscatter (background light)
    - DtConvBlock: Extracts and transfers detail enhancement
    
    Process:
    1. Input image passes through BsConvBlock → Backscatter estimate
    2. Detail transfer combines original with backscatter for DtConvBlock
    3. Final output applies detail transfer multiplicatively to backscatter-subtracted image
    4. Clamp to [0, 1] range for valid image values
    """
    
    def __init__(self):
        """Initialize FINet with two processing branches"""
        super(FINet, self).__init__()
        
        # ============================================================
        # Branch 1: BsConvBlock (Backscatter Subtraction)
        # ============================================================
        # Purpose: Estimate and model the backscatter/haze in the image
        # Input: 3-channel RGB image
        # Output: 3-channel backscatter estimate (same spatial size)
        # Role: Separates scene radiance from atmospheric scattering
        self.BsNet = BsConvBlock(3, 3)
        
        # ============================================================
        # Branch 2: DtConvBlock (Detail Transfer)
        # ============================================================
        # Purpose: Extract detail enhancement mapping based on backscatter
        # Input: 6-channel tensor (backscatter estimate + original image)
        # Output: 8-channel intermediate detail features → 3-channel detail transfer
        # Role: Learns how to enhance details given the backscatter information
        self.DtNet = DtConvBlock(3, 8)

    def forward(self, x):
        """
        Forward pass through FINet
        
        Args:
            x: Input tensor of shape (batch, 3, height, width)
               Values in range [0, 1]
        
        Returns:
            Enhanced image tensor of shape (batch, 3, height, width)
            Values clamped to [0, 1] range
        
        Process:
            1. Backscatter Estimation (BSE):
               - Pass input through BsConvBlock
               - Outputs backscatter estimate (same size as input)
               - Represents the haze/scattering component
            
            2. Detail Enhancement (DTE):
               - Concatenate: [BSE, input image] → 6-channel tensor
               - Pass through DtConvBlock
               - Outputs detail transfer weights (8 channels → 3 channels)
               - Represents how much detail enhancement to apply
            
            3. Combine:
               - Input with backscatter removed: (input - BSE)
               - Scale detail transfer: multiply by DTE weights
               - Restore backscatter: add BSE back
               - Formula: ((x - BSE) * DTE) + BSE
               - This preserves natural appearance while enhancing details
            
            4. Normalize:
               - Clamp output to [0, 1] range
               - Ensures valid image values
        """
        
        # Step 1: Estimate backscatter component
        BSE = self.BsNet(x)
        
        # Step 2: Prepare input for detail transfer network
        # Concatenate: backscatter estimate with original image
        # Creates 6-channel tensor: [BSE, original_x, original_y, original_z]
        DTE = self.DtNet(torch.cat((x*0+BSE, x), 1))
        
        # Step 3: Apply detail enhancement
        # Formula: Enhanced = ((Original - Backscatter) * DetailTransfer) + Backscatter
        # This performs scattering removal and detail-aware enhancement
        out = ((x - BSE) * DTE + BSE)
        
        # Step 4: Normalize output to valid image range [0, 1]
        return torch.clamp(out, 0., 1.)


# ============================================================================
# Component Explanation:
# ============================================================================
# BsConvBlock (Backscatter Subtraction):
#   - Estimates atmospheric scattering component
#   - Uses adaptive pooling to capture global backscatter
#   - Output helps identify what needs to be removed
#
# DtConvBlock (Detail Transfer):
#   - Applies dilated convolutions for multi-scale detail extraction
#   - Learns how backscatter affects detail visibility
#   - Outputs multiplicative enhancement weights
#
# Combined Processing:
#   - Subtracts estimated backscatter from original
#   - Applies learned detail enhancement
#   - Restores controlled backscatter for natural appearance
#   - Result: Clearer image with enhanced visible details
#
# ============================================================================
