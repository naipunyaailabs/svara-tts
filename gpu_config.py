#!/usr/bin/env python3
"""
Configuration module for device selection in the TTS system.
"""

import torch

# Configuration flag to enforce GPU usage
ENFORCE_GPU = True

def get_device():
    """
    Get the appropriate device based on availability and configuration.
    
    Returns:
        torch.device: The selected device
        
    Raises:
        RuntimeError: If GPU is enforced but not available
    """
    if ENFORCE_GPU:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a GPU to run.")
        return torch.device("cuda")
    else:
        # Original flexible device selection
        return torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

# Export the device
try:
    device = get_device()
    print(f"Using device: {device}")
except RuntimeError as e:
    print(f"Error: {e}")
    device = None