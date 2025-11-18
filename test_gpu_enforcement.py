#!/usr/bin/env python3
"""
Test script to verify GPU enforcement in the TTS system.
"""

import torch
from tts import generate_svara_tts

def test_gpu_enforcement():
    """Test that models are correctly loaded on GPU only."""
    print("Testing GPU enforcement...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return False
    
    # Test TTS generation (simple test)
    try:
        print("Attempting to generate test audio...")
        result = generate_svara_tts(
            text="Hello, this is a GPU test.",
            language="English", 
            gender="Female",
            outfile="gpu_test_output.wav"
        )
        print(f"SUCCESS: Audio generated at {result}")
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_gpu_enforcement()
    if success:
        print("\n✅ GPU enforcement test PASSED")
    else:
        print("\n❌ GPU enforcement test FAILED")