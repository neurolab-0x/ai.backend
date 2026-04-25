"""
Generate a simple test audio file for voice API testing
"""

import sys
import numpy as np
import os

# Try to import scipy, if not available use wave module
try:
    from scipy.io import wavfile
    USE_SCIPY = True
except ImportError:
    import wave
    import struct
    USE_SCIPY = False
    print("Note: scipy not available, using wave module")

def generate_test_audio(filename="data/testing_data/test_audio.wav", duration=2.0, sample_rate=16000):
    """
    Generate a simple test audio file with a sine wave
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a combination of frequencies (more interesting than single tone)
    # This creates a more "voice-like" signal
    freq1 = 440  # A4 note
    freq2 = 554  # C#5 note
    freq3 = 659  # E5 note
    
    # Create signal with multiple harmonics
    signal = (
        0.3 * np.sin(2 * np.pi * freq1 * t) +
        0.2 * np.sin(2 * np.pi * freq2 * t) +
        0.1 * np.sin(2 * np.pi * freq3 * t)
    )
    
    # Add some amplitude modulation to make it more dynamic
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    signal = signal * envelope
    
    # Normalize to prevent clipping
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Convert to int16
    audio_int16 = (signal * 32767).astype(np.int16)
    
    # Save as WAV file
    if USE_SCIPY:
        wavfile.write(filename, sample_rate, audio_int16)
    else:
        # Use wave module
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    print(f"✓ Generated test audio file: {filename}")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  File size: {os.path.getsize(filename)} bytes")
    
    return filename

if __name__ == "__main__":
    # Generate test audio files
    print("Generating test audio files...\n")
    
    # Single test file
    generate_test_audio("data/testing_data/test_audio.wav", duration=2.0)
    
    # Multiple files for batch testing
    print("\nGenerating batch test files...")
    for i in range(3):
        generate_test_audio(f"data/testing_data/test_audio_{i+1}.wav", duration=1.5)
    
    print("\n✓ All test audio files generated successfully!")
    print("\nYou can now test the voice API with:")
    print("  python test_voice_api.py")
