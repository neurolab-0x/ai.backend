import sys
import os
import numpy as np
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.realtime import process_streaming_chunk, StreamBuffer
# safely import config
try:
    from src.config.settings import REAL_TIME_CONFIG
except ImportError:
    REAL_TIME_CONFIG = {'buffer_size_seconds': 5}

def simulate_streaming_session(duration_seconds=5, sampling_rate=256, channels=16):
    """
    Simulates a streaming session by generating random EEG data
    and processing it through the realtime pipeline.
    """
    print(f"Starting simulation: {duration_seconds}s @ {sampling_rate}Hz, {channels} channels")
    
    # Initialize buffer
    stream_buffer = StreamBuffer(max_size=REAL_TIME_CONFIG.get('buffer_size_seconds', 5) * sampling_rate)
    
    # Simulate chunks (e.g., 0.1s chunks)
    chunk_size = int(sampling_rate * 0.1)
    total_chunks = int((duration_seconds * sampling_rate) / chunk_size)
    
    print(f"Chunk size: {chunk_size} samples. Total chunks: {total_chunks}")
    
    start_time = time.time()

    model_type = "enhanced_cnn_lstm"

    model_path = f"model/{model_type}.h5"
    
    for i in range(total_chunks):
        # Generate mock data: (samples, channels) usually, but check endpoint expectation
        # api/realtime.py logic handles (channels, samples) or (samples, channels).
        # Let's generate (samples, channels) as typical for raw reading
        # Adding some sine wave pattern to mimic signal
        t = np.linspace(i*0.1, (i+1)*0.1, chunk_size, endpoint=False)
        signal = np.sin(2 * np.pi * 10 * t) # 10Hz alpha
        noise = np.random.normal(0, 0.5, (chunk_size, channels))
        chunk_data = (noise + signal[:, np.newaxis]).T # (channels, samples)
        
        # Process
        try:
            result = process_streaming_chunk(
                chunk_data,
                model_path=model_path, # No model loaded for simulation, will use dummy logic
                clean_artifacts=False, # Skip cleaning for speed/mock
                stream_buffer=stream_buffer
            )
            
            # Print status every 5 chunks
            if i % 5 == 0:
                print(f"Chunk {i+1}/{total_chunks}: "
                      f"State={result.get('dominant_state')}, "
                      f"Conf={result.get('confidence'):.2f}, "
                      f"Buffered={stream_buffer.buffer.shape if stream_buffer.buffer is not None else 0}")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            break
            
        # Simulate real-time delay (optional, commented out for speed)
        # time.sleep(0.1) 
        
    total_time = time.time() - start_time
    print(f"\nSimulation complete in {total_time:.2f}s")
    if 'result' in locals():
        print("Final Result Sample:", json.dumps(result, indent=2))

if __name__ == "__main__":
    simulate_streaming_session()
