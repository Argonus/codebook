import os
import tensorflow as tf
import gc
import time

# Configure TensorFlow to use memory growth to prevent allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled on {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPUs detected, running on CPU")

# Configure memory usage limits
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )

# Limit TensorFlow memory usage
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress some TensorFlow warnings

# Import DataLoader after TensorFlow configuration
from src.model.tensorflow_data_loader import DataLoader
from src.utils.consts import TF_BUFFER_SIZE

# Define smaller batch size for efficient processing
BATCH_SIZE = 16

def main():
    print("Starting dataset loading with memory optimizations...")
    start_time = time.time()
    
    try:
        # Create DataLoader with reduced buffer size
        loader = DataLoader()
        loader.get_datasets()
    except Exception as e:
        print(f"Error processing datasets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()