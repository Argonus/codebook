import tensorflow as tf

def configure_gpu():
    """Configure GPU settings for optimal performance and memory usage"""
    try:
        # Get GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            print("No GPU devices found!")
            return
            
        for device in physical_devices:
            # Enable memory growth to prevent TF from allocating all memory at once
            tf.config.experimental.set_memory_growth(device, True)
            
            # Set memory limit to 24GB for RTX 3090 Ti (which has 24GB VRAM)
            GB = 1024 * 1024 * 1024
            memory_limit = int(18 * GB)  # Leave some headroom
            
            # Configure virtual devices with memory limit
            config = tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=memory_limit
            )
            tf.config.experimental.set_virtual_device_configuration(device, [config])
            
            # Configure memory allocator
            tf.config.experimental.enable_tensor_float_32_execution(True)
            
            # Set GPU options for better memory handling
            tf.config.experimental.set_synchronous_execution(True)
            
            print("\nGPU configuration successful!")
            print(f"Device: {device.name}")
            print(f"Memory limit: {memory_limit / GB:.1f} GB")
            
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def get_optimal_thread_count():
    """Calculate optimal thread counts based on CPU cores
    
    Returns:
        tuple: (inter_op_threads, intra_op_threads, cpu_count)
    """
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    inter_op_threads = max(1, int(cpu_count * 0.3))    
    intra_op_threads = max(1, int(cpu_count * 0.5))
    
    return inter_op_threads, intra_op_threads, cpu_count

def optimize_tensorflow():
    """Apply TensorFlow optimizations for better performance and resource usage."""
    tf.keras.backend.clear_session()
    try:
        inter_threads, intra_threads, cpu_count = get_optimal_thread_count()

        # Optimize CPU threads
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.optimizer.set_jit('autoclustering')
        tf.config.experimental.set_synchronous_execution(True)
        
        # Get current TF configuration status
        xla_status = 'enabled (autoclustering)' if tf.config.optimizer.get_jit() else 'disabled'
        sync_status = 'enabled' if tf.config.experimental.get_synchronous_execution() else 'disabled'
        
        print("\nTensorFlow optimizations status:")
        print(f"- CPU Threads: {intra_threads} intra-op, {inter_threads} inter-op ({cpu_count} cores total)")
        print(f"- XLA JIT: {xla_status}")
        print(f"- Synchronous Execution: {sync_status}")
        
        # Print GPU device details
        print("\nGPU Device Details:")
        for device in tf.config.list_physical_devices('GPU'):
            print(f"- Device: {device.device_type} {device.name}")
            
            # Get memory info for logical device
            logical_devices = tf.config.list_logical_devices('GPU')
            if logical_devices:
                config = tf.config.get_logical_device_configuration(device)
                if config and config[0].memory_limit:
                    print(f"- Memory limit: {config[0].memory_limit / (1024**3):.1f} GB")
                else:
                    print("- Memory limit: Dynamic (growth enabled)")
        
    except Exception as e:
        print(f"Error applying TensorFlow optimizations: {e}")
