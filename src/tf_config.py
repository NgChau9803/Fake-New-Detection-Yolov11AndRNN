import os
import tensorflow as tf

def configure_tensorflow():
    """Configure TensorFlow with specific memory settings"""
    # Limit GPU memory growth if GPUs are available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPUs detected, using CPU")
    
    # Set environment variables for CPU configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    
    # Configure thread settings
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    print("TensorFlow configured with memory settings") 