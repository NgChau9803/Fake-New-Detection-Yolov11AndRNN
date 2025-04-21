import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import tensorflow as tf
import keras
from io import BytesIO
import cv2
import traceback
import hashlib
from typing import Tuple, Dict, List, Any, Union

def preprocess_image(image_path, target_size=(224, 224), normalize=True, 
                    add_preprocessing=True, grayscale=False, noise_removal=False):
    """
    Preprocess image for model input with advanced techniques and robust error handling
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size for resizing
        normalize (bool): Whether to normalize pixel values to [0,1]
        add_preprocessing (bool): Whether to apply additional preprocessing (model-specific)
        grayscale (bool): Whether to convert image to grayscale
        noise_removal (bool): Whether to apply noise removal techniques
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        # Try to open the image
        try:
            img = Image.open(image_path)
            
            # Check if image is corrupted
            try:
                img.verify()
                # Need to reopen after verify
                img = Image.open(image_path)
            except Exception as e:
                print(f"Warning: Image {image_path} is corrupted: {e}")
                return np.zeros((*target_size, 3 if not grayscale else 1))
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return np.zeros((*target_size, 3 if not grayscale else 1))
        
        # Convert to grayscale if specified
        if grayscale and img.mode != 'L':
            img = img.convert('L')
            # Convert back to RGB but with grayscale content
            if not isinstance(target_size, tuple) or len(target_size) < 3 or target_size[2] == 3:
                img = img.convert('RGB')
        # Otherwise ensure RGB
        elif not grayscale and img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply noise removal if specified
        if noise_removal:
            img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # Resize with proper aspect ratio
        img.thumbnail((max(target_size), max(target_size)), Image.LANCZOS)
        
        # Create new image with the target size and paste the resized image in the center
        new_img = Image.new(img.mode, target_size[:2], (0, 0, 0))
        paste_x = (target_size[0] - img.width) // 2
        paste_y = (target_size[1] - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        # Convert to numpy array
        img_array = np.array(new_img)
        
        # Normalize if specified
        if normalize:
            img_array = img_array / 255.0
        
        # Apply model-specific preprocessing if needed
        if add_preprocessing:
            try:
                if "resnet" in keras.applications.__dict__:
                    if normalize:  # Only if we normalized to [0,1]
                        # Convert from [0,1] to ResNet expected range
                        img_array = keras.applications.resnet.preprocess_input(img_array * 255.0)
                    else:
                        img_array = keras.applications.resnet.preprocess_input(img_array)
            except Exception as e:
                print(f"Warning: Model-specific preprocessing failed: {e}")
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        traceback.print_exc()
        return np.zeros((*target_size, 3 if not grayscale else 1))

def extract_image_features(image_array, method='basic', model=None):
    """
    Extract features from image for analysis

    Args:
        image_array (numpy.ndarray): Image array
        method (str): Feature extraction method 
            ('basic', 'histogram', 'hog', 'cnn', 'yolo')
        model (keras.Model): Optional pre-trained model for feature extraction
        
    Returns:
        dict: Dictionary of image features
    """
    if image_array is None or not isinstance(image_array, np.ndarray) or image_array.size == 0:
        return {
            'width': 0,
            'height': 0,
            'channels': 0,
            'aspect_ratio': 0,
            'mean_pixel_value': 0,
            'std_pixel_value': 0,
            'brightness': 0,
            'contrast': 0
        }
    
    try:
        # Basic features
        height, width = image_array.shape[:2]
        channels = 1 if len(image_array.shape) < 3 else image_array.shape[2]
        aspect_ratio = width / height if height > 0 else 0
        
        # Pixel statistics
        mean_pixel = np.mean(image_array)
        std_pixel = np.std(image_array)
        
        features = {
            'width': width,
            'height': height,
            'channels': channels,
            'aspect_ratio': float(aspect_ratio),
            'mean_pixel_value': float(mean_pixel),
            'std_pixel_value': float(std_pixel),
            'brightness': float(mean_pixel),
            'contrast': float(std_pixel)
        }
        
        # Advanced feature extraction based on method
        if method == 'histogram':
            try:
                # Calculate histogram features
                hist_features = []
                if channels == 1 or len(image_array.shape) < 3:
                    hist = cv2.calcHist([image_array], [0], None, [32], [0, 1 if np.max(image_array) <= 1 else 255])
                    hist_features = hist.flatten().tolist()
                else:
                    for i in range(min(3, channels)):
                        hist = cv2.calcHist([image_array], [i], None, [32], [0, 1 if np.max(image_array) <= 1 else 255])
                        hist_features.extend(hist.flatten().tolist())
                
                features['histogram'] = hist_features[:10]  # Truncate for brevity
            except Exception as e:
                print(f"Error calculating histogram features: {e}")
        
        elif method == 'hog':
            try:
                # Convert to 8-bit image if needed
                img_for_hog = image_array
                if np.max(image_array) <= 1.0:
                    img_for_hog = (image_array * 255).astype(np.uint8)
                
                # Convert to grayscale if needed
                if len(img_for_hog.shape) == 3 and img_for_hog.shape[2] > 1:
                    img_for_hog = cv2.cvtColor(img_for_hog, cv2.COLOR_RGB2GRAY)
                
                # Calculate HOG features
                hog_features = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(
                    cv2.resize(img_for_hog, (64, 64))
                )
                features['hog_features'] = hog_features.flatten()[:10].tolist()  # Truncate for brevity
            except Exception as e:
                print(f"Error calculating HOG features: {e}")
                
        elif method == 'cnn' and model is not None:
            try:
                # Use pre-trained CNN for feature extraction
                # Prepare image for the model
                img_for_cnn = np.expand_dims(image_array, axis=0)
                
                # Extract features
                cnn_features = model.predict(img_for_cnn)
                features['cnn_features'] = cnn_features.flatten()[:10].tolist()  # Truncate for brevity
            except Exception as e:
                print(f"Error extracting CNN features: {e}")
                
        elif method == 'yolo':
            try:
                # Use YOLOv3/v4/v5 for object detection
                from keras.applications import ResNet50
                
                # Create a feature extractor model if none provided
                feature_extractor = model
                if feature_extractor is None:
                    # Use ResNet50 as a substitute for YOLOv11
                    # In a real implementation, you would use the actual YOLOv11
                    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                    feature_extractor = keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
                
                # Prepare image for the model (assuming image is already preprocessed)
                img_tensor = np.expand_dims(image_array, axis=0)
                
                # Extract features
                yolo_features = feature_extractor.predict(img_tensor)
                
                # Calculate feature statistics
                features['object_features'] = {
                    'mean': float(np.mean(yolo_features)),
                    'std': float(np.std(yolo_features)),
                    'max': float(np.max(yolo_features)),
                    'min': float(np.min(yolo_features)),
                    'shape': list(yolo_features.shape)
                }
            except Exception as e:
                print(f"Error extracting YOLO features: {e}")
                
        return features
    
    except Exception as e:
        print(f"Error extracting image features: {e}")
        return {
            'width': 0 if 'width' not in locals() else width,
            'height': 0 if 'height' not in locals() else height,
            'channels': 0 if 'channels' not in locals() else channels,
            'aspect_ratio': 0,
            'mean_pixel_value': 0,
            'std_pixel_value': 0,
            'brightness': 0,
            'contrast': 0
        }

def manual_augment_image(img_array, config=None):
    """Apply manual data augmentation to image without TensorFlow"""
    if config is None:
        config = {
            'enabled': True,
            'flip': True,
            'rotation': 0.1,
            'zoom': 0.1,
            'contrast': 0.1,
            'brightness': 0.1,
            'blur': 0.0,
            'noise': 0.0,
            'cutout': 0.0,
            'mixup': None,
            'hue': 0.0,
            'saturation': 0.0
        }
    
    if not config.get('enabled', True) or img_array is None:
        return img_array
    
    try:
        # Convert to PIL Image
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Random horizontal flip
        if config.get('flip', False) and np.random.random() > 0.5:
            img = ImageOps.mirror(img)
        
        # Random rotation
        if config.get('rotation', 0) > 0:
            max_angle = 360 * config.get('rotation', 0)
            if max_angle > 0:
                angle = np.random.uniform(-max_angle, max_angle)
                img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        # Random zoom (crop and resize)
        if config.get('zoom', 0) > 0:
            zoom_factor = 1 - config.get('zoom', 0) * np.random.random()
            if zoom_factor < 1:
                width, height = img.size
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = left + new_width
                bottom = top + new_height
                img = img.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)
        
        # Random contrast
        if config.get('contrast', 0) > 0:
            factor = 1.0 + config.get('contrast', 0) * np.random.uniform(-1, 1)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        # Random brightness
        if config.get('brightness', 0) > 0:
            factor = 1.0 + config.get('brightness', 0) * np.random.uniform(-1, 1)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # Random hue/saturation (color)
        if config.get('saturation', 0) > 0 and img.mode == 'RGB':
            factor = 1.0 + config.get('saturation', 0) * np.random.uniform(-1, 1)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)
            
        # Random blur
        if config.get('blur', 0) > 0 and np.random.random() < config.get('blur', 0):
            radius = np.random.uniform(0, 2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
        # Random noise
        if config.get('noise', 0) > 0 and np.random.random() < config.get('noise', 0):
            img_array = np.array(img) / 255.0
            noise = np.random.normal(0, config.get('noise', 0) * 0.1, img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 1)
            return noisy_img
            
        # Cutout (random erasing)
        if config.get('cutout', 0) > 0 and np.random.random() < config.get('cutout', 0):
            img_array = np.array(img) / 255.0
            h, w = img_array.shape[:2]
            
            # Calculate size of cutout
            size = int(min(h, w) * config.get('cutout', 0))
            
            # Get random position
            x = np.random.randint(0, w - size + 1)
            y = np.random.randint(0, h - size + 1)
            
            # Apply cutout
            img_array[y:y+size, x:x+size] = 0
            return img_array
        
        # MixUp (if second image provided)
        if config.get('mixup') is not None and isinstance(config.get('mixup'), np.ndarray):
            mixup_img = config.get('mixup')
            alpha = np.random.beta(0.2, 0.2)
            img_array = np.array(img) / 255.0
            mixed = alpha * img_array + (1 - alpha) * mixup_img
            return np.clip(mixed, 0, 1)
            
        # Convert back to numpy array and normalize
        return np.array(img) / 255.0
    
    except Exception as e:
        print(f"Error in manual image augmentation: {e}")
        traceback.print_exc()
        return img_array

def augment_image(img_array, config=None):
    """Apply data augmentation to image using TensorFlow with advanced techniques"""
    if config is None:
        config = {
            'enabled': True,
            'flip': True,
            'rotation': 0.1,
            'zoom': 0.1,
            'contrast': 0.1,
            'brightness': 0.1,
            'shift': 0.1,
            'shear': 0.0,
            'channel_shift': 0.0,
            'gaussian_noise': 0.0,
            'gaussian_blur': 0.0,
            'cutout': 0.0,
            'grayscale': 0.0,
            'random_crop': False,
            'advanced': True
        }
    
    if not config.get('enabled', True) or img_array is None:
        return img_array
    
    # If TensorFlow augmentation fails, fall back to manual augmentation
    try:
        # Create augmentation layers
        augmentation_layers = []
        
        # Basic augmentations
        if config.get('flip', True):
            augmentation_layers.append(tf.keras.layers.RandomFlip("horizontal"))
        
        if config.get('rotation', 0) > 0:
            augmentation_layers.append(tf.keras.layers.RandomRotation(config.get('rotation', 0.1)))
        
        if config.get('zoom', 0) > 0:
            augmentation_layers.append(tf.keras.layers.RandomZoom(config.get('zoom', 0.1)))
        
        if config.get('contrast', 0) > 0:
            augmentation_layers.append(tf.keras.layers.RandomContrast(config.get('contrast', 0.1)))
            
        if config.get('brightness', 0) > 0:
            # TensorFlow doesn't have a direct RandomBrightness, so we'll use a lambda layer
            def random_brightness(x, factor):
                return x + tf.random.uniform([], -factor, factor)
            
            augmentation_layers.append(tf.keras.layers.Lambda(
                lambda x: random_brightness(x, config.get('brightness', 0.1))
            ))
            
        if config.get('shift', 0) > 0:
            augmentation_layers.append(tf.keras.layers.RandomTranslation(
                config.get('shift', 0.1), config.get('shift', 0.1)
            ))
            
        # Advanced augmentations
        if config.get('advanced', False):
            if config.get('shear', 0) > 0:
                # Apply random shearing
                augmentation_layers.append(
                    tf.keras.layers.RandomAffine(
                        0, shear=config.get('shear', 0.1) * 180  # Convert to degrees
                    )
                )
                
            if config.get('channel_shift', 0) > 0:
                # Apply random channel shifting
                def channel_shift(x, intensity):
                    return tf.image.random_hue(x, intensity)
                
                augmentation_layers.append(tf.keras.layers.Lambda(
                    lambda x: channel_shift(x, config.get('channel_shift', 0.1))
                ))
                
            if config.get('gaussian_noise', 0) > 0:
                # Add Gaussian noise
                def add_gaussian_noise(x, stddev):
                    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=x.dtype)
                    return x + noise
                
                augmentation_layers.append(tf.keras.layers.Lambda(
                    lambda x: add_gaussian_noise(x, config.get('gaussian_noise', 0.1))
                ))
                
            if config.get('gaussian_blur', 0) > 0 and np.random.random() < config.get('gaussian_blur', 0):
                # Apply Gaussian blur
                def gaussian_blur(x):
                    return tf.image.resize(
                        tf.image.resize(x, [tf.shape(x)[1]//2, tf.shape(x)[2]//2], 
                                method=tf.image.ResizeMethod.GAUSSIAN),
                        [tf.shape(x)[1], tf.shape(x)[2]],
                        method=tf.image.ResizeMethod.BILINEAR
                    )
                
                augmentation_layers.append(tf.keras.layers.Lambda(
                    lambda x: tf.cond(
                        tf.random.uniform([], 0, 1) < config.get('gaussian_blur', 0),
                        lambda: gaussian_blur(x),
                        lambda: x
                    )
                ))
                
            if config.get('cutout', 0) > 0 and np.random.random() < config.get('cutout', 0):
                # Apply random erasing (cutout)
                def random_erasing(x):
                    batch_size = tf.shape(x)[0]
                    img_height, img_width = tf.shape(x)[1], tf.shape(x)[2]
                    
                    # Calculate cutout size
                    cutout_size = tf.cast(tf.minimum(img_height, img_width) * config.get('cutout', 0.1), tf.int32)
                    
                    # Generate random coordinates
                    y1 = tf.random.uniform([batch_size], 0, img_height - cutout_size, dtype=tf.int32)
                    x1 = tf.random.uniform([batch_size], 0, img_width - cutout_size, dtype=tf.int32)
                    
                    # Create cutout boxes
                    boxes = tf.stack([y1, x1, y1 + cutout_size, x1 + cutout_size], axis=1)
                    
                    # Create batch indices
                    batch_indices = tf.range(batch_size)[:, tf.newaxis]
                    
                    # Create mask
                    mask = tf.ones_like(x)
                    
                    # Apply cutout
                    for i in range(batch_size):
                        y_start, x_start, y_end, x_end = boxes[i]
                        mask = tf.tensor_scatter_nd_update(
                            mask,
                            tf.stack([
                                tf.ones([cutout_size, cutout_size], dtype=tf.int32) * i,
                                tf.range(y_start, y_end)[:, tf.newaxis] + tf.zeros([cutout_size, cutout_size], dtype=tf.int32),
                                tf.zeros([cutout_size, cutout_size], dtype=tf.int32) + tf.range(x_start, x_end)[tf.newaxis, :]
                            ], axis=-1),
                            tf.zeros([cutout_size, cutout_size, 3], dtype=x.dtype)
                        )
                    
                    return x * mask
                
                augmentation_layers.append(tf.keras.layers.Lambda(
                    lambda x: tf.cond(
                        tf.random.uniform([], 0, 1) < config.get('cutout', 0),
                        lambda: random_erasing(x),
                        lambda: x
                    )
                ))
                
            if config.get('grayscale', 0) > 0 and np.random.random() < config.get('grayscale', 0):
                # Randomly convert to grayscale
                def random_grayscale(x):
                    return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
                
                augmentation_layers.append(tf.keras.layers.Lambda(
                    lambda x: tf.cond(
                        tf.random.uniform([], 0, 1) < config.get('grayscale', 0),
                        lambda: random_grayscale(x),
                        lambda: x
                    )
                ))
        
        # Apply all augmentations
        sequential_model = tf.keras.Sequential(augmentation_layers)
        
        # Apply augmentation
        img_tensor = tf.convert_to_tensor(img_array)
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        augmented_img = sequential_model(img_tensor, training=True)[0]  # Remove batch dimension
        
        # Ensure values are in valid range
        augmented_img = tf.clip_by_value(augmented_img, 0, 1)
        
        return augmented_img.numpy()
    
    except Exception as e:
        print(f"TensorFlow augmentation failed, falling back to manual augmentation: {e}")
        traceback.print_exc()
        return manual_augment_image(img_array, config) 

def get_image_hash(image_path):
    """Calculate a hash for an image to use as a unique identifier
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        str: Hash string uniquely identifying the image
    """
    try:
        # If the file doesn't exist, return a hash of the path
        if not os.path.exists(image_path):
            return hashlib.md5(image_path.encode()).hexdigest()
        
        # For small files, use the file content
        if os.path.getsize(image_path) < 10 * 1024 * 1024:  # 10MB
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        
        # For larger files, use a combination of stats and partial content
        stat = os.stat(image_path)
        file_info = f"{image_path}_{stat.st_size}_{stat.st_mtime}"
        
        # Read the first 1MB of the file
        with open(image_path, 'rb') as f:
            file_start = f.read(1024 * 1024)
            
        # Create a hash from the combined information
        hash_data = file_info.encode() + file_start
        return hashlib.md5(hash_data).hexdigest()
    
    except Exception as e:
        print(f"Error calculating image hash: {e}")
        # Fallback to just the path hash
        return hashlib.md5(str(image_path).encode()).hexdigest() 