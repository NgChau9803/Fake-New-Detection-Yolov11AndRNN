import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import tensorflow as tf
from io import BytesIO

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input with better error handling"""
    try:
        # Try to open the image
        img = Image.open(image_path)
        
        # Check if image is corrupted
        try:
            img.verify()
            # Need to reopen after verify
            img = Image.open(image_path)
        except:
            print(f"Warning: Image {image_path} is corrupted, using empty image")
            return np.zeros((*target_size, 3))
        
        # Convert to RGB (handles grayscale and RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize with proper aspect ratio
        img.thumbnail((max(target_size), max(target_size)), Image.LANCZOS)
        
        # Create new image with the target size and paste the resized image in the center
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_size[0] - img.width) // 2
        paste_y = (target_size[1] - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        # Convert to numpy array and normalize
        img_array = np.array(new_img) / 255.0  # Normalize to [0,1]
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return np.zeros((*target_size, 3))

def manual_augment_image(img_array, config=None):
    """Apply manual data augmentation to image without TensorFlow"""
    if config is None:
        config = {
            'flip': True,
            'rotation': 0.1,
            'zoom': 0.1,
            'contrast': 0.1,
            'brightness': 0.1
        }
    
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
        
        # Convert back to numpy array and normalize
        return np.array(img) / 255.0
    
    except Exception as e:
        print(f"Error in manual image augmentation: {e}")
        return img_array

def augment_image(img_array, config=None):
    """Apply data augmentation to image using TensorFlow"""
    if config is None or not config.get('enabled', True):
        return img_array
    
    # If TensorFlow augmentation fails, fall back to manual augmentation
    try:
        # Create augmentation layers
        augmentation_layers = []
        
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
        
        # Apply all augmentations
        sequential_model = tf.keras.Sequential(augmentation_layers)
        
        # Apply augmentation
        img_tensor = tf.convert_to_tensor(img_array)
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        augmented_img = sequential_model(img_tensor, training=True)[0]  # Remove batch dimension
        
        return augmented_img.numpy()
    
    except Exception as e:
        print(f"TensorFlow augmentation failed, falling back to manual augmentation: {e}")
        return manual_augment_image(img_array, config) 