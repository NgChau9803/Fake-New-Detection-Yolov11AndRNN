import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
from tqdm import tqdm
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.target_size = self.config.get('target_size', (224, 224))
        self.channels = self.config.get('channels', 3)
        
    def load_image(self, image_path, normalize=True, convert_to_rgb=True):
        """
        Load an image with error handling for corrupted images
        
        Args:
            image_path: Path to the image file
            normalize: Whether to normalize pixel values to [0,1]
            convert_to_rgb: Whether to convert to RGB
            
        Returns:
            Image as numpy array or None if loading failed
        """
        try:
            # Try to open the image
            with Image.open(image_path) as img:
                if convert_to_rgb and img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(self.target_size)
                img_array = np.array(img)
                
                # Handle grayscale images
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=2)
                
                # Handle RGBA images
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                
                # Normalize if requested
                if normalize:
                    img_array = img_array / 255.0
                    
                return img_array
                
        except (IOError, SyntaxError) as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image instead of None
            if normalize:
                return np.zeros((*self.target_size, self.channels))
            else:
                return np.zeros((*self.target_size, self.channels), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Unexpected error loading image {image_path}: {e}")
            if normalize:
                return np.zeros((*self.target_size, self.channels))
            else:
                return np.zeros((*self.target_size, self.channels), dtype=np.uint8)
    
    def augment_image(self, image, augmentation_level='standard'):
        """
        Apply different levels of image augmentation
        
        Args:
            image: PIL Image or numpy array
            augmentation_level: 'minimal', 'standard', or 'aggressive'
            
        Returns:
            Augmented image as numpy array
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            img = Image.fromarray(image)
        else:
            img = image
            
        if augmentation_level == 'minimal':
            # Just apply random horizontal flip
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                
        elif augmentation_level == 'standard':
            # Apply standard augmentations
            # Random horizontal flip
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                
            # Random brightness/contrast adjustment
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = ImageEnhance.Brightness(img).enhance(factor)
                
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = ImageEnhance.Contrast(img).enhance(factor)
                
        elif augmentation_level == 'aggressive':
            # Apply more aggressive augmentations
            # Random horizontal flip
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                
            # Random brightness/contrast/saturation adjustment
            if random.random() > 0.5:
                factor = random.uniform(0.7, 1.3)
                img = ImageEnhance.Brightness(img).enhance(factor)
                
            if random.random() > 0.5:
                factor = random.uniform(0.7, 1.3)
                img = ImageEnhance.Contrast(img).enhance(factor)
                
            if random.random() > 0.5:
                factor = random.uniform(0.7, 1.3)
                img = ImageEnhance.Color(img).enhance(factor)
                
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
                
            # Random perspective distortion
            if random.random() > 0.7:
                width, height = img.size
                scale = random.uniform(0.1, 0.3)
                x1 = random.uniform(0, scale) * width
                y1 = random.uniform(0, scale) * height
                x2 = random.uniform(1 - scale, 1) * width
                y2 = random.uniform(0, scale) * height
                x3 = random.uniform(1 - scale, 1) * width
                y3 = random.uniform(1 - scale, 1) * height
                x4 = random.uniform(0, scale) * width
                y4 = random.uniform(1 - scale, 1) * height
                
                coeffs = find_coeffs(
                    [(0, 0), (width, 0), (width, height), (0, height)],
                    [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                )
                
                img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
                
        # Convert back to numpy array
        img_array = np.array(img)
        
        # Normalize to [0,1] for return
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
            
        return img_array
    
    def extract_features(self, image, method='basic'):
        """
        Extract features from images using different methods
        
        Args:
            image: Image as numpy array
            method: Feature extraction method
            
        Returns:
            Extracted features
        """
        if method == 'basic':
            # Simple resizing and flattening
            if isinstance(image, np.ndarray):
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image)
            else:
                img = image
                
            img = img.resize((64, 64))
            features = np.array(img).flatten() / 255.0
            return features
            
        elif method == 'histogram':
            # Color histograms
            if isinstance(image, np.ndarray):
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image)
            else:
                img = image
                
            r, g, b = img.split()
            r_hist = np.array(r.histogram()) / 256.0
            g_hist = np.array(g.histogram()) / 256.0
            b_hist = np.array(b.histogram()) / 256.0
            return np.concatenate([r_hist, g_hist, b_hist])
            
        return None
    
    def batch_process_images(self, image_paths, preprocess_func=None):
        """
        Process a batch of images with progress tracking
        
        Args:
            image_paths: List of image paths
            preprocess_func: Function to apply to each image
            
        Returns:
            Processed images as numpy array
        """
        processed_images = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            img_array = self.load_image(img_path)
            
            if img_array is not None:
                if preprocess_func:
                    img_array = preprocess_func(img_array)
                processed_images.append(img_array)
                
        return np.array(processed_images)
        
    @staticmethod
    def create_tf_dataset(images, labels=None, batch_size=32, shuffle=True, augment=False):
        """
        Create a TensorFlow dataset from images and labels
        
        Args:
            images: Numpy array of images
            labels: Numpy array of labels (optional)
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            augment: Whether to apply data augmentation
            
        Returns:
            TensorFlow dataset
        """
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(images)
            
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
            
        if augment and labels is not None:
            # Add augmentation using tf.image operations
            def augment_map_fn(x, y):
                x = tf.image.random_flip_left_right(x)
                x = tf.image.random_brightness(x, 0.2)
                x = tf.image.random_contrast(x, 0.8, 1.2)
                return x, y
                
            dataset = dataset.map(augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def find_coeffs(source_coords, target_coords):
    """Helper function for perspective transform"""
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        
    A = np.matrix(matrix, dtype=float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8) 