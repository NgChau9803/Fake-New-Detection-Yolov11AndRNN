# YOLOv11 Backbone for Fake News Detection

This document provides an overview of the YOLOv11 backbone implementation for the fake news detection model.

## Overview

The YOLOv11 backbone is a state-of-the-art feature extraction network based on the YOLO (You Only Look Once) architecture, adapted for multimodal fake news detection. It provides robust visual feature extraction with attention mechanisms and efficient design patterns.

Key features:
- Cross Stage Partial (CSP) design for efficient feature extraction
- Channel and spatial attention mechanisms for adaptive feature refinement
- Feature Pyramid Network (FPN) for multi-scale feature representation
- Configurable width and depth multipliers for model scaling
- Integration with the multimodal fusion architecture

## Architecture

The YOLOv11 backbone consists of the following components:

1. **Convolution Block**: Basic building block with convolution, batch normalization, and activation
2. **CSP Bottleneck**: Cross Stage Partial bottleneck with residual connection
3. **Channel Attention**: Squeeze-and-excitation style attention for channel-wise feature refinement
4. **CSP Stage**: Main feature extraction stage combining multiple CSP bottlenecks with attention
5. **Feature Pyramid Network**: Multi-scale feature fusion for improved detection performance
6. **YOLOv11 Backbone**: Main backbone class integrating all components

## Usage

### Basic Usage

To use the YOLOv11 backbone in your model:

```python
from src.models.yolo_backbone import YOLOv11Backbone

# Create YOLOv11 backbone
backbone = YOLOv11Backbone(
    input_shape=(224, 224, 3),
    width_mult=0.75,  # Controls model width
    depth_mult=0.67,  # Controls model depth
    use_fpn=True,     # Enable Feature Pyramid Network
    pooling="avg"     # Global pooling method
)

# Extract features from an image tensor
features = backbone(image_tensor)
```

### Integration with ImageFeatureExtractor

The YOLOv11 backbone is integrated into the `ImageFeatureExtractor` class:

```python
from src.models.image_model import ImageFeatureExtractor

# Create image feature extractor with YOLOv11 backbone
image_extractor = ImageFeatureExtractor(
    input_shape=(224, 224, 3),
    backbone_type="yolov11",
    pretrained=False,
    output_dim=512,
    use_attention=True,
    pooling="avg",
    backbone_params={
        "width_mult": 0.75,
        "depth_mult": 0.67,
        "use_fpn": True
    }
)

# Extract features
features = image_extractor(image_tensor)
```

### Using with the Full Model

To use the YOLOv11 backbone in the full multimodal fake news detection model, use the provided configuration file:

```python
import yaml
from src.models.model_factory import create_model

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model with YOLOv11 backbone
model = create_model(config, vocab_size=20000)
```

## Visualization

To visualize the features and attention maps produced by the YOLOv11 backbone, use the provided test script:

```bash
python tools/test_yolo_backbone.py --image path/to/image.jpg --visualize
```

For more detailed visualization with the full model, use the visualization tool:

```bash
python tools/visualize_attention.py --model path/to/model --image path/to/image.jpg --text "Sample news text"
```

## Configuration

The YOLOv11 backbone can be configured through the following parameters:

- `input_shape`: Input image dimensions (height, width, channels)
- `width_mult`: Width multiplier controlling the number of channels (default: 0.75)
- `depth_mult`: Depth multiplier controlling the number of blocks (default: 0.67)
- `use_fpn`: Whether to use Feature Pyramid Network (default: True)
- `pooling`: Pooling method - "avg", "max", or None (default: "avg")
- `classification`: Whether to add classification head (default: False)
- `num_classes`: Number of classes for classification head (default: 0)
- `dropout_rate`: Dropout rate for classification head (default: 0.0)

## Performance Considerations

The YOLOv11 backbone provides a good balance between performance and computational efficiency:

- Width and depth multipliers can be adjusted to scale the model according to hardware constraints
- FPN can be disabled for faster inference at a slight cost to accuracy
- Attention mechanisms can be disabled to reduce computational cost

## Example Configuration

Here's an example configuration for the YOLOv11 backbone in the model config:

```yaml
model:
  image:
    backbone: "yolov11"
    input_shape: [224, 224, 3]
    pooling: 'avg'
    trainable: false
    dropout_rate: 0.2
    use_attention: true
    output_dim: 512
    width_mult: 0.75
    depth_mult: 0.67
    use_fpn: true
```

## References

The YOLOv11 backbone implementation is inspired by:
- YOLOv4, YOLOv5, and YOLOv7 architectures
- Cross Stage Partial Networks
- Convolutional Block Attention Module (CBAM)
- Feature Pyramid Networks (FPN) 