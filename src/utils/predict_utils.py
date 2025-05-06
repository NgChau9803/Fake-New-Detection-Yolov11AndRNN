import numpy as np

def predict_labels_from_probs(probs, threshold=0.5):
    """
    Convert model output probabilities to class labels ('fake' or 'real') for a batch.
    Args:
        probs: np.ndarray or list of probabilities (shape: (batch_size,) or (batch_size, 1))
        threshold: float, threshold for classifying as 'fake'
    Returns:
        np.ndarray of string labels ('fake' or 'real')
    """
    probs = np.asarray(probs).flatten()
    labels = (probs >= threshold).astype(int)
    return np.where(labels == 1, 'fake', 'real')

def predict_label_single(model, text, image, metadata, threshold=0.5, **kwargs):
    """
    Predict 'fake' or 'real' for a single sample using the model.
    Args:
        model: Trained Keras model
        text: np.ndarray or list (shape: (seq_len,))
        image: np.ndarray (shape: (224, 224, 3))
        metadata: np.ndarray or list (shape: (num_features,))
        threshold: float, threshold for classifying as 'fake'
        **kwargs: Any additional fields required by the model
    Returns:
        str: 'fake' or 'real'
    """
    import numpy as np
    # Prepare input as batch of 1
    inputs = {
        'text': np.expand_dims(text, 0),
        'image': np.expand_dims(image, 0),
        'metadata': np.expand_dims(metadata, 0),
    }
    # Add any extra fields
    for k, v in kwargs.items():
        inputs[k] = np.expand_dims(v, 0)
    prob = model.predict(inputs)[0][0]
    return 'fake' if prob >= threshold else 'real' 