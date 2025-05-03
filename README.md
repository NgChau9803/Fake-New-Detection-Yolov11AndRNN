# Multimodal Fake News Detection System

A machine learning system for detecting fake news using multimodal analysis of text, images, and metadata.

## Project Overview

This project implements a multimodal approach to fake news detection, leveraging:

1. **Text Analysis**: BiLSTM networks with attention mechanisms to process textual content
2. **Image Analysis**: YOLOv11 and other CNN-based models to analyze visual content
3. **Metadata Processing**: Structured data handling with feature engineering
4. **Cross-Modal Fusion**: Advanced techniques to combine features from different modalities

## Datasets

The system works with two primary datasets:

1. **Fakeddit**: A large-scale multimodal dataset with text, metadata, and associated images
2. **FakeNewsNet**: A comprehensive dataset containing news content from various domains (GossipCop, PolitiFact) with article text, images, and social context

## Installation and Setup

### System Requirements

- Python 3.8+
- TensorFlow 2.x
- At least 8GB RAM (16GB+ recommended)
- GPU support is optional but recommended

### Setup Environment

```bash
# Set up conda environment
conda create -n tf python=3.8
conda activate tf

# Install requirements
pip install -r requirements.txt

# Set up NLTK resources
python setup_nltk.py
```

## Running the System

### Data Processing

The system provides flexible ways to process the datasets based on memory constraints:

```bash
# Process all datasets
python process_datasets.py --dataset all

# Process only Fakeddit with reduced sample size to manage memory
python process_datasets.py --dataset fakeddit --sample-size 5000 --chunk-size 1000

# Process only FakeNewsNet
python process_datasets.py --dataset fakenewsnet

# Fix image paths without full processing
python process_datasets.py --fix-images
```

### Model Training

Training can be done on individual datasets or combined data:

```bash
# Train on combined dataset (recommended)
python train_model.py --dataset combined

# Train on Fakeddit only with memory optimization
python train_model.py --dataset fakeddit --sample-size 5000 --lightweight

# Train on FakeNewsNet with fewer epochs
python train_model.py --dataset fakenewsnet --epochs 10
```

### Image Path Fixing

If you encounter image path issues, use the dedicated tool:

```bash
# Fix image paths for all datasets
python fix_images.py --dataset all

# Analyze image paths before fixing
python fix_images.py --analyze

# Create synthetic images for missing ones
python fix_images.py --create-synthetic --num-synthetic 200
```

## Project Structure

```
├── config/                  # Configuration files
│   ├── config.yaml          # Main configuration
│   └── slang_dict.yaml      # Text processing dictionaries
├── data/                    # Data directory
│   ├── raw/                 # Original datasets
│   ├── processed/           # Processed datasets
│   ├── images/              # Image files
│   └── cache/               # Cache for feature extraction
├── src/                     # Source code
│   ├── data/                # Data processing modules
│   ├── models/              # Model implementations
│   ├── training/            # Training utilities
│   ├── evaluation/          # Evaluation metrics
│   └── utils/               # Utility functions
├── process_datasets.py      # Data processing script
├── train_model.py           # Model training script
├── fix_images.py            # Image path fixing utility
└── requirements.txt         # Dependencies
```

## Model Architecture

The architecture follows a multimodal approach:

1. **Input Layer**: Processes text, images, and metadata
2. **Feature Extraction**: Modality-specific processing with BiLSTM for text, YOLOv11 for images
3. **Fusion Network**: Cross-modal attention mechanisms to combine features
4. **Classification Head**: Multi-layer perceptron for final classification
5. **Output Layer**: Binary classification (real/fake)

## Memory Management

The system includes several memory optimization strategies:

- Chunked data processing
- Sample size reduction options
- Lightweight model configurations
- Garbage collection during processing
- Resource limiting to prevent crashes

## Troubleshooting

If you encounter memory issues:
1. Reduce sample size with `--sample-size` option
2. Use smaller chunk sizes with `--chunk-size`
3. Activate lightweight mode with `--lightweight` during training
4. Process datasets individually instead of combined

## Model Training

After fixing the dataset processing issues, we've successfully trained a text-based fake news detection model using scikit-learn:

- **Model**: TF-IDF Vectorizer + LogisticRegression with class balancing
- **Performance**: 
  - Training accuracy: 91.46%
  - Validation accuracy: 75.20%
  - F1-score: 0.74-0.75

### Key Features for Classification

**Features indicating fake news:**
- 'psbattle' (likely from "Photoshop Battle" subreddit)
- 'says'
- 'girl'
- 'car'
- 'tiny'
- 'police'

**Features indicating real news:**
- 'poster'
- 'available'
- '2018'
- 'colorized'
- 'circa'
- 'cutouts'
- 'discussions'

## Usage

### Data Processing
```bash
python process_datasets.py --dataset all --sample-size 5000
```

### Model Training
```bash
python train_model.py --dataset combined --sample-size 5000
```

The model will be saved to `data/processed/combined_sklearn_model.pkl` for later use.

### Making Predictions
```bash
# Predict with a single text input
python predict.py --text "Breaking News: New study shows that drinking water is linked to immortality"

# Run in interactive mode
python predict.py --interactive
```

Example output:
```
Prediction: FAKE NEWS
Confidence: 57.93%
Probability real: 42.07%
Probability fake: 57.93%

Important features in this text:
  shows: +0.5563
  water: +0.2822
  breaking: -0.2371
  news: +0.2353
  drinking: -0.0738
```

## System Requirements

The system has been adapted to run on Linux environments with limited memory. Previous Windows-specific dependencies have been resolved.