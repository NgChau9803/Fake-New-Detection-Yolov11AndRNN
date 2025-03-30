# Fake News Detection System

A multimodal fake news detection system using RNN for text analysis and YOLOv11 for image analysis.

## Project Overview

This project implements a multimodal fake news detection system that leverages various machine learning and deep learning techniques to identify fake news articles.

## Project Structure

```
fake_news_detection/
├── config/             # Configuration files
├── data/               # Data storage
│   ├── raw/            # Original datasets
│   ├── processed/      # Processed datasets
│   └── images/         # Downloaded images
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
│   ├── data/           # Data processing modules
│   ├── models/         # Model implementations
│   ├── training/       # Training utilities
│   └── evaluation/     # Evaluation utilities
├── tools/              # Utility scripts
└── logs/               # Training logs
```

## Datasets

This system works with two datasets:
- **Fakeddit**: A multimodal dataset containing text, images, and metadata
- **FakeNewNet**: A news article dataset with textual and metadata information

## Features

- Multimodal analysis combining text, images, and metadata
- Text analysis using RNN with attention mechanisms
- Image analysis using YOLOv11 
- Fusion network for combined feature analysis
- Comprehensive evaluation with standard metrics
- Explainable AI using LIME for text and image interpretability

## Features
- Multimodal analysis of text and images.
- User-friendly interface for data preparation and model training.
- Evaluation metrics to assess model performance.

## How to Contribute
We welcome contributions! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Setup and Installation

1. Create a virtual environment (optional):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare directories:
   ```
   mkdir -p data/raw/fakeddit data/raw/fakenewnet data/processed data/images logs models
   ```

4. Place your datasets in the appropriate directories.

## Usage

1. Prepare datasets:
   ```
   python tools/prepare_datasets.py --config config/config.yaml
   ```

2. Download images:
   ```
   python tools/download_images.py --config config/config.yaml
   ```

3. Train the model:
   ```
   python main.py --mode train --config config/config.yaml
   ```

4. Evaluate a trained model:
   ```
   python main.py --mode evaluate --model models/your_trained_model --config config/config.yaml
   ```

## Citation

If you use this code in your research, please cite