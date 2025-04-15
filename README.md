# Fake News Detection with Multi-Modal Learning

This project implements a multi-modal deep learning system for fake news detection using text, image, and metadata.

## Data Processing Pipeline

The data processing pipeline is designed to handle two datasets:

1. **Fakeddit** - Contains news articles with text and associated images
2. **FakeNewNet** - Contains news articles from GossipCop and PolitiFact with text and associated images

### Directory Structure

- The project code is in `D:\UniversityResource\Py\BDM\FinalProject`
- Raw data and images are stored on a separate disk `E:\BDMProject`
- Processed data is stored in the `data` directory within the project

### Running the Pipeline

To process the datasets, use the main script:

```bash
# Process all datasets from raw to TensorFlow datasets
python -m src.main --all

# Or run specific steps
python -m src.main --process_datasets --combine --preprocess
```

### Pipeline Steps

1. **Raw Data Processing**
   - Load raw data from Fakeddit and FakeNewNet
   - Standardize format across datasets
   - Extract metadata and image paths

2. **Preprocessing**
   - Text cleaning and normalization
   - Image preprocessing and validation
   - Feature extraction

3. **Dataset Creation**
   - Create TensorFlow datasets for training, validation, and testing
   - Support for balanced sampling to handle class imbalance
   - Option for cross-dataset validation

### Checking Processed Data

To check and validate the processed data:

```bash
# Check all aspects of processed data
python -m src.utils.check_data --check-all

# Check specific aspects
python -m src.utils.check_data --visualize
python -m src.utils.check_data --check-stats
```

## Model Architecture

The model follows a multi-modal architecture:

1. **Input Layers**
   - Text Input: 256-dimensional vector
   - Image Input: 512-dimensional vector
   - Metadata Input: 128-dimensional vector

2. **Feature Extraction**
   - Text: BiLSTM with attention
   - Image: YOLOv11 for object detection
   - Metadata: Embedding layers

3. **Fusion Network**
   - Cross-modal attention
   - Adaptive feature fusion

4. **Classification Head**
   - Multi-layer perceptron
   - Binary classification (Fake/Real)

## Configuration

Configuration is managed through YAML files:

```bash
# View the configuration file
cat config/config.yaml
```

The configuration file controls:
- Data paths and processing options
- Model architecture parameters
- Training hyperparameters

## Requirements

- Python 3.8+
- TensorFlow 2.x
- PyYAML
- pandas
- numpy
- scikit-learn
- NLTK
- PIL

## Citation

If you use this code, please cite:

```
@article{fakenews2023,
  title={Fake News Detection using Multi-Modal Deep Learning},
  author={Your Name},
  year={2023}
}
```

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