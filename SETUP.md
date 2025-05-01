# Fake News Detection System - Setup Guide

This guide provides step-by-step instructions for setting up and running the Multimodal Fake News Detection System.

## System Requirements

- Python 3.11 (recommended)
- Sufficient disk space for datasets and models (~150GB recommended)
- At least 16GB of RAM (24GB+ recommended)
- GPU support is optional but recommended for faster training

## Installation

### Windows

1. Install Python 3.11 from the [official website](https://www.python.org/downloads/release/python-3116/).

2. Clone or download this repository to your local machine.

3. Open PowerShell and navigate to the project directory:
   ```
   cd path\to\FinalProject
   ```

4. Run the setup script:
   ```
   .\setup_and_run.ps1
   ```

   This script will:
   - Create a Python virtual environment using Python 3.11
   - Install all required dependencies
   - Create necessary directories
   - Provide an interactive menu to run the pipeline

### Linux/macOS

1. Install Python 3.11 if not already installed:
   
   On Ubuntu/Debian:
   ```
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev
   ```
   
   On macOS (using Homebrew):
   ```
   brew install python@3.11
   ```

2. Clone or download this repository to your local machine.

3. Open Terminal and navigate to the project directory:
   ```
   cd path/to/FinalProject
   ```

4. Make the setup script executable:
   ```
   chmod +x setup_and_run.sh
   ```

5. Run the setup script:
   ```
   ./setup_and_run.sh
   ```

## Manual Setup (Alternative)

If you prefer to set up manually instead of using the scripts:

1. Create a virtual environment:
   ```
   # Windows
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate

   # Linux/macOS
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```
   mkdir -p data/raw data/processed data/images data/cache logs/tensorboard models explanations config
   ```

## Data Preparation

### Dataset Structure

The system expects datasets to be organized in the following structure:

1. **Fakeddit**:
   - Place TSV files in `data/raw/fakeddit/`
   - Place images in `data/images/fakeddit/`

2. **FakeNewsNet**:
   - Place content in `data/raw/fakenewsnet/`
   - Place images in `data/images/fakenewsnet/`

### Running the Pipeline

#### Using the Interactive Menu

The setup scripts provide an interactive menu with the following options:

1. **Run Data Preprocessing**: Process the raw datasets
2. **Train Model**: Train the fake news detection model
3. **Evaluate Model**: Evaluate the model performance
4. **Generate Visualizations**: Create explanations and visualizations
5. **Run Complete Pipeline**: Execute all steps in sequence
6. **Exit**: Exit the program

#### Manual Execution

To run steps manually:

1. **Environment Setup and Path Validation**:
   ```
   python -m src.main --setup
   ```

2. **Data Preprocessing**:
   ```
   python -m src.main --process_datasets
   ```
   Or using the Python API:
   ```python
   from src.data.dataset import DatasetProcessor
   import yaml
   
   with open('config/config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   processor = DatasetProcessor(config)
   processor.process_datasets()
   combined_df = processor.combine_datasets()
   preprocessed_df = processor.preprocess_dataset(combined_df)
   ```

3. **Training**:
   ```
   python -m src.main --train
   ```
   The training process will:
   - Create a timestamped output directory
   - Save model configuration in JSON format
   - Generate training visualizations
   - Save the trained model

4. **Evaluation**:
   ```
   # Basic evaluation
   python -m src.main --evaluate
   
   # Detailed evaluation with additional metrics
   python -m src.main --evaluate --detailed_evaluation
   
   # Evaluate specific model
   python -m src.main --evaluate --model_path path/to/model
   ```

5. **Explanation Generation**:
   ```
   # Generate explanations
   python -m src.main --explain
   
   # Generate explanations for specific model
   python -m src.main --explain --model_path path/to/model
   ```

6. **Complete Pipeline**:
   ```
   python -m src.main --all
   ```

7. **Help and Options**:
   ```
   python -m src.main --help
   ```
   This will display all available command line options:
   - `--config`: Path to configuration file (default: config/config.yaml)
   - `--setup`: Setup environment and validate paths
   - `--process_datasets`: Process and prepare datasets
   - `--train`: Train the model
   - `--evaluate`: Evaluate the model
   - `--explain`: Generate explanations for predictions
   - `--all`: Run the complete pipeline
   - `--model_path`: Path to saved model for evaluation or prediction
   - `--detailed_evaluation`: Perform detailed evaluation with additional metrics

## Configuration

The system can be configured by editing the `config/config.yaml` file. Key configuration options include:

- Dataset parameters (paths, processing options)
- Model architecture (embedding dimensions, network structure)
- Training parameters (batch size, learning rate, etc.)
- Evaluation metrics
- Output paths

## Troubleshooting

### Common Issues

1. **Dependency Installation Errors**
   - Try updating pip: `pip install --upgrade pip`
   - Install dependencies one by one to identify problematic packages

2. **GPU Support**
   - For TensorFlow GPU support, you need a compatible CUDA installation
   - If you don't have a GPU, set `gpu_support: false` in the config

3. **Memory Errors**
   - Reduce batch size in `config/config.yaml`
   - Process smaller subsets of data for testing

4. **File Not Found Errors**
   - Ensure all paths in `config/config.yaml` are correct
   - Check that dataset files are in the expected locations

### Getting Help

If you encounter issues not covered here, please:
1. Check the console output for specific error messages
2. Refer to the documentation in the `docs` directory
3. Open an issue in the project repository with details about your problem

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Fake News Detection Research Papers](https://scholar.google.com/scholar?q=multimodal+fake+news+detection)
- [Python 3.11 Documentation](https://docs.python.org/3.11/)