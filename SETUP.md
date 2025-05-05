# Fake News Detection System - Setup Guide

**Quick Start:**

- For most users, just run `./setup_and_run.sh` (Linux/macOS) or `./setup_and_run.ps1` (Windows).
- These scripts will automatically:
  - Create a Python 3.11 virtual environment
  - Install all dependencies
  - Create all required directories
  - Provide an interactive menu for all pipeline steps

---

## System Requirements

- Python 3.11 (recommended)
- Sufficient disk space for datasets and models (~150GB recommended)
- At least 16GB of RAM (24GB+ recommended)
- GPU support is optional but recommended for faster training

## Automated Installation (Recommended)

### Linux/macOS

1. Ensure Python 3.11 is installed:
   ```
   python3.11 --version
   # If not found, install as follows:
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev
   # Or on macOS:
   brew install python@3.11
   ```
2. Make the setup script executable and run it:
   ```
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

### Windows

1. Ensure Python 3.11 is installed and available as `py -3.11`.
2. Run the setup script in PowerShell:
   ```
   ./setup_and_run.ps1
   ```

- The script will handle all environment setup, dependencies, and directory creation.
- You do **not** need to manually create venvs or install requirements.

## Manual Setup (Alternative)
(Only if you cannot use the scripts)

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

- If you get errors about Python version or venv creation, ensure Python 3.11 is installed and available as `python3.11` (Linux/macOS) or `py -3.11` (Windows).
- If you have issues with permissions, try running the script with `sudo` (Linux/macOS) or as Administrator (Windows).
- If venv activation fails, try activating manually: `source venv/bin/activate` (Linux/macOS) or `.\.venv\Scripts\Activate.ps1` (Windows).

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