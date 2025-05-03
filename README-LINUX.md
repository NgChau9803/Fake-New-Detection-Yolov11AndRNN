# Running Multimodal Fake News Detection on Linux

This guide will help you run the Multimodal Fake News Detection system on Linux systems.

## Prerequisites

- Linux operating system (tested on Ubuntu/Lubuntu)
- Python 3.10+ installed
- At least 4GB of RAM
- Sufficient disk space for datasets

## Setup Instructions

### 1. Install Miniconda (Recommended)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
```

### 2. Create a Conda Environment

```bash
conda create -y -n tf python=3.10
conda activate tf
```

### 3. Install Dependencies

```bash
# Install TensorFlow and core dependencies
conda install -y tensorflow
conda install -y pyyaml pandas scikit-learn nltk pillow tqdm matplotlib lime shap textblob opencv seaborn

# Install additional packages
pip install contractions emoji tensorflow-addons
```

## Common Issues & Solutions

### 1. TensorFlow Memory Mapping Error

If you encounter this error:
```
ImportError: ... failed to map segment from shared object
```

Run these commands to fix memory settings:
```bash
chmod +x fix_memory.sh
sudo ./fix_memory.sh
```

### 2. Missing Images Problem

If preprocessing completes but reports very few valid images (e.g., "Records with valid images: 1 (0.00%)"):

```bash
# Make the fix script executable
chmod +x fix_images_and_run.sh

# Run the comprehensive fix
./fix_images_and_run.sh
```

This script will:
- Fix image paths
- Create synthetic images
- Update dataset files
- Patch the code to handle missing images
- Run preprocessing with optimized settings

### 3. Out of Memory (OOM) Issues

If your system kills the process during preprocessing:

1. Make sure you're using the patched code:
   ```bash
   bash apply_dataset_patch.sh
   ```

2. Run with reduced batch size:
   ```bash
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   export TF_NUM_INTEROP_THREADS=1
   export TF_NUM_INTRAOP_THREADS=2
   python -m src.main --process_datasets
   ```

## Running the System

### 1. Data Processing

```bash
python -m src.main --process_datasets
```

### 2. Training

```bash
python -m src.main --train
```

### 3. Evaluation

```bash
python -m src.main --evaluate
```

### 4. Visualization

```bash
python -m src.main --visualize
```

### 5. Full Pipeline

```bash
python -m src.main --all
```

## Troubleshooting

If you encounter issues:

1. Check log files in the `logs` directory
2. Ensure your conda environment is active (`conda activate tf`)
3. Verify that all dependencies are installed
4. Check if synthetic images have been created in `data/images/synthetic`
5. Make sure the patched code is applied with `bash apply_dataset_patch.sh`

## Support

For additional help:
- Review `IMAGE_SOLUTIONS.md` for detailed explanations of image-related fixes
- Check `SETUP_INSTRUCTIONS.md` for complete setup instructions 