# MEDSRGAN - CT Scan Super-Resolution

This project implements a Medical Super-Resolution Generative Adversarial Network (MEDSRGAN) for enhancing the resolution of CT scan images. The architecture is inspired by SRGAN and RCAN, incorporating attention mechanisms and residual learning for improved performance in medical image super-resolution.

[Paper Link](https://paperswithcode.com/paper/medsrgan-medical-images-super-resolution)

[Kaggle](https://www.kaggle.com/code/parsakh/medsrgan-ct-scan)
## Project Structure

The project is organized into a modular and professional structure to facilitate development, testing, and deployment.

```
medsrgan/
├── data/
│   ├── custom_dataset.py       # Defines CustomDataset for loading LR and HR image pairs.
│   └── dataset/                # Directory for 'train' and 'test' folders containing PNG image files.
├── config/
│   └── config.yml              # Centralized configuration file.
├── models/
│   ├── init.py             # Initializes the 'models' Python package.
│   ├── discriminator.py        # Implements the Discriminator network.
│   └── generator.py            # Implements the Generator network (featuring RWMAB and SRC blocks).
├── losses/
│   ├── init.py             # Initializes the 'losses' Python package.
│   ├── discriminator_loss.py   # Defines the Discriminator's loss function.
│   └── generator_loss.py       # Defines the Generator's perceptual loss function.
├── utils/
│   ├── init.py             # Initializes the 'utils' Python package.
│   ├── early_stopping.py       # Implements early stopping logic.
│   ├── transforms.py           # Defines image transformations (LR/HR, noise).
│   └── visualize.py            # Utility for plotting training performance.
├── trainer/
│   └── trainer.py              # Encapsulates the training and evaluation loop.
├── main.py                     # Main script to initiate the training process.
└── README.md                   # Project README file.                 # Project README file.
```

## Features

-   **Modular Design**: Code is organized into logical modules (data, models, losses, utils, trainer) for better readability, maintainability, and reusability.
-   **Generator Network**:
    -   Based on Residual Whole Map Attention Blocks (RWMAB) and Short Residual Connections (SRC).
    -   Utilizes PixelShuffle for efficient upsampling.
-   **Discriminator Network**:
    -   A patch-based discriminator that classifies image patches as real or fake.
    -   Includes intermediate feature extraction for perceptual loss.
-   **Perceptual Loss**:
    -   Generator loss combines content loss (VGG-based perceptual loss and L1 loss), adversarial loss, and adversarial feature matching loss.
-   **Early Stopping**: Prevents overfitting by monitoring a validation metric (e.g., PSNR) and stopping training if no improvement is observed for a specified number of epochs.
-   **Model Saving**: Automatically saves generator and discriminator checkpoints periodically and the best model based on validation performance.
-   **Performance Visualization**: Plots training losses (Generator and Discriminator), discriminator scores (real vs. fake), and image quality metrics (PSNR, SSIM).
-   **Sample Image Generation**: Saves sample super-resolved images during training to visually track progress.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd medsrgan
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The `CustomDataset` expects image paths in a pandas DataFrame.
The original notebook used data from `/kaggle/input/chest-ctscan-images/Data/`. But here it is necessary to put the data inside the dataset folder.
If you are running this code locally, ensure your data is structured similarly or modify `main.py` to point to your dataset location.
For initial testing without actual data, the `CustomDataset` includes a `use_dummy=True` option, which generates placeholder images.

## Usage

Before running, adjust the desired settings in `config/config.yml`. To train the MEDSRGAN model, execute the `main.py` script:

```bash
python main.py
```

## Training Configuration

Adjust training parameters in `main.py`:

-   `image_size`: Target resolution for HR images.
-   `batch_size`: Number of images per batch.
-   `learning_rate`: Initial learning rate for optimizers.
-   `epochs`: Total number of training epochs.
-   `patience`: Number of epochs for early stopping.
-   `min_delta`: Minimum change in validation metric to qualify as an improvement for early stopping.
-   `device`: 'cuda' for GPU training (recommended) or 'cpu'.
-   `model_logs_dir`: Directory to save model checkpoints.
-   `outputs_dir`: Directory to save generated sample images.

## Output

During training, the script will:
-   Print epoch-wise training progress (losses, scores, metrics).
-   Save sample super-resolved images to `output_images/MonthXX_DayYY_HourZZ_MinAA/`.
-   Save model checkpoints (Generator and Discriminator state_dicts) to `model_logs/MonthXX_DayYY_HourZZ_MinAA/`.
-   Save the best model based on validation PSNR to `model_logs/MonthXX_DayYY_HourZZ_MinAA/best_model.pth`.
-   Generate `performance_plots.png` in the root directory, showing training curves.

## Evaluation

The `Trainer` class includes an `evaluate` method that calculates PSNR and SSIM on the test dataset. This is called periodically during training and can be called independently after training.

## Future Improvements

-   **Command-line Arguments**: Implement `argparse` for easier configuration of training parameters.
-   **Logging**: Integrate a more robust logging system (e.g., TensorBoard, MLflow) for better tracking of experiments.
-   **Configuration Files**: Use YAML or JSON files for managing configurations.
-   **Data Augmentation**: Explore more advanced data augmentation techniques.
-   **Hyperparameter Tuning**: Implement tools for hyperparameter optimization.
-   **Deployment**: Develop a simple inference script or API for using the trained model.
