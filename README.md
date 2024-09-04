
# MedSRGAN: Medical Images Super-Resolution Using Generative Adversarial Networks

This repository contains a PyTorch implementation of **MedSRGAN**, a novel approach for enhancing the resolution of medical images using Generative Adversarial Networks (GANs). MedSRGAN is designed specifically to improve the quality of medical images, such as CT scans and MRIs, by reconstructing high-resolution images from their low-resolution counterparts. The model aims to help medical professionals make more accurate diagnoses by providing clearer and more detailed images.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Background

Super-resolution of medical images is a critical task for enhancing image quality, which is essential for accurate diagnosis and treatment planning. Traditional methods often fail to capture fine details and can introduce artifacts that compromise diagnostic utility. **MedSRGAN** leverages GANs to provide a powerful and effective solution to this problem by:
1. Utilizing a generator network to produce high-resolution images.
2. Training a discriminator network to distinguish between high-resolution and low-resolution images, thus encouraging the generator to produce more realistic results.

## Features

- **GAN-based Model**: Employs a GAN framework to enhance the resolution of medical images effectively.
- **Customizable Network**: Easily adjustable network architecture for various datasets and requirements.
- **Training and Evaluation Scripts**: End-to-end training and evaluation scripts to facilitate reproducibility.
- **Support for Multiple Modalities**: Adaptable to different types of medical imaging modalities (e.g., MRI, CT).
- **Pre-trained Models**: Available pre-trained models for quick testing and validation.

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/parsakhavarinejad/MedSRGAN-Pytorch.git
cd MedSRGAN
pip install -r requirements.txt
```

Make sure you have PyTorch installed. You can follow the instructions from the [PyTorch official website](https://pytorch.org/get-started/locally/).

## Usage

### 1. Pre-trained Model Inference

<!-- To use a pre-trained model for super-resolution:

```bash
python infer.py --input_dir path/to/low_res_images --output_dir path/to/save_results --model_path path/to/pretrained/model
``` -->
Not yet developed

### 2. Training from Scratch

To train the model from scratch:

```bash
python train.py --config configs/train_config.yaml
```

You can modify the `train_config.yaml` file to set the training parameters, such as learning rate, batch size, and number of epochs.

## Dataset

To train the MedSRGAN model, you will need a dataset of paired low-resolution and high-resolution medical images. You can use any publicly available datasets, such as:

- [FastMRI](https://fastmri.org/)
- [NAMIC MRBrainS](https://www.nitrc.org/projects/mrbrains/)

Make sure to organize your dataset as follows:

```
/data
    /train
        /LR  # Low-resolution images
        /HR  # High-resolution images
    /val
        /LR
        /HR
```

Update the dataset paths in the configuration file (`configs/train_config.yaml`).

## Training

To train the model:

1. Prepare your dataset.
2. Adjust the training parameters in `configs/train_config.yaml`.
3. Run the training script:

```bash
python train.py --config configs/train_config.yaml
```

During training, the model checkpoints and logs will be saved in the `checkpoints` and `logs` directories, respectively.

## Evaluation

To evaluate the trained model:

```bash
python evaluate.py --model_path path/to/trained/model --dataset_dir path/to/evaluation/dataset
```

The evaluation script will compute common metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) to assess the model's performance.

## Results

| Metric     | Value (Dataset 1) | Value (Dataset 2) |
|------------|------------------|-------------------|
| PSNR (dB)  | XX.XX            | XX.XX             |
| SSIM       | XX.XX            | XX.XX             |

Qualitative results:

![Sample Results](assets/sample_results.png)

## Contributing

We welcome contributions to improve this project. Please submit a pull request or open an issue to discuss your ideas.

### How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Original Paper: *MedSRGAN: Medical Images Super-Resolution Using Generative Adversarial Networks*.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [FastMRI Dataset](https://fastmri.org/)
- [NAMIC MRBrainS Dataset](https://www.nitrc.org/projects/mrbrains/)

---

Like our GPT? Try our full AI-powered search engine and academic features for free at [consensus.app](https://consensus.app/?utm_source=chatgpt).
```

Copy this content to your `README.md` file to use it in your GitHub repository.