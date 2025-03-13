# DCGAN on CelebA Dataset

## Overview
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic face images using the **CelebA dataset**. The model is trained using **PyTorch** and leverages convolutional layers to improve the quality of generated images.

## Dataset
The project uses the **CelebA (CelebFaces Attributes Dataset)**, which contains over 200,000 celebrity images. The dataset is automatically downloaded and preprocessed before training.

## Requirements
To run this project, you need the following dependencies:

- Python 3.7+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- PIL (Pillow)

You can install the required packages using:

```bash
pip install torch torchvision numpy matplotlib pillow
```

## Model Architecture
The project implements a **DCGAN**, which consists of:

- **Generator**: A neural network that creates fake images from random noise.
- **Discriminator**: A neural network that distinguishes real images from fake ones.

### Generator
The generator takes a random noise vector (latent space) and generates images through transposed convolutional layers.

### Discriminator
The discriminator is a CNN-based binary classifier that determines whether an image is real or fake.

## Training Details
The model is trained using **Adam optimizer** with the following hyperparameters:

- **Batch size**: 128
- **Image size**: 64x64
- **Latent vector size**: 100
- **Learning rate**: 0.0002
- **Beta1**: 0.5
- **Epochs**: 8

## Usage
To train the model, run the notebook:

```bash
jupyter notebook DCGAN_Assignment3_038_A2.ipynb
```

The notebook will:
1. Download and preprocess the CelebA dataset.
2. Define the DCGAN architecture.
3. Train the model using the dataset.
4. Generate and visualize new images.

## Results
After training, the generator produces **realistic-looking face images**. The output can be visualized using matplotlib.

## Acknowledgments
- This implementation is inspired by the official **DCGAN PyTorch tutorial**.
- CelebA dataset: [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


