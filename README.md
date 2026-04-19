# Object Detection using CNN on CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.14-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarvesh1karandikar/Object-Detection-using-CNN-on-CIFAR-10/blob/master/CNN%2075%20Percent%20Accuracy.ipynb)

A convolutional neural network (CNN) image classifier trained on CIFAR-10, achieving **75.1% test accuracy** across 10 object categories using a 3-block CNN with Adam optimizer and exponential learning rate decay.

---

## Quick Start

### Train the model (one-time, ~10 min on GPU / 60 min on CPU)
```bash
pip install -r requirements.txt
python train.py
```

### Run the Gradio demo
```bash
python app.py
```
Then open http://localhost:7860 and upload any image.

---

## What This Does

This project builds and trains a CNN to classify 32x32 colour images into one of 10 CIFAR-10 classes:

> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Two models are developed:
- **Base model** — 2 conv layers + 2 FC layers, ~73% accuracy
- **BetterModel** — 3 conv layers + 2 FC layers, **75.1% test accuracy** (25 epochs)

The base notebook (`CNN - Cifar10 Base.ipynb`) also explores a fully-connected TinyNet (~50% accuracy) and dropout experiments, useful as a learning baseline for comparing FC vs. convolutional architectures.

---

## Architecture

### BetterModel (best result — 75.1% accuracy)

| Layer | Config | Output shape |
|-------|--------|--------------|
| Input | 32x32 RGB image | (?, 32, 32, 3) |
| Conv1 | 7x7 kernel, 32 filters, stride 1, ReLU | (?, 32, 32, 32) |
| MaxPool1 | 3x3, stride 2 | (?, 16, 16, 32) |
| MinMax Norm | Feature normalisation | (?, 16, 16, 32) |
| Conv2 | 5x5 kernel, 64 filters, stride 1, ReLU | (?, 16, 16, 64) |
| MaxPool2 | 3x3, stride 2 | (?, 8, 8, 64) |
| Conv3 | 3x3 kernel, 32 filters, stride 1, ReLU | (?, 8, 8, 32) |
| MaxPool3 | 3x3, stride 2 | (?, 4, 4, 32) |
| Flatten | — | (?, 512) |
| FC1 | 384 units, ReLU | (?, 384) |
| FC2 (output) | 10 units (softmax cross-entropy loss) | (?, 10) |

**Optimizer:** Adam with exponential learning rate decay
- Initial LR: `5e-4`
- Decay rate: `0.96` every 500 steps

**Training config:** 25 epochs, batch size 64, 49,000 training / 1,000 validation / 10,000 test samples.

---

## Results

| Model | Epochs | Val Accuracy | Test Accuracy |
|-------|--------|-------------|---------------|
| TinyNet (FC only) | 25 | ~49% | 50.6% |
| BaseModel (2-conv) | 5 | — | ~73% |
| **BetterModel (3-conv)** | **25** | **74.7%** | **75.1%** |

Training and validation accuracy curves are generated inline in the notebook. The BetterModel reached ~71% validation accuracy by epoch 5 and climbed steadily to 74.7% by epoch 15 before converging.

---

## Tech Stack

- **Python 3.6** — core language
- **TensorFlow 1.14** — model definition, training, and checkpoint saving
- **NumPy** — data manipulation
- **Matplotlib** — loss and accuracy visualisation
- **Jupyter Notebook** — interactive development environment

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/sarvesh1karandikar/Object-Detection-using-CNN-on-CIFAR-10.git
cd Object-Detection-using-CNN-on-CIFAR-10
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download CIFAR-10 data

```bash
bash get_datasets.sh
```

This downloads the CIFAR-10 Python dataset (~163 MB) from the University of Toronto and unpacks it into `data/`.

### 4. Run the notebook

```bash
jupyter notebook "CNN 75 Percent Accuracy.ipynb"
```

Or run it directly in Google Colab — no local setup required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarvesh1karandikar/Object-Detection-using-CNN-on-CIFAR-10/blob/master/CNN%2075%20Percent%20Accuracy.ipynb)

---

## Live Demo (Concept)

A portfolio demo would work like this:

1. User uploads any image (or provides a URL)
2. Image is resized to 32x32 and normalised
3. Model runs inference and returns top-3 class predictions with confidence scores
4. Output: e.g. "dog: 72%, cat: 15%, deer: 6%"

This could be deployed as a [Gradio](https://gradio.app/) app on [Hugging Face Spaces](https://huggingface.co/spaces) — free hosting, no GPU required for inference at this model size.

---

## What I Learned / Key Insights

- **Depth matters more than width at this scale** — adding a third conv layer with smaller 3x3 kernels improved accuracy by ~2% over the two-layer baseline at a low parameter cost.
- **Normalisation between blocks helps** — applying min-max normalisation on the output of the first pooling layer stabilised training and reduced early loss spikes.
- **FC networks plateau fast on image tasks** — the fully-connected TinyNet topped out at ~50%, confirming that spatial feature extraction via convolution is essential even for small 32x32 images.
- **Learning rate decay is critical for CNNs** — exponential decay on Adam allowed the network to make large updates early and fine-tune later, avoiding oscillation near convergence.
- **Dropout experimentation** — the base notebook explores multiple keep-probabilities (0, 0.25, 0.50, 0.75) on a small training subset, demonstrating how dropout acts as a regulariser against overfitting.
