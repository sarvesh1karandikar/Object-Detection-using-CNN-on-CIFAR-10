# CLAUDE.md — Object Detection using CNN on CIFAR-10

## Project Summary

This repository contains two Jupyter notebooks implementing image classification on the CIFAR-10 dataset using Convolutional Neural Networks built with TensorFlow 1.14. The project was developed as a coursework exercise (CSCI-599) and demonstrates the progression from a fully-connected baseline to a 3-layer CNN achieving 75.1% test accuracy.

**Repo:** `sarvesh1karandikar/Object-Detection-using-CNN-on-CIFAR-10`  
**Branch:** `master`  
**Notebooks:**
- `CNN 75 Percent Accuracy.ipynb` — main deliverable (BetterModel, 75.1% accuracy)
- `CNN - Cifar10 Base.ipynb` — baseline experiments (TinyNet FC, dropout study)

---

## How to Run

### Prerequisites

- Python 3.6
- TensorFlow 1.x (not compatible with TF 2.x without migration)
- See `requirements.txt` for full dependencies

### Steps

```bash
# 1. Clone
git clone https://github.com/sarvesh1karandikar/Object-Detection-using-CNN-on-CIFAR-10.git
cd Object-Detection-using-CNN-on-CIFAR-10

# 2. Install deps
pip install -r requirements.txt

# 3. Download CIFAR-10 (Linux/macOS)
bash get_datasets.sh
# Windows: run get_datasets.sh via Git Bash or manually download
# from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# 4. Launch notebook
jupyter notebook "CNN 75 Percent Accuracy.ipynb"
```

The notebook runs end-to-end: loads data, defines the model, trains for 25 epochs, evaluates on the test set, and saves a TF checkpoint to `lib/tf_models/problem2/`.

---

## Model Architecture Details

### BetterModel (final, 75.1% test accuracy)

```
Input:      (batch, 32, 32, 3)
Conv1:      7x7 kernel, 32 filters, stride 1, padding SAME  → (batch, 32, 32, 32)
ReLU + MaxPool(3x3, stride 2) + MinMax Normalisation        → (batch, 16, 16, 32)
Conv2:      5x5 kernel, 64 filters, stride 1, padding SAME  → (batch, 16, 16, 64)
ReLU + MaxPool(3x3, stride 2)                               → (batch, 8, 8, 64)
Conv3:      3x3 kernel, 32 filters, stride 1, padding SAME  → (batch, 8, 8, 32)
ReLU + MaxPool(3x3, stride 2)                               → (batch, 4, 4, 32)
Flatten                                                     → (batch, 512)
FC1:        384 units, ReLU                                 → (batch, 384)
FC2:        10 units (logits)                               → (batch, 10)
Loss:       softmax cross-entropy
```

**Optimizer:** Adam  
**LR schedule:** Exponential decay — initial `5e-4`, decay rate `0.96` per 500 global steps  
**Batch size:** 64  
**Epochs:** 25  
**Data split:** 49,000 train / 1,000 val / 10,000 test

### BaseModel (2-conv baseline, ~73% accuracy)

Same structure as BetterModel but without Conv3 and without the MinMax normalisation layer between Conv1 and Conv2. Trained for only 5 epochs.

---

## Current Accuracy and Limitations

| Model | Test Accuracy |
|-------|--------------|
| TinyNet (fully-connected) | 50.6% |
| BaseModel (2-conv) | ~73% |
| BetterModel (3-conv) | **75.1%** |

**Limitations:**

1. **TF 1.x only** — uses `tf.Session`, `tf.contrib`, and `tf.placeholder` APIs that were removed in TensorFlow 2.0. Running this code requires either TF 1.14 or `tf.compat.v1` migration.
2. **Low resolution input** — CIFAR-10 images are 32x32; the model cannot generalise to higher-resolution photos without resizing, which discards detail.
3. **No data augmentation** — no random flips, crops, or colour jitter used during training. Adding these would likely push accuracy to 80%+.
4. **No batch normalisation** — the MinMax normalisation used is a non-standard approach; proper BatchNorm layers would improve gradient flow and training stability.
5. **No regularisation in BetterModel** — the dropout layers in `fc3` and `fc4` are commented out; this risks overfitting on longer training runs.
6. **Checkpoint format** — saved as TF 1.x `.ckpt` files; not portable to TF 2.x or PyTorch without conversion.
7. **Accuracy ceiling** — state-of-the-art on CIFAR-10 exceeds 99% (EfficientNet, ViT). This custom CNN sits below the commonly cited 80% threshold for "good" results on the benchmark.

---

## Enhancement TODO List (Portfolio Demo-ability)

### Accuracy improvements

- [ ] **Add data augmentation** — random horizontal flip, random crop (pad by 4, crop to 32), colour jitter. Expected gain: +3–5%.
- [ ] **Add BatchNorm** after every conv layer. Expected gain: +2–3%, faster convergence.
- [ ] **Re-enable dropout** in FC layers (keep prob ~0.5). Reduces overfitting on val set.
- [ ] **Migrate to TF 2 / Keras** — rewrite using `tf.keras.Sequential` for cleaner code and TF2 compatibility.
- [ ] **Try ResNet-like skip connections** — a 9-layer ResNet typically hits 93%+ on CIFAR-10.

### Demo / deployment

- [ ] **Gradio UI (Quick win)** — wrap inference in a `gr.Interface` with image upload, display top-3 predictions and confidence bars. Deploy free on HuggingFace Spaces.
- [ ] **Convert model to ONNX or TF SavedModel** — enables browser or mobile inference via ONNX.js or TFLite.
- [ ] **Streamlit app (Medium lift)** — richer UI with sample images, class descriptions, and a confusion matrix viewer.
- [ ] **HuggingFace Space + improved model (Big lift)** — retrain with augmentation + BatchNorm to 80%+, then publish as a polished Space with a demo video.

### Code quality

- [ ] Modularise model definitions into `lib/models.py`
- [ ] Add a `train.py` script so the notebook is not required to train
- [ ] Pin and document Python/TF versions with a `conda` environment file

---

## Recommended Demo Tier

**Quick Win — Gradio UI on HuggingFace Spaces**

**Justification:** The trained model checkpoint already exists and inference at 32x32 is near-instant on CPU. A Gradio interface requires fewer than 30 lines of code to wrap: load checkpoint, resize uploaded image to 32x32, run forward pass, return top-3 labels with confidence. Hosting on HuggingFace Spaces is free, public, and embeddable in a portfolio page — maximum visibility for minimal effort, and it gives recruiters a live, clickable demo without them needing to clone anything.
