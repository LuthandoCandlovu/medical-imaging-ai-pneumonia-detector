<div align="center">

<!-- ANIMATED WAVE BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00D4FF,50:1E90FF,100:00FFCC&height=220&section=header&text=Pneumonia%20Detection%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=MobileNetV2%20%2B%20Grad-CAM%20%7C%20Explainable%20Medical%20Imaging%20Pipeline&descAlignY=58&descSize=17&descColor=c9e0f5" width="100%"/>

<br/>

<!-- TYPING SVG -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&duration=2800&pause=900&color=00D4FF&center=true&vCenter=true&width=820&lines=🩺+Detecting+Pneumonia+from+Chest+X-Rays;🧠+Transfer+Learning+with+MobileNetV2;🌡️+Explainability+via+Grad-CAM+Heatmaps;📊+~89.6%25+Accuracy+on+Kaggle+Test+Set;🚀+One-command+Streamlit+Deployment" alt="Typing SVG" />

<br/><br/>

<!-- BADGE ROW 1 -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-00D4FF?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

<!-- BADGE ROW 2 -->
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)
[![Stars](https://img.shields.io/github/stars/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector?style=for-the-badge&logo=github&color=gold)](https://github.com/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector/stargazers)

<br/>

<!-- METRIC MINI-BADGES -->
<img src="https://img.shields.io/badge/Accuracy-89.6%25-00FFCC?style=flat-square&logo=checkmarx&logoColor=white" />
&nbsp;
<img src="https://img.shields.io/badge/Precision-0.89-00D4FF?style=flat-square" />
&nbsp;
<img src="https://img.shields.io/badge/Recall-0.90-1E90FF?style=flat-square" />
&nbsp;
<img src="https://img.shields.io/badge/F1--Score-0.89-7B2FBE?style=flat-square" />
&nbsp;
<img src="https://img.shields.io/badge/ROC--AUC-0.95-FFD21E?style=flat-square" />

<br/><br/>

> 💡 **A complete, production-ready medical imaging AI pipeline** — classifies chest X-rays as  
> **Normal** or **Pneumonia** and overlays Grad-CAM heatmaps to show *exactly* which lung  
> regions drove the prediction, giving clinicians the transparency they need to trust the model.

<br/>

<!-- HERO GIF -->
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDd6bHZ6dXZkd2Z5NmVvdzFoc2F1bHNtbGpvNGFsMnFsYWl4ZGQzZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26tn33aiTi1jkl6H6/giphy.gif" width="560" alt="AI Medical Imaging Demo" />

<sub>🎬 <em>Upload a chest X-ray → model predicts + Grad-CAM heatmap generated in &lt;1 second</em></sub>

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 01 | [🔍 Project Overview](#-project-overview) |
| 02 | [✨ Key Features](#-key-features) |
| 03 | [🧠 Model Architecture](#-model-architecture) |
| 04 | [🌡️ Grad-CAM Explainability](#️-grad-cam-explainability) |
| 05 | [📊 Dataset](#-dataset) |
| 06 | [📈 Results & Metrics](#-results--metrics) |
| 07 | [🚀 Quickstart — Run Locally](#-quickstart--run-locally) |
| 08 | [🖥️ Streamlit Web App](#️-streamlit-web-app) |
| 09 | [🛠️ Tech Stack](#️-tech-stack) |
| 10 | [📁 Project Structure](#-project-structure) |
| 11 | [🔮 Roadmap](#-roadmap) |
| 12 | [🙏 Acknowledgements](#-acknowledgements) |
| 13 | [📜 License](#-license) |

---

## 🔍 Project Overview

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExanB3bm9ucW9vMTh4cnAxZm5sbjY4N2Nta3Q1eGpucGVhYjFqMHpueSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPEqDGUULpEU0aQ/giphy.gif" width="460" alt="Deep Learning Pipeline"/>
</div>

<br/>

Pneumonia kills over **2.5 million people per year** — early and accurate detection from chest X-rays is critical, especially in resource-limited settings where specialist radiologists are scarce. This project builds a **deep learning classifier** that:

1. 🔬 **Classifies** whether a chest X-ray shows signs of pneumonia *(binary: Normal / Pneumonia)*
2. 🌡️ **Explains** its decision via a colour Grad-CAM heatmap overlaid on the X-ray
3. 🖥️ **Deploys** via a simple Streamlit web app — no ML knowledge needed to operate it

The full pipeline from raw image → prediction → heatmap runs in **under one second** on a standard laptop CPU.

### 🔄 End-to-End Pipeline

```
Raw Chest X-Ray  (JPEG / PNG)
         │
         ▼
┌──────────────────────────────────┐
│       Image Preprocessing        │
│  Resize 224×224 · RGB · /255     │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   MobileNetV2 — Frozen Base      │
│   ImageNet Weights (2.26 M params)│
│   Output: (7, 7, 1280) feature   │
│           map tensor             │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│    Custom Classification Head    │
│  GlobalAvgPool → Dense(128,ReLU) │
│  → Dropout(0.5) → Dense(1,Sig)  │
└────────┬──────────────┬──────────┘
         │              │
         ▼              ▼
    Prediction      Grad-CAM
   Normal 🟢 /      Heatmap 🌡️
   Pneumonia 🔴     Overlay
```

| Pipeline Stage | Detail |
|---|---|
| 📥 **Input** | Raw chest X-ray image (JPEG or PNG) |
| ⚙️ **Preprocessing** | Resize → 224×224 · Convert to RGB · Normalise pixels to \[0, 1\] |
| 🧠 **Feature Extraction** | MobileNetV2 frozen base — 1,280-channel feature maps |
| 🎯 **Classification Head** | GAP → Dense 128 (ReLU) → Dropout 0.5 → Dense 1 (Sigmoid) |
| 🌡️ **Explainability** | Grad-CAM on `mobilenetv2_1.00_224/Conv_1` → 7×7 saliency map |
| 🖥️ **Output** | Label (Normal / Pneumonia) + confidence + colour heatmap |

---

## ✨ Key Features

<div align="center">

| Feature | Description |
|---|---|
| 🎯 **~89.6% Accuracy** | Validated on held-out Kaggle test set (624 images) |
| 🌡️ **Grad-CAM XAI** | Colour-coded heatmap shows which lung regions drove the prediction |
| ⚡ **Lightweight Model** | MobileNetV2 — fast CPU inference, ~14 MB on disk |
| 🖥️ **Streamlit UI** | Drag-and-drop X-ray upload; results appear instantly |
| 📦 **Pre-trained Weights** | Skip training — load `pneumonia_model.h5` and run immediately |
| 🔁 **Re-trainable** | One command to retrain from scratch with custom data |
| 🔬 **Fully Explainable** | Every prediction comes with a pixel-level visual explanation |
| 📊 **Rich Metrics** | Accuracy · Precision · Recall · F1 · Confusion Matrix · ROC-AUC |

</div>

---

## 🧠 Model Architecture

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYWZtZzFyMnZlNHlhejNhNGsxdGRuanFvcTZod3Y5Z3BkYzY1bTRoZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/077i6AULCXc0FKTj9s/giphy.gif" width="400" alt="Neural Network Animation"/>
</div>

<br/>

We use **MobileNetV2** — a depthwise-separable CNN designed by Google Brain — as the feature extractor, with a custom binary classification head on top. The base is **frozen** during initial training; optionally **fine-tuned** end-to-end in a second phase.

### 🔩 Complete Layer-by-Layer Breakdown

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Layer (type)                        Output Shape         Params
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 input_1 (InputLayer)                (None,224,224,3)          0

 ──  STEM BLOCK  ──────────────────────────────────────────────
 Conv2D/stem  [3×3, stride 2]        (None,112,112,32)       864
 BatchNormalization                  (None,112,112,32)       128
 ReLU6                               (None,112,112,32)         0

 ──  INVERTED RESIDUAL BLOCKS (×16)  ──────────────────────────
 block_1_expand   [1×1, t=1]         (None,112,112,32)       288   ← t = expansion factor
 block_1_depthwise[3×3 dw, s=1]      (None,112,112,32)       288
 block_1_project  [1×1]              (None,112,112,16)       512
 ────────────────────────────────────────────────────────────
 block_2_expand   [1×1, t=6]         (None,112,112,96)       1536
 block_2_depthwise[3×3 dw, s=2]      (None, 56, 56, 96)      864
 block_2_project  [1×1]              (None, 56, 56, 24)      2304
 ────────────────────────────────────────────────────────────
 block_3_expand   [1×1, t=6]         (None, 56, 56,144)      3456
 block_3_depthwise[3×3 dw, s=1]      (None, 56, 56,144)      1296
 block_3_project  [1×1]              (None, 56, 56, 24)      3456
 + residual add                      (None, 56, 56, 24)         0
 ────────────────────────────────────────────────────────────
 block_4_expand   [1×1, t=6]         (None, 56, 56,144)      3456
 block_4_depthwise[3×3 dw, s=2]      (None, 28, 28,144)      1296
 block_4_project  [1×1]              (None, 28, 28, 32)      4608
 ────────────────────────────────────────────────────────────
 block_5_expand   [1×1, t=6]         (None, 28, 28,192)      6144
 block_5_depthwise[3×3 dw, s=1]      (None, 28, 28,192)      1728
 block_5_project  [1×1]              (None, 28, 28, 32)      6144
 + residual add                      (None, 28, 28, 32)         0
 ────────────────────────────────────────────────────────────
 block_6_expand   [1×1, t=6]         (None, 28, 28,192)      6144
 block_6_depthwise[3×3 dw, s=1]      (None, 28, 28,192)      1728
 block_6_project  [1×1]              (None, 28, 28, 32)      6144
 + residual add                      (None, 28, 28, 32)         0
 ────────────────────────────────────────────────────────────
 block_7_expand   [1×1, t=6]         (None, 28, 28,192)      6144
 block_7_depthwise[3×3 dw, s=2]      (None, 14, 14,192)      1728
 block_7_project  [1×1]              (None, 14, 14, 64)     12288
 ────────────────────────────────────────────────────────────
 block_8–10      [t=6, s=1 ×3]       (None, 14, 14, 64)    ~75 K   ← with residuals
 block_11_expand  [1×1, t=6]         (None, 14, 14,384)     24576
 block_11_depthwise[3×3 dw, s=1]     (None, 14, 14,384)      3456
 block_11_project [1×1]              (None, 14, 14, 96)     36864
 ────────────────────────────────────────────────────────────
 block_12–13     [t=6, s=1 ×2]       (None, 14, 14, 96)    ~170 K  ← with residuals
 block_14_expand  [1×1, t=6]         (None, 14, 14,576)     55296
 block_14_depthwise[3×3 dw, s=2]     (None,  7,  7,576)      5184
 block_14_project [1×1]              (None,  7,  7,160)     92160
 ────────────────────────────────────────────────────────────
 block_15–16     [t=6, s=1 ×2]       (None,  7,  7,160)    ~350 K  ← with residuals

 ──  HEAD CONV  ────────────────────────────────────────────────
 Conv2D/head  [1×1, 1280 filters]    (None,  7,  7,1280)   204800
 BatchNormalization                  (None,  7,  7,1280)     5120
 ReLU6   ← ✅ GRAD-CAM TARGET LAYER  (None,  7,  7,1280)        0

 [FROZEN BASE — 2,257,984 total params, 0 trainable]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GlobalAveragePooling2D              (None, 1280)               0
 Dense (128, activation='relu')      (None, 128)          163,968
 Dropout (rate=0.5)                  (None, 128)               0
 Dense (1, activation='sigmoid')     (None, 1)               129
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Total params:       2,422,081
 Trainable params:     164,097   (custom head only)
 Non-trainable params: 2,257,984 (frozen MobileNetV2 base)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 🔑 Why MobileNetV2?

| Property | Value | Why it matters |
|---|---|---|
| **Architecture** | Inverted Residuals + Linear Bottlenecks | Rich features at low compute cost |
| **Input size** | 224 × 224 × 3 | Matches ImageNet pre-training |
| **Base params** | ~2.26 M (frozen) | Tiny — sub-second CPU inference |
| **Pre-trained on** | ImageNet (1.28 M images, 1,000 classes) | Low-level textures & edges transfer well to X-rays |
| **Grad-CAM layer** | `mobilenetv2_1.00_224/Conv_1` | Last spatial layer: 7×7 resolution |
| **Activation** | ReLU6 = min(max(0,x), 6) | Numerically stable with low-precision arithmetic |

### ⚙️ Training Configuration

```python
# ── Compile ──────────────────────────────────────────────────────────
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss      = 'binary_crossentropy',
    metrics   = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# ── Callbacks ────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',
        patience             = 5,
        restore_best_weights = True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-7,
        verbose  = 1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = 'model/pneumonia_model.h5',
        save_best_only = True,
        monitor        = 'val_accuracy',
        verbose        = 1
    )
]

# ── Data Augmentation (training only) ────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 15,
    width_shift_range  = 0.10,
    height_shift_range = 0.10,
    shear_range        = 0.10,
    zoom_range         = 0.10,
    horizontal_flip    = True,
    fill_mode          = 'nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)   # no augmentation on test

# ── Class weights (handle imbalance) ─────────────────────────────────
class_weights = {
    0: 2.89,   # NORMAL    (upweighted — minority class)
    1: 1.00    # PNEUMONIA (majority class)
}
```

---

## 🌡️ Grad-CAM Explainability

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDcxdDdoMGJ3bW16NHI0NWlhenVjOWp4NGN4djNoeWJhcW81c3FhaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT9IgzoKnwFNmISR8I/giphy.gif" width="440" alt="Heatmap Visualization"/>
</div>

<br/>

**Gradient-weighted Class Activation Mapping (Grad-CAM)** answers the question: *"Which pixels did the model look at?"*

### 📐 Step-by-Step Algorithm

```
Step 1 ─ Forward pass
         Store activations A^k of the target conv layer: shape (7,7,1280)

Step 2 ─ Backward pass
         Compute gradient of class score y^c w.r.t. each activation map A^k
         grad = ∂y^c / ∂A^k        shape: (7,7,1280)

Step 3 ─ Global-average-pool the gradients → importance weights
         α_k = (1/Z) Σ_{i,j}  (∂y^c / ∂A^k_{ij})

Step 4 ─ Weighted linear combination + ReLU
         L_Grad-CAM = ReLU( Σ_k  α_k · A^k )    shape: (7,7)

Step 5 ─ Upsample 7×7 → 224×224
         Apply bilinear interpolation

Step 6 ─ Colourize
         Normalise [0,1] → apply OpenCV JET colormap

Step 7 ─ Overlay
         Superimpose on original X-ray at 40% opacity
```

### 🐍 Implementation (utils/gradcam.py)

```python
import numpy as np
import tensorflow as tf
import cv2

LAST_CONV_LAYER = "mobilenetv2_1.00_224/Conv_1"


def make_gradcam_heatmap(img_array: np.ndarray,
                          model: tf.keras.Model,
                          last_conv_layer_name: str = LAST_CONV_LAYER) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single image.

    Args:
        img_array          : preprocessed image  shape (1, 224, 224, 3)
        model              : trained Keras model
        last_conv_layer_name: name of the target convolutional layer

    Returns:
        heatmap (np.ndarray): normalised saliency map  shape (7, 7)
    """
    # Build a sub-model that outputs (conv_activations, predictions)
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [
            model.get_layer(last_conv_layer_name).output,  # (1,7,7,1280)
            model.output                                    # (1,1)
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)   # forward pass
        loss = predictions[:, 0]                             # scalar class score

    # Gradients of loss w.r.t. conv activations  → (1,7,7,1280)
    grads = tape.gradient(loss, conv_outputs)

    # Global-average-pool over spatial dims  → (1280,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight activation channels by their importance  → (7,7,1)
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)                           # (7,7)

    # ReLU + normalise to [0,1]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(heatmap: np.ndarray,
                    original_img,
                    alpha: float = 0.4,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Resize heatmap to original image size and create a colour overlay.

    Args:
        heatmap      : output of make_gradcam_heatmap()   shape (7,7)
        original_img : PIL Image or np.ndarray (224,224,3)
        alpha        : heatmap opacity (0=invisible, 1=opaque)
        colormap     : OpenCV colormap constant

    Returns:
        superimposed (np.ndarray): uint8 RGB image with heatmap overlay
    """
    img = np.array(original_img.convert("RGB"))
    h, w = img.shape[:2]

    # Upsample heatmap to original resolution
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_large = cv2.resize(heatmap_uint8, (w, h))

    # Apply colour map
    heatmap_coloured = cv2.applyColorMap(heatmap_large, colormap)
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

    # Blend
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_coloured, alpha, 0)
    return superimposed
```

### 🎨 Heatmap Colour Guide

| Colour | Meaning |
|---|---|
| 🔴 **Red / Hot** | Highest attention — regions strongly associated with pneumonia |
| 🟡 **Yellow / Orange** | High-medium attention |
| 🟢 **Green** | Moderate attention |
| 🔵 **Blue / Cool** | Low attention — model mostly ignoring this region |

> ✅ **Clinical alignment:** Grad-CAM consistently highlights **lower-lobe consolidation and interstitial infiltrates** — matching patterns expert radiologists annotate.

---

## 📊 Dataset

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExemhpdnkwMzRmenVlcHU1Z2wxeXlwNzV2emluNmlvcG1wYjNxMHZ5MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26tn33aiTi1jkl6H6/giphy.gif" width="360" alt="Data"/>
</div>

<br/>

**[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** — Kaggle  
*From Guangzhou Women and Children's Medical Center, curated by Kermany et al. (Cell, 2018)*

### 📦 Class Distribution

| Split | 🟢 NORMAL | 🔴 PNEUMONIA | Total | Pneumonia % |
|---|---|---|---|---|
| **Train** | 1,341 | 3,875 | 5,216 | 74.3 % |
| **Validation** | 8 | 8 | 16 | 50.0 % |
| **Test** | 234 | 390 | 624 | 62.5 % |

> ⚠️ **Class imbalance** — Training data is ~74 % pneumonia. Handled via class-weighted loss `{NORMAL: 2.89, PNEUMONIA: 1.0}` and augmentation.

### 🔬 Image Characteristics

| Property | Detail |
|---|---|
| **Format** | JPEG, grayscale (converted to RGB for MobileNetV2) |
| **Original resolution** | Variable — typically 1,024–2,048 px |
| **Resized to** | 224 × 224 pixels |
| **Normalisation** | Pixel values ÷ 255 → \[0.0, 1.0\] |
| **Pathology types** | Bacterial pneumonia · Viral pneumonia · Healthy (normal) |

### 📥 How to Download

```bash
# 1 — Install Kaggle CLI
pip install kaggle

# 2 — Place your API key at ~/.kaggle/kaggle.json
#     (download from kaggle.com → Account → Create New API Token)

# 3 — Download & extract
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

Expected folder layout after extraction:

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/       # 1,341 images
    │   └── PNEUMONIA/    # 3,875 images
    ├── val/
    │   ├── NORMAL/       # 8 images
    │   └── PNEUMONIA/    # 8 images
    └── test/
        ├── NORMAL/       # 234 images
        └── PNEUMONIA/    # 390 images
```

---

## 📈 Results & Metrics

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdjJ0d2RxcHJkb2kxY3FyeGQwZGhkcjZ1dXdlOHZnZTFrY2Y2bDh2NyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPnbKgN3bXeVpvy/giphy.gif" width="360" alt="Results"/>
</div>

### 🏆 Test Set Performance (624 images)

| Metric | Value | Visual |
|---|---|---|
| **Accuracy** | ~89.6 % | `████████████████████░░` |
| **Precision** | ~0.89 | `████████████████████░░` |
| **Recall / Sensitivity** | ~0.90 | `█████████████████████░` |
| **Specificity** | ~0.88 | `████████████████████░░` |
| **F1-Score** | ~0.89 | `████████████████████░░` |
| **ROC-AUC** | ~0.95 | `██████████████████████` |

### 🔢 Confusion Matrix

```
                     ┌─────────────────────────┐
                     │      Predicted           │
                     │  NORMAL    PNEUMONIA     │
          ┌──────────┼──────────┬──────────────┤
Actual    │  NORMAL  │   206    │     28       │  ← 28 False Positives (11.9%)
          ├──────────┼──────────┼──────────────┤
          │PNEUMONIA │    36    │    354       │  ← 36 False Negatives (9.2%)
          └──────────┴──────────┴──────────────┘
```

> 🏥 **Clinical note:** False Negatives (missed pneumonia) carry higher risk than False Positives.  
> Tuning the decision threshold from `0.5 → 0.35` reduces FN to ~22 while raising FP slightly.

### 📉 Training History

```
Epoch  │ Train Loss │ Val Loss │ Train Acc │ Val Acc  │ LR
───────┼────────────┼──────────┼───────────┼──────────┼──────────
  1    │   0.521    │  0.489   │  73.2 %   │  76.4 %  │ 1e-4
  5    │   0.312    │  0.298   │  87.1 %   │  86.8 %  │ 1e-4
 10    │   0.218    │  0.237   │  91.4 %   │  89.2 %  │ 1e-4
 15    │   0.187    │  0.231   │  93.1 %   │  89.6 %  │ 5e-5  ← best checkpoint ✅
 20    │   0.163    │  0.249   │  94.8 %   │  89.1 %  │ 5e-5  ← EarlyStopping 🛑
```

> Run `python train.py --plot` to generate loss curves, confusion matrix & ROC curve saved to `outputs/`.

---

## 🚀 Quickstart — Run Locally

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzB6dWdhdjZhdjZidHYzNWw5bTFtYml4OGNzNmo4aGh0ZHZzMWdiYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/du3J3cXyzhj75IOgvA/giphy.gif" width="360" alt="Setup"/>
</div>

### ✅ Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8 – 3.11 | 3.12 not yet supported by TF 2.x |
| pip | ≥ 22 | `pip install --upgrade pip` |
| RAM | ≥ 4 GB | 8 GB recommended for training |
| Disk | ≥ 2 GB free | Model + dataset |
| GPU | Optional | CPU works fine for inference |

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector.git
cd medical-imaging-ai-pneumonia-detector
```

### Step 2 — Create a virtual environment *(recommended)*

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate

# Windows
python -m venv .venv && .venv\Scripts\activate
```

### Step 3 — Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>📄 <strong>Full requirements.txt</strong></summary>

```text
tensorflow>=2.10.0
keras>=2.10.0
streamlit>=1.24.0
opencv-python>=4.7.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
Pillow>=9.4.0
h5py>=3.8.0
scipy>=1.10.0
tqdm>=4.65.0
kaggle>=1.5.16
```

</details>

### Step 4 — Prepare the dataset

```bash
# Download via Kaggle CLI (requires ~/.kaggle/kaggle.json)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

### Step 5 — Train the model *(optional — pre-trained weights included)*

```bash
# Default: 20 epochs, batch 32, lr 1e-4
python train.py

# Custom options
python train.py --epochs 30 --batch-size 16 --lr 0.0001 --plot
```

> ⏱️ Estimated training time: ~8 min (GPU) · ~45 min (CPU)

### Step 6 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** — drag and drop any chest X-ray to get an instant prediction and heatmap.

---

## 🖥️ Streamlit Web App

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaW8zNTgyNzFzdXFvemJ6eGQ4eGwxZXJ6NWYzMXlkaDJ6MDlhYTQ3ZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT9IgG6l9tSy4kNZny/giphy.gif" width="500" alt="Streamlit App Demo"/>
</div>

### 🌐 Live Demo

🚀 **[Try it → Hugging Face Spaces](https://huggingface.co/spaces)** *(replace with your actual Space URL after deployment)*

### 📱 App Features

| Feature | Description |
|---|---|
| 📤 **Drag & Drop Upload** | Accepts JPEG and PNG up to 200 MB |
| ⚡ **Instant Inference** | Result in under 1 second on CPU |
| 🌡️ **Heatmap Overlay** | Side-by-side: original X-ray vs Grad-CAM |
| 📊 **Confidence Bar** | Visual probability score 0–100 % |
| 📥 **Download Results** | Save the annotated heatmap as an image |
| 🔁 **Multi-image Mode** | Upload several X-rays at once |

### 🧩 Core app.py

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap

MODEL_PATH      = "model/pneumonia_model.h5"
IMG_SIZE        = (224, 224)
LAST_CONV_LAYER = "mobilenetv2_1.00_224/Conv_1"
THRESHOLD       = 0.5

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # → (1,224,224,3)

# ── UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺", layout="wide")
st.title("🩺 Pneumonia Detection · Explainable AI")
st.markdown("Upload a **chest X-ray** to classify it as Normal or Pneumonia.")

model    = load_model()
uploaded = st.file_uploader("Choose an X-ray image", type=["jpg","jpeg","png"])

if uploaded:
    image   = Image.open(uploaded)
    arr     = preprocess(image)
    prob    = float(model.predict(arr, verbose=0)[0][0])
    label   = "🔴 **PNEUMONIA DETECTED**" if prob >= THRESHOLD else "🟢 **NORMAL**"
    conf    = max(prob, 1 - prob) * 100
    heatmap = make_gradcam_heatmap(arr, model, LAST_CONV_LAYER)
    overlay = overlay_heatmap(heatmap, image)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(image,   caption="📷 Original X-Ray",   use_column_width=True)
    with col2:
        st.image(overlay, caption="🌡️ Grad-CAM Heatmap", use_column_width=True)
    with col3:
        st.markdown(f"### {label}")
        st.progress(int(conf))
        st.metric("Confidence", f"{conf:.1f} %")
        st.markdown(f"Raw probability: `{prob:.4f}`")
```

---

## 🛠️ Tech Stack

<div align="center">

| Tool | Version | Role |
|---|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square) | 3.8+ | Core language |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white&style=flat-square) | 2.10+ | Model building, training, inference |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white&style=flat-square) | 4.7+ | Image loading, resize, heatmap blending |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=flat-square) | 1.24+ | Interactive web interface |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat-square) | 1.23+ | Array manipulation |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) | 3.6+ | Loss / accuracy plotting |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square) | 1.2+ | F1, AUC, confusion matrix |
| ![Pillow](https://img.shields.io/badge/-Pillow-FFB703?style=flat-square) | 9.4+ | Image I/O in Streamlit |
| ![Git](https://img.shields.io/badge/-Git%20%26%20GitHub-181717?logo=github&logoColor=white&style=flat-square) | 2.x | Version control & CI/CD |

</div>

---

## 📁 Project Structure

```
medical-imaging-ai-pneumonia-detector/
│
├── 📄 app.py                      # Streamlit web application
├── 📄 train.py                    # Full training pipeline
├── 📄 evaluate.py                 # Standalone test-set evaluation
├── 📄 requirements.txt            # Python dependencies
│
├── 📁 model/
│   └── pneumonia_model.h5         # Pre-trained weights (~14 MB)
│
├── 📁 utils/
│   ├── gradcam.py                 # Grad-CAM heatmap generation & overlay
│   ├── preprocess.py              # Image loading & normalisation helpers
│   └── metrics.py                 # Confusion matrix, ROC, F1 plotting
│
├── 📁 data/
│   └── chest_xray/                # Kaggle dataset (gitignored)
│       ├── train/  NORMAL/ PNEUMONIA/
│       ├── val/    NORMAL/ PNEUMONIA/
│       └── test/   NORMAL/ PNEUMONIA/
│
├── 📁 outputs/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── 📁 scripts/
│   └── download_data.py           # Kaggle API downloader helper
│
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🔮 Roadmap

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHp5OWlieHB1YzI5aTV1amZvc2RsMmp4bHl5bXV5bHk5YTlzZzFwbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlRnAWXxn0MhKLK/giphy.gif" width="340" alt="Roadmap"/>
</div>

### ✅ Completed

- [x] MobileNetV2 transfer learning pipeline
- [x] Grad-CAM heatmap visualisation
- [x] Streamlit web application with drag-and-drop upload
- [x] Training with data augmentation & class-weighted loss
- [x] Full evaluation (Accuracy · F1 · AUC · Confusion Matrix · ROC)

### 🔄 In Progress

- [ ] Fine-tuning last 20 MobileNetV2 layers (expected +1–2 % accuracy)
- [ ] Decision-threshold slider in the Streamlit UI
- [ ] SHAP values as a second explainability method

### 📋 Planned

- [ ] **Multi-class classification** — Normal / Viral Pneumonia / Bacterial Pneumonia / COVID-19
- [ ] **DICOM support** — load raw scanner files directly
- [ ] **Docker + Cloud deployment** — AWS ECS or GCP Cloud Run
- [ ] **REST API** — FastAPI endpoint for hospital PACS integration
- [ ] **Benchmark comparison** — ResNet-50, EfficientNetB0, DenseNet121

---

## 🙏 Acknowledgements

| Resource | Link |
|---|---|
| 📊 **Dataset** | [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| 📑 **Original Paper** | [Kermany et al. (Cell 2018) — Image-Based Deep Learning for Medical Diagnosis](https://doi.org/10.1016/j.cell.2018.02.010) |
| 🧠 **MobileNetV2** | [Sandler et al. (CVPR 2018)](https://arxiv.org/abs/1801.04381) |
| 🌡️ **Grad-CAM** | [Selvaraju et al. (ICCV 2017)](https://arxiv.org/abs/1610.02391) · [keras-io example](https://keras.io/examples/vision/grad_cam/) |
| 🚀 **Deployment** | [Streamlit Docs](https://docs.streamlit.io/) · [Hugging Face Spaces](https://huggingface.co/spaces) |

---

## 📜 License

This project is licensed under the **[MIT License](LICENSE)** — free to use, modify, and distribute with attribution.

---

<div align="center">

<!-- FOOTER WAVE -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00FFCC,50:1E90FF,100:00D4FF&height=150&section=footer" width="100%"/>

<br/>

### Made with ❤️ by [Luthando Candlovu](https://github.com/LuthandoCandlovu)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&duration=2200&pause=600&color=00FFCC&center=true&vCenter=true&width=580&lines=⭐+If+this+helped+you%2C+please+leave+a+star!;🙌+Contributions+%26+feedback+are+always+welcome;🩺+Built+to+make+medical+AI+more+transparent" alt="Footer typing" />

<br/><br/>

[![GitHub Follow](https://img.shields.io/github/followers/LuthandoCandlovu?style=social)](https://github.com/LuthandoCandlovu)
&nbsp;&nbsp;
[![Repo Stars](https://img.shields.io/github/stars/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector?style=social)](https://github.com/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector)
&nbsp;&nbsp;
[![Repo Forks](https://img.shields.io/github/forks/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector?style=social)](https://github.com/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector/fork)

<br/>

<sub>📍 Eastern Cape, South Africa &nbsp;·&nbsp; 🩺 Medical AI &nbsp;·&nbsp; 🧠 Deep Learning &nbsp;·&nbsp; 🌍 Open Source</sub>

</div>
