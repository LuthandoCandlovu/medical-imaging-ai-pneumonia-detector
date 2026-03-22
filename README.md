<div align="center">

<!-- ANIMATED HEADER -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&duration=3000&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=🩺+Pneumonia+Detection+from+Chest+X-Rays;Powered+by+MobileNetV2+%2B+Grad-CAM;Explainable+AI+for+Medical+Imaging" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-00D4FF?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector?style=for-the-badge&color=gold)](https://github.com/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector/stargazers)

<br/>

> **A complete medical imaging AI pipeline** — detects pneumonia from chest X-rays  
> and highlights the lung regions that influenced every prediction using **Grad-CAM heatmaps**.

<br/>

<img src="https://img.shields.io/badge/Accuracy-~89.6%25-00FFCC?style=flat-square" />
<img src="https://img.shields.io/badge/Precision-~0.89-00D4FF?style=flat-square" />
<img src="https://img.shields.io/badge/Recall-~0.90-1E90FF?style=flat-square" />
<img src="https://img.shields.io/badge/F1--Score-~0.89-00FFCC?style=flat-square" />

</div>

---

## 🔍 Overview

This project demonstrates a **production-ready medical imaging pipeline** built on transfer learning and explainable AI. It trains a MobileNetV2-based classifier, evaluates it against held-out data, and wraps everything in a clean Streamlit web app — so anyone can upload a chest X-ray and get an instant, trustworthy prediction.

| Stage | What happens |
|---|---|
| 📥 **Input** | Raw chest X-ray images (JPEG/PNG) |
| ⚙️ **Preprocessing** | Resize to 224×224 · Normalise to \[0, 1\] |
| 🧠 **Model** | MobileNetV2 (ImageNet) + custom top layers |
| 🌡️ **Explainability** | Grad-CAM heatmap overlay |
| 🖥️ **Output** | Normal / Pneumonia label + confidence + heatmap |

---

## ✨ Key Features

- 🎯 **Accurate** — ~89.6 % test accuracy on the public Kaggle dataset  
- 🌡️ **Explainable** — Grad-CAM heatmaps pinpoint the lung regions driving each decision  
- 🖥️ **Interactive** — Upload any X-ray, get an instant result in the Streamlit UI  
- ⚡ **Efficient** — MobileNetV2 is lightweight enough to run on a laptop CPU  
- 🔁 **Reproducible** — Pre-trained weights included; re-training is one command away  

---

## 🧠 Model Architecture

```
Input Image (224 × 224 × 3)
        │
        ▼
┌─────────────────────────────────┐
│  MobileNetV2  (ImageNet weights)│
│  ── frozen base encoder ──      │
└──────────────┬──────────────────┘
               │
               ▼
     Global Average Pooling 2D
               │
               ▼
        Dense (128, ReLU)
               │
               ▼
          Dropout (0.5)
               │
               ▼
      Dense (1, Sigmoid) ──► Normal / Pneumonia
```

> 🔬 **Grad-CAM target layer:** `mobilenetv2_1.00_224/Conv_1`

---

## 📊 Dataset

**[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** — Kaggle

| Split | 🟢 NORMAL | 🔴 PNEUMONIA |
|---|---|---|
| **Train** | 1,341 images | 3,875 images |
| **Test** | 234 images | 390 images |

All images are resized to **224 × 224** and pixel values normalised to **\[0, 1\]**.

---

## 📈 Results

| Metric | Value | Progress |
|---|---|---|
| **Accuracy** | ~89.6 % | `████████████████████░` |
| **Precision** | ~0.89 | `████████████████████░` |
| **Recall** | ~0.90 | `█████████████████████` |
| **F1-Score** | ~0.89 | `████████████████████░` |

> 📝 Confusion matrix and ROC curve are generated automatically by `train.py`.

---

## 🚀 How to Run Locally

### 1 — Clone the repository

```bash
git clone https://github.com/LuthandoCandlovu/medical-imaging-ai-pneumonia-detector.git
cd medical-imaging-ai-pneumonia-detector
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Train the model *(optional — pre-trained weights included)*

```bash
python train.py
```

### 4 — Launch the web app

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser and start uploading X-rays.

---

## 🖥️ Live Demo

🚀 **[Try it on Hugging Face Spaces](https://huggingface.co)** ← replace with your actual URL after deployment

---

## 🛠️ Technologies

| Library | Purpose |
|---|---|
| ![TF](https://img.shields.io/badge/-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white&style=flat-square) | Model building & training |
| ![CV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white&style=flat-square) | Image preprocessing |
| ![ST](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=flat-square) | Interactive web interface |
| ![NP](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat-square) | Array operations |
| ![MPL](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) | Plotting & metrics |
| ![SK](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square) | Evaluation metrics |
| ![GH](https://img.shields.io/badge/-Git%20%26%20GitHub-181717?logo=github&logoColor=white&style=flat-square) | Version control |

---

## 🔮 Future Improvements

- [ ] Fine-tune pre-trained model layers for higher accuracy  
- [ ] Multi-class classification (COVID-19, viral / bacterial pneumonia)  
- [ ] Cloud deployment on AWS or GCP  
- [ ] Integration with a medical-grade DICOM viewer  

---

## 📁 Project Structure

```
medical-imaging-ai-pneumonia-detector/
│
├── app.py               # Streamlit web application
├── train.py             # Training script (MobileNetV2 + Grad-CAM)
├── requirements.txt     # Python dependencies
├── model/
│   └── pneumonia_model.h5   # Pre-trained weights
├── utils/
│   └── gradcam.py       # Grad-CAM implementation
└── README.md
```

---

## 📜 License

This project is licensed under the **[MIT License](LICENSE)**.

---

## 🙏 Acknowledgements

- **Dataset** — [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Base model** — [MobileNetV2 original paper](https://arxiv.org/abs/1801.04381)  
- **Grad-CAM** — Implementation adapted from [keras-io](https://keras.io/examples/vision/grad_cam/)

---

<div align="center">

### Made with ❤️ by [Luthando Candlovu](https://github.com/LuthandoCandlovu)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&duration=2000&pause=500&color=00FFCC&center=true&vCenter=true&width=500&lines=If+this+project+helped+you%2C+leave+a+⭐+star!;Contributions+and+feedback+are+welcome+🙌" alt="Footer typing" />

</div>
