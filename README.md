# Medical Imaging AI – Pneumonia Detection

This project uses a deep learning model (MobileNetV2) to classify chest X‑ray images as **Normal** or **Pneumonia**.

## Dataset
Chest X-Ray Images (Pneumonia) from Kaggle.

## Project Structure
- data/ – contains train, val, test folders
- models/ – saved models
- preprocess.py – image loading and augmentation
- model.py – builds the CNN model
- 	rain.py – trains and saves the model
- predict.py – predicts on a single image
- pp.py – Streamlit web app

## How to Run
1. Install requirements: pip install -r requirements.txt
2. Train: python train.py
3. Predict: python predict.py
4. Web app: streamlit run app.py

## Results
Achieved 92% accuracy on test set.
