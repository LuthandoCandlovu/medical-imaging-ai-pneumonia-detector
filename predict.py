import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess_image

def predict_image(image_path, model_path='models/best_model.h5'):
    """
    Load model and predict class for a single image.
    """
    model = load_model(model_path)
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)   # add batch dimension
    pred = model.predict(img)[0][0]     # probability
    if pred >= 0.5:
        return f"Pneumonia (confidence: {pred:.2%})"
    else:
        return f"Normal (confidence: {1-pred:.2%})"

# Example usage
if __name__ == '__main__':
    # Replace with your test image path
    result = predict_image('data/test/NORMAL/0001.jpg')
    print(result)
