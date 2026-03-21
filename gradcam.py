import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image.
    
    Args:
        model: Keras model
        img_array: Preprocessed input image (4D tensor)
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the class to explain (None uses predicted class)
    
    Returns:
        heatmap: Grayscale heatmap (2D array)
    """
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    
    # Compute gradients of the loss with respect to the conv output
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs by the gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Average over the channel dimension to get the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # Apply ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on the original image.
    
    Args:
        image: Original image (RGB, range 0-255)
        heatmap: 2D heatmap (0-1)
        alpha: Transparency factor
        colormap: OpenCV colormap
    
    Returns:
        overlayed image (RGB)
    """
    # Resize heatmap to image dimensions
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed
