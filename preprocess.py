import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path, target_size=(224,224)):
    """
    Load an image, resize, and normalize pixel values.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert BGR to RGB
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0         # normalize to [0,1]
    return img

def create_data_generators(train_dir, val_dir, test_dir, batch_size=32, target_size=(224,224)):
    """
    Create ImageDataGenerator for training, validation, and testing.
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for validation/test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator
