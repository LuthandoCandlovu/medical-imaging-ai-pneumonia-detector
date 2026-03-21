import os
from preprocess import create_data_generators
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths (adjust if needed)
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Create generators
train_gen, val_gen, test_gen = create_data_generators(train_dir, val_dir, test_dir)

# Build model
model = build_model()

# Callbacks
checkpoint = ModelCheckpoint('models/best_model.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max',
                             verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the final model
model.save('models/final_model.h5')
