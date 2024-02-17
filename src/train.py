import os
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Define directories
artificial_dir = os.path.join('resources', 'imagesets', 'artificial')
real_dir = os.path.join('resources', 'imagesets', 'real')


# Function to load images from directory
def load_and_preprocess_images(directory, target_size=(256, 256)):
    images = []
    for filename in os.listdir(directory):
        # Check if the file is a supported image format
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Load the image using PIL
            image = Image.open(os.path.join(directory, filename))
            # Convert RGBA image to RGB if it has four channels
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            # Resize the image to the target size using OpenCV
            resized_image = cv2.resize(np.array(image), target_size, interpolation=cv2.INTER_LINEAR)
            # Print the shape of the resized image
            # Convert the image to numpy array and normalize pixel values to [0, 1]
            image = resized_image / 255.0
            # Append the preprocessed image to the list
            images.append(image)
    return np.array(images)


# Load and preprocess images from the directories
artificial_images = load_and_preprocess_images(artificial_dir)
real_images = load_and_preprocess_images(real_dir)

# Create labels for the images
artificial_labels = np.ones(len(artificial_images))  # 1 for artificial images
real_labels = np.zeros(len(real_images))  # 0 for real images

# Combine images and labels
X = np.concatenate([artificial_images, real_images], axis=0)
y = np.concatenate([artificial_labels, real_labels], axis=0)

# Shuffle the data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
# Define the new model architecture
model = tf.keras.Sequential([
    # Increase image size to 256x256
    tf.keras.layers.Input(shape=(256, 256, 3)),

    # Add convolutional layers with max pooling
    # I've had better results with 32, 64, 128, but the model size get's big quick doing things like that.
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten the output for dense layers
    tf.keras.layers.Flatten(),

    # Add dense layers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    # Output layer with sigmoid activation for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks for training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with callbacks
history = model.fit(X_train, y_train, epochs=32, batch_size=24,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stopping])

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Save the final trained model
model.save('final_model.h5')
print("Final model saved as 'final_model.h5'")
