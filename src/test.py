import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load a saved model
model = tf.keras.models.load_model('final_model.h5')


# Function to load and preprocess a single image
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
            print(f"Resized image shape for {filename}: {resized_image.shape}")
            # Convert the image to numpy array and normalize pixel values to [0, 1]
            image = resized_image / 255.0
            # Append the preprocessed image to the list
            images.append(image)
    return np.array(images)


# Path to the image you want to test
test_directory = 'resources/imagesets/testing'

# Load and preprocess images from the test directory
test_images = load_and_preprocess_images(test_directory)

# Get the filenames of images in the test directory
test_filenames = [filename for filename in os.listdir(test_directory) if filename.endswith(".jpg") or filename.endswith(".png")]

# Make predictions using the loaded model
predictions = model.predict(test_images)

# Create a list of tuples containing image, filename, predicted class, and confidence
image_info_list = []
for image, filename, prediction in zip(test_images, test_filenames, predictions):
    predicted_class = "Artificial" if prediction[0] > 0.5 else "Real"
    confidence = abs(prediction[0] - 0.5) * 2 * 100  # Normalize confidence to percentage
    image_info_list.append((image, filename, predicted_class, confidence))

# Function to display images with their information
def display_images(image_info_list):
    num_images = len(image_info_list)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))

    for i, (image, filename, predicted_class, confidence) in enumerate(image_info_list):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]
        ax.imshow(image)
        ax.set_title(filename, fontsize=10)
        ax.axis('off')
        ax.text(10, 25, f"{predicted_class}\nConfidence: {confidence:.2f}%", fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))

    # Hide any empty subplots
    for i in range(len(image_info_list), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Display images with their information
display_images(image_info_list)