import os
import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("D:\\CU\\face_rec\\facefeatures_new_model2.h5")

# Directory containing the images to test
image_directory = "D:\\CU\\face_rec\\t"

# Iterate over all files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read and preprocess the image
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # Resize if needed
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform inference
        predictions = model.predict(image)
        print(predictions)
        class_index = np.argmax(predictions[0])
        class_name = 'Test'  # Replace with your own class names
        if class_index == 0:
            class_name = 'Akash'
        elif class_index == 1:
            class_name = 'Debojyoti'
        # Add more elif statements for additional classes

        print(f"Image: {filename} - Predicted class: {class_name}")
