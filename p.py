import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('facefeatures_new_model2.h5')

# Dictionary of class labels
class_labels = {
    0: 'Akshay',
    1: 'Alexanda',
    2: 'Alia',
    3: 'Amitav',
    4: 'Andy',
    5: 'Anuxka',
    6: 'Bili',
    7: 'camilia',
    8: 'Zack',
    # Add more class labels as needed
}

# Load and preprocess the image
image_path = "D:\\CU\\face_rec\\Dataset\\a.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))  # Resize if needed
image = image / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Perform prediction
predictions = model.predict(image)
print(predictions)
class_indices = np.argmax(predictions, axis=1)

# Get the predicted class labels
predicted_labels = [class_labels[idx] for idx in class_indices]

print(f"Predicted classes: {predicted_labels}")
