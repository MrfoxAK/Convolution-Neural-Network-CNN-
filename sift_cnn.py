import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

# Load SIFT features from file or generate your own SIFT features
sift_features = np.load("D:\\CU\\face_rec\\sift_features.npy") # Example random SIFT features, replace with your own data

# Reshape the SIFT features to match the expected input shape of VGG16
sift_features = sift_features.reshape((128, 1, 1))

# Create VGG16 model
model = Sequential()
model.add(VGG16(weights='imagenet', include_top=False, input_shape=(128, 1, 1)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='softmax'))  # Assuming 1000 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
labels = np.random.randint(0, 1000, (78709,))  # Example random labels, replace with your actual label data
model.fit(sift_features, labels, epochs=10, batch_size=32)
