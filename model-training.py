import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to a fixed size
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

compliant_images, compliant_labels = load_images_from_folder(r'C:\Users\HP\OneDrive\Documents\CODING\complaint_uniforms', 1)
non_compliant_images, non_compliant_labels = load_images_from_folder(r'C:\Users\HP\OneDrive\Documents\CODING\non_complaint uniforms', 0)

# Combine and preprocess data
X = np.concatenate((compliant_images, non_compliant_images), axis=0)
y = np.concatenate((compliant_labels, non_compliant_labels), axis=0)

X = X / 255.0  # Normalize pixel values
y = tf.keras.utils.to_categorical(y, 2)  # One-hot encode labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),  # Flatten the output to feed into dense layers
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('model.h5')