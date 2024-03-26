import numpy as np 
import pandas as pd
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

directory = r"Agricultural-crops"

def get_folder_names(directory_path):
    folder_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    return folder_names

directory_path = directory
categories = get_folder_names(directory_path)

img_data = []

for catago in categories:
    folder = os.path.join(directory, catago)
    label = categories.index(catago)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        try:
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (100, 100))
            if img_array is not None and not img_array.size == 0:
                img_data.append([img_array, label])
            else:
                print(f"Error loading or resizing image: {img_path}")
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")

random.shuffle(img_data)

X = []
Y = []
for features, labels in img_data:
    X.append(features)
    Y.append(labels)
    
X = np.array(X)
Y = np.array(Y)
X = X / 255

from tensorflow.keras.utils import to_categorical
Y = to_categorical(Y, num_classes=30)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Update the model
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=X_train.shape[1:], activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(units=30, activation="softmax"))
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
cnn.fit(X_train, Y_train, epochs=10, validation_split=0.2)

# Make predictions on test data
Y_pred = cnn.predict(X_test)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Display classification report
print(classification_report(Y_test_classes, Y_pred_classes, target_names=categories))


# # Load and resize the image
# test = cv2.imread('Cherry_season.jpg')
# test = cv2.resize(test, (100, 100))

# # Ensure image has 3 color channels
# test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

# # Normalize pixel values to be between 0 and 1
# test = test / 255.0

# # Add batch dimension
# test = np.expand_dims(test, axis=0)

# # Perform prediction
# value = cnn.predict(test)

# output = np.argmax(value)

