import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split


EPOCHS = 20
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python maskverified.py data_directory [model.h5]")

    # Get resized images and their respective category (0 or 1)
    images, D = load_data(sys.argv[1])
    # Split data into training and testing sets
    D = tf.keras.utils.to_categorical(D)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(D), test_size=TEST_SIZE
    )

    # Compile model
    model = get_model()

    # Fit model to training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate model on
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    """
    Load images into an image list and their respective category into a determiner list (D)
    Return tuple (image list, determiner list)
    """
    images = []
    D = []
    # Go through each directory in data_dir named after each category
    for dir in [0, 1]:
        tempimgs = []
        # For each file in each dir
        for folder in os.listdir(os.path.join(f'{data_dir}', f'{dir}')):
            for filename in os.listdir(os.path.join(f'{data_dir}', f'{dir}', f'{folder}')):
                # Read it using cv2, resize it, then format it as an array
                if filename[-4:] == ".jpg":
                    img = cv2.imread(os.path.join(f'{data_dir}', f'{dir}', f'{folder}', filename), cv2.IMREAD_UNCHANGED)
                    array = np.asarray(cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA))
                    # Add the image and label to their corresponding lists
                    tempimgs.append(array)

        # Ratio of mask to no mask is too big (about 30)
        # If model predicts innacurately, it still has a high chance of predicting right so even out ratio of mask:no mask
        if dir == 0:
            for img in random.sample(tempimgs, 3000):
                images.append(img)
                D.append(dir)
        else:
            for img in tempimgs:
                images.append(img)
                D.append(dir)

    # Return the images list and the corresponding D list
    return images, D


def get_model():
    """
    The model to fit the data to
    Input layer: (30, 30, 3) array
    Output layer: weighted numbers for case 0 (no mask) and case 1 (wearing mask) of that specific input
    """
    model = tf.keras.models.Sequential([
        # Input layer 32 3x3 kernels
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="sigmoid", input_shape=(30, 30, 3)
        ),
        # Maxpooling 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Hidden Dropout
        tf.keras.layers.Dropout(.15),

        # Flatten
        tf.keras.layers.Flatten(),

        # Dense layer --> output layer
        tf.keras.layers.Dense(32, activation="sigmoid"),
        # Output layer
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.summary()
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
