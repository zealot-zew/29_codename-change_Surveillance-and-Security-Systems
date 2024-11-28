import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def ML_prediction(X):
    # Labels for camera placement at each node: 1 = camera, 0 = no camera
    y = np.array([1, 1, 1, 1])  # Place cameras at all nodes for simplicity

    # Define the model
    model = Sequential()

    # Input layer (number of nodes as features)
    model.add(Dense(32, input_dim=X.shape[1], activation='relu'))

    # Hidden layer
    model.add(Dense(16, activation='relu'))

    # Output layer (binary: camera at node or not)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=50, batch_size=2)

    # Predict camera placement for each node
    predictions = model.predict(X)
    print(predictions)  # Outputs values between 0 and 1, with values close to 1 indicating camera placement