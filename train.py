import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load the capital letters dataset
alphabet_data = pd.read_csv("data/A_Z Handwritten Data.csv").astype("float32")

# Rename the label column
alphabet_data.rename(columns={'0': "label"}, inplace=True)

# Remove ambiguous labels of O and I which are not used in VIN
alphabet_data = alphabet_data[(alphabet_data["label"] != 8) & (alphabet_data["label"] != 14)]

# Separate the features from the labels
alphabet_x = alphabet_data.drop("label", axis=1).values
alphabet_y = alphabet_data["label"]

# Load the digits dataset
(digits_train_x, digits_train_y), (digits_test_x, digits_test_y) = keras.datasets.mnist.load_data()

# Merge the two datasets
digits_data = np.concatenate((digits_train_x, digits_test_x))
digits_target = np.concatenate((digits_train_y, digits_test_y))
digits_target += 26

# Reshape the data to be compatible with the neural network
digits_data = np.reshape(digits_data, (digits_data.shape[0], digits_data.shape[1], digits_data.shape[2], 1))
alphabet_data = np.reshape(alphabet_x, (alphabet_x.shape[0], 28, 28, 1))

# Combine the data and labels into one array
data = np.concatenate((digits_data, alphabet_data))
target = np.concatenate((digits_target, alphabet_y))

# Split the data into training, validation, and testing sets
train_ratio = 0.8
validation_ratio = 0.18
test_ratio = 0.02

train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=1 - train_ratio)

val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=test_ratio/(test_ratio + validation_ratio))

# Normalize the input data
train_data = train_data / 255.0
val_data = val_data / 255.0
test_data = test_data / 255.0

# Convert the labels to one-hot encoding
train_labels = keras.utils.to_categorical(train_labels)
val_labels = keras.utils.to_categorical(val_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Define the model architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (5, 5), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(36, activation="softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Save the best version of the model to disk
best_val_loss_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="model1.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min"
)

# Train the model
training = model.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels), 
    batch_size=256, 
    callbacks=[best_val_loss_checkpoint]
)

# Save model with the smallest validation loss
model.load_weights("models/best_val_loss_model.h5")
model.save('/content/drive/MyDrive/Internship/models/model.h5')

