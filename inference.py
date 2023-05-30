import os
import argparse
import numpy as np
import cv2
import tensorflow as tf

# create argument for the script for specifying the folder with the images
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Path to the input folder with data")
args = parser.parse_args()

# if input folder argument is specified
if args.input:
    # supress warnings
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.get_logger().setLevel('ERROR')

    # load pre-trained model
    model_path = os.path.join(os.getcwd(), "model.h5")
    model = tf.keras.models.load_model(model_path)

    # create dictionary for character labels
    alphabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 9: 'J', 10: 'K', 11: 'L',
        12: 'M', 13: 'N', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
        23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
    asc = {key: ord(value) for key, value in alphabet.items()}

    # image preprocessing, converts image to the format used by NN
    def preprocess_image(image):
        resized_image = cv2.resize(image, (28, 28))
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        normalized_image = grayscale_image / 255.0
        preprocessed_image = np.reshape(normalized_image, (28, 28, 1))
        return preprocessed_image
    
    # for each image within the input directory predict the VIN character
    for filename in os.listdir(args.input):
        file_path = os.path.join(args.input, filename)
        if os.path.isfile(file_path) and any(file_path.lower().endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.gif']):
            image = cv2.imread(file_path)
            image = preprocess_image(image)
            prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
            predicted_class = np.argmax(prediction, axis=-1)
            predicted_class = int(predicted_class)
            print('0' + str(asc[predicted_class]), ", ", file_path, sep="")

# if no input folder is specified
else:
    print("Error: No input folder specified")
