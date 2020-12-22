import numpy as np
import os
import cv2
import math
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

class ImageModel:
    def __init__(self, path):
        self.path = path
        self.dec = load_model("models/image_dec_vgg16.h5")
        with open('models/image_reverse_word_map.json') as json_file:
            self.reverse_word_map = json.load(json_file)

    def preprocess_image(self):
        test_image = plt.imread(self.path)
        test_image = cv2.resize(test_image, (224, 224))

        X = []
        X.append(test_image)
        X = np.array(X)

        return X

    def predict_sequence(self, X, max_tokens = 50):
        """
        To make predictions
        """
        model = VGG16(weights='imagenet', include_top=True)
        feature_extractor = Model(model.input, model.get_layer('fc2').output)
        # Process the image with the pre-trained image-model to get the transfer-values....
        transfer_values_test = feature_extractor.predict(X)

        shape = (1, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)

        # The first input-token is the special start-token for 'ssss '.
        token_int = 2

        # Initialize an empty output-text.
        output_text = ''

        # Initialize the number of tokens we have processed.
        count_tokens = 0

        list_of_indices = []

        token_end = 3

        while token_int != token_end and count_tokens < max_tokens:
            # Update the input-sequence to the decoder with the last token that was sampled....
            decoder_input_data[0, count_tokens] = token_int

            # Wrap the input-data in a dict for clarity and safety,
            # so we are sure we input the data in the right order.
            x_data = \
            {
                'transfer_values_input': transfer_values_test,
                'decoder_input': decoder_input_data
            }

            # Input this data to the decoder and get the predicted output....
            decoder_output = self.dec.predict(x_data)

            # Get the last predicted token as a one-hot encoded array....
            token_onehot = decoder_output[0, count_tokens, :]

            # Convert to an integer-token.....
            token_int = np.argmax(token_onehot)

            list_of_indices.append(token_int)

            # Increment the token-counter.....
            count_tokens += 1

        # This is the sequence of tokens output by the decoder....
        words = [self.reverse_word_map.get(str(word)) for word in list_of_indices if word]
        output_text = ' '.join(words)
        return output_text

    def get_pred(self):
        X = self.preprocess_image()

        output = self.predict_sequence(X)

        return output
