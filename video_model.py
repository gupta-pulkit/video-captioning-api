import numpy as np
import os
import cv2
import math
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

class VideoModel:
    def __init__(self, path):
        self.path = path
        self.enc = load_model("models/video_enc.h5")
        self.dec = load_model("models/video_dec.h5")
        with open('models/reverse_word_map.json') as json_file:
            self.reverse_word_map = json.load(json_file)

    def extract_frames(self):
        """
        Extract the frames from the videos and store the extracted frames as images in jpg format

        videos_name: list of video names whose frames are to be extracted
        source_path: path to the videos folder
        target_path: path to the target folder where frames are stored
        """
        source_path = self.path
        model = InceptionV3(weights='imagenet', include_top=True)
        feature_extractor = Model(model.input, model.get_layer('avg_pool').output)

        print("Extracting frames")
        count = 0
        video_captured = cv2.VideoCapture(source_path)
        l = []

        while(video_captured.isOpened()):
            frameId = video_captured.get(1)
            ret, frame = video_captured.read()

            if ret != True:
                break

            if count == 15:
                break

            if frameId % 10 == 0:
                filename = "frame" + str(count) + ".jpg"
                count += 1
                img_data = cv2.resize(frame, (299, 299))
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input_inception_v3(img_data)

                features = feature_extractor.predict(img_data)
                features = features.flatten()
                l.append(features)

        video_captured.release()

        print("Frames extracted")

        X = []
        X.append(l)
        X = np.array(X)

        return X

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(self, list_of_indices):
        """
        To convert numerical sequences to list of words
        """
        # Looking up words in dictionary
        words = [self.reverse_word_map.get(str(idx)) for idx in list_of_indices if idx and idx not in [3, 4]]
        return(words)

    def predict_sequence(self, source, n_steps = 20, cardinality = 2400):
        """
        To make predictions
        """
        # encode
        state = self.enc.predict(source)
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # collect predictions
        output = list()
        for t in range(n_steps):
            # predict next char
            yhat, h, c = self.dec.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat

        out = np.array(output).argmax(axis = 1)

        return ' '.join(self.sequence_to_text(out))

    def get_pred(self):
        X = self.extract_frames()

        output = self.predict_sequence(X)

        return output
