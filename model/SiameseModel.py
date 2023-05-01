from tensorflow.keras.applications import resnet
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import cv2
import os

class SiameseModel(keras.Model):

    def __init__(self, network=None, image_size=(224, 224), margin=0.5, threshold=0.5):
        super().__init__()
        # save params
        self.margin = margin
        self.threshold = threshold
        self.unknown_name = "unknown"
        self.image_size, self.image_dim = image_size, image_size + (3,)
        # build siamese network
        self.network = network if network is not None else self._get_siamese_network()
        self.tracker = keras.metrics.Mean(name="loss")
        # initialize database dict
        self.registered_faces = {}
        
    # --------------- PUBLIC API ------------------\

    def register(self, input_image, label):
        # extract embedding vector from input image & save to database
        # @input_image: normalized face image (as numpy array)
        # @label: person name
        # return None
        image_tensor = self._preprocess_numpy_image(input_image)
        embedding_vector = self.network(image_tensor)
        self.registered_faces[label] = embedding_vector

    def recognize_face(self, input_image, threshold=0.8, verbose=False):
        # recognize face on input_image based on registered faces 
        # @input_image: normalized face image (as numpy array)
        # return: if distance <= threshold, return a label from database, otherwise "unknown"
        image_tensor = self._preprocess_numpy_image(input_image)
        input_vector = self.network(image_tensor)
        min_dist, min_name = 1000, self.unknown_name
        for name, registered_vector in self.registered_faces.items():    
            distance = self._calculate_distance(registered_vector, input_vector)
            if verbose:
                print(f"dist:{distance}, name:{name}")
            if distance <= min_dist:
                min_dist = distance
                min_name = name
        min_name = min_name if min_dist <= threshold else self.unknown_name
        print(f"prediction: {min_name}")
        return min_name

    # ---------------------------------------------

    def _preprocess_numpy_image(self, numpy_image):
        image_tensor = tf.convert_to_tensor(numpy_image)
        image_tensor = tf.expand_dims(image_tensor , 0)
        image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
        image_tensor = tf.image.resize(image_tensor, self.image_size)
        return image_tensor

    def _calculate_distance(self, anchor_vector, target_vector):
        distance = tf.reduce_sum(tf.square(anchor_vector - target_vector), axis=-1)
        return distance

    def _compute_triplet_distances(self, inputs):
        (anchor, positive, negative) = inputs
        # embed the images using the siamese network        
        anchorEmbedding = self.network(anchor)
        positiveEmbedding = self.network(positive)
        negativeEmbedding = self.network(negative)

        # calculate the anchor to positive and negative distance
        apDistance = tf.reduce_sum(
            tf.square(anchorEmbedding - positiveEmbedding), axis=-1
        )
        anDistance = tf.reduce_sum(
            tf.square(anchorEmbedding - negativeEmbedding), axis=-1
        )

        # return the distances
        return (apDistance, anDistance)

    def _compute_loss(self, apDistance, anDistance):
        loss = apDistance - anDistance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def _get_siamese_network(self):
        inputs = keras.Input(self.image_dim)
        x = resnet.preprocess_input(inputs)

        # fetch the pre-trained resnet 50 model and freeze the weights
        baseCnn = resnet.ResNet50(weights="imagenet", include_top=False)
        baseCnn.trainable=False

        # pass the pre-processed inputs through the base cnn and get the
        # extracted features from the inputs
        extractedFeatures = baseCnn(x)

        # pass the extracted features through a number of trainable layers
        x = layers.GlobalAveragePooling2D()(extractedFeatures)
        x = layers.Dense(units=1024, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units=256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(units=128)(x)

        # build the embedding model and return it
        embedding = keras.Model(inputs, outputs, name="embedding")
        return embedding

    def call(self, inputs):
        # compute the distance between the anchor and positive,
        # negative images
        (apDistance, anDistance) = self._compute_triplet_distances(inputs)
        return (apDistance, anDistance)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # compute the distance between the anchor and positive,
            # negative images
            (apDistance, anDistance) = self._compute_triplet_distances(inputs)

            # calculate the loss of the siamese network
            loss = self._compute_loss(apDistance, anDistance)

        # compute the gradients and optimize the model
        gradients = tape.gradient(
            loss,
            self.network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_variables)
        )

        # update the metrics and return the loss
        self.tracker.update_state(loss)
        return {"loss": self.tracker.result()}

    def test_step(self, inputs):
        # compute the distance between the anchor and positive,
        # negative images
        (apDistance, anDistance) = self._compute_triplet_distances(inputs)

        # calculate the loss of the siamese network
        loss = self._compute_loss(apDistance, anDistance)

        # update the metrics and return the loss
        self.tracker.update_state(loss)
        return {"loss": self.tracker.result()}

    @property
    def metrics(self):
        return [self.tracker]