"""
File contains the Deep Neural Network Model.
Author: Hari Vidharth, S4031180
"""

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DDQNModel(object):
    def buildModel(self, learningRate, nActions, inputShape, fc1Dims, fc2Dims):
        """
        The neural network model architecture for training the agent.
        """
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=tf_config)
        tf.compat.v1.keras.backend.set_session(sess)
        model = Sequential()
        model.add(Dense(fc1Dims, input_shape=(inputShape,), activation="relu"))
        model.add(Dense(fc2Dims, activation="relu"))
        model.add(Dense(nActions))
        model.compile(optimizer=Adam(learning_rate=learningRate), loss="mse")
        model.summary()

        return model
