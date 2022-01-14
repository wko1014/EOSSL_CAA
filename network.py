# Import APIs
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class E1(tf.keras.Model):
    """
    The spatio-spectro characteristics representer
    """
    def __init__(self, Fs, Nc, Nt):
        super(E1, self).__init__()

        conv = lambda D, kernel: layers.SeparableConv2D(
            D, kernel, padding='valid', depthwise_regularizer='L1L2', pointwise_regularizer='L1L2', activation='elu'
        )

        # Encoder part
        self.E1 = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(Nc, Nt, 1)),
                # spectral convolution
                conv(4, (1, int(Fs/2))),
                # spatial convolution
                conv(8, (Nc, 1)),
                # temporal convolutions
                conv(16, (1, 30)),
                conv(32, (1, 15)),
                conv(64, (1, 5)),
            ]
        )

        self.C1 = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(64,)),
                # stopped band prediction is five-class classification
                layers.Dense(units=5, activation=None, kernel_regularizer='L1L2'),
            ]
        )

    @tf.function
    def spectral_embedding(self, x):
        return self.E1(x)

    def band_prediction(self, f):
        f = tf.math.reduce_mean(f, axis=-2) # Global average pooling into temporal dimension
        f = tf.squeeze(f)
        return self.C1(f)

class E2(tf.keras.Model):
    """
    The spatio-temporal dynamics representer
    """
    def __init__(self, Nc, Nt):
        super(E2, self).__init__()
        # Encoder part
        conv = lambda D, kernel, pad: layers.Conv2D(
            D, kernel, padding=pad, kernel_regularizer='L1L2', activation='elu'
        )

        # Define layers
        x = layers.Input(shape=(Nc, Nt, 1))
        T1, S1 = conv(8, (1, 30), 'same'), conv(8, (Nc, 1), 'valid')
        T2, S2 = conv(16, (1, 15), 'same'), conv(16, (Nc, 1), 'valid')
        T3, S3 = conv(32, (1, 5), 'same'), conv(32, (Nc, 1), 'valid')
        f1 = T1(x)
        f2 = T2(f1)
        f3 = T3(f2)
        # Multi-scale representation structure
        f_multi = layers.concatenate((S1(f1), S2(f2), S3(f3)))
        self.E2 = tf.keras.Model(x, f_multi)

        self.C2 = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(56,)),
                # stationary condition detection is four-class classification
                layers.Dense(units=4, activation=None, kernel_regularizer='L1L2'),
            ]
        )

    @tf.function
    def temporal_embedding(self, x):
        return self.E2(x)

    def stationary_detection(self, f):
        f = tf.math.reduce_mean(f, axis=-2) # Global average pooling into temporal dimension
        f = tf.squeeze(f)
        return self.C2(f)

class ALN(tf.keras.Model):
    """
    The adaptive layer normalization
    """
    def __init__(self, f1, f2):
        self.mu = tf.concat((tf.math.reduce_mean(f1, -2), tf.math.reduce_mean(f2, -2)), -1)
        self.f_concat = self.mu # Similar to GAP on the temporal dimension!
        self.sigma = tf.concat((tf.math.reduce_std(f1, -2), tf.math.reduce_std(f2, -2)), -1)

    @tf.function
    def statistics(self):
        return self.mu, self.sigma

    def normalization(self, mu_star, sigma_star):
        return (self.f_concat - mu_star)/sigma_star

class C(tf.keras.Model):
    """
    The classifier
    """
    def __init__(self, No):
        super(C, self).__init__()
        # Define layer
        self.C = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(120,)),
                layers.Dense(units=No, activation=None, kernel_regularizer='L1L2'),
            ]
        )

    @tf.function
    def classification(self, f_ALN):
        return self.C(tf.squeeze(f_ALN))