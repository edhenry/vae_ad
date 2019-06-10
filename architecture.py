import importlib

import numpy as np
import tensorflow as tf
from keras.layers import Activation, Dense, Input
from keras.models import Model, Sequential

class VAE(object):
    """
    Variational Autoencoder

    This object is composed of an Encoder and a Decoder object
    """
    def __init__(self, input_data: np.ndarray):
        
        inputs = Input(shape=(784,))
        z_inputs = Input(shape=(50,))

        input_data = tf.convert_to_tensor(input_data)
        
        self.ae = self.model(input_data)

    def model(self, input_data: np.ndarray):
        """
        x -> mu, log_sigma_sq -> N(mu, log_sigma_sq) -> Z -> x
        """
        
        self.input = input_data
        self.z = tf.Variable(tf.zeros(shape=[1,50], dtype='float32'))

        #TODO Add modularity for encoder architecture
        # instead of using a vanilla MLP like outlined
        # below, we may want to use something like a CNNVAE
        with tf.variable_scope("enc_input_hidden_1"):
            self.x = Dense(300, activation='relu', input_dim=784)(self.input)
        with tf.variable_scope("enc_hidden_1_hidden_2"):
            self.x = Dense(200, activation='relu', input_dim=300)(self.x)
        with tf.variable_scope("enc_hidden_2_hidden_3"):
            self.x = Dense(100, activation='relu', input_dim=200)(self.x)

        # Borrowed from https://github.com/tegg89/VAE-Tensorflow/blob/master/model.py
        encoder_mu_weights = tf.Variable(tf.random_normal([100, 30], stddev=0.1), name='encoder_mu_weights')
        encoder_sigma_weights = tf.Variable(tf.random_normal([100, 30], stddev=0.1), name='encoder_sigma_weights')

        encoder_mu_bias = tf.Variable(tf.zeros([30]), name="encoder_mu_bias")
        encoder_sigma_bias = tf.Variable(tf.zeros([30]), name="encoder_sigma_bias")

        with tf.variable_scope("encoder_mu"):
            encoder_mu = tf.matmul(self.x, encoder_mu_weights) + encoder_mu_bias

        with tf.variable_scope("encoder_sigma_bias"):
            encoder_sigma = tf.matmul(self.x, encoder_sigma_weights) + encoder_sigma_bias

        # Sample an epsilon and generate a z from the latent space provided by the encoder
        # as outlined in "Autoencoding Variational Bayes" : Kingma et al.
        # http://arxiv.org/abs/1312.6114
        epsilon = tf.random_normal(tf.shape(encoder_sigma), name='epsilon')
      
        sample_encoder = tf.exp(0.5 * encoder_sigma)

        kl_divergence = -0.5 * tf.reduce_sum(1. + encoder_sigma - tf.pow(encoder_mu, 2) - tf.exp(encoder_sigma), reduction_indices=1)

        self.z = encoder_mu + tf.multiply(sample_encoder, epsilon)

        with tf.variable_scope("enc_hidden_3_z"):
            self.z = Dense(50, activation='relu', input_dim=100)(self.z)
        with tf.variable_scope("dec_z_hidden_1"):
            self.z = Dense(50, activation='relu', input_dim=50)(self.z)
        with tf.variable_scope("dec_hidden_1_hidden_2"):
            self.z = Dense(100, activation='relu', input_dim=50)(self.z)
        with tf.variable_scope("dec_hidden_2_hidden_3"):
            self.z = Dense(200, activation='relu', input_dim=100)(self.z)
        with tf.variable_scope("dec_hidden_3_hidden_4"):
            self.z = Dense(300, activation='relu', input_dim=200)(self.z)
        with tf.variable_scope("dec_hidden_4_reconstruction"):
            self.reconstruction = Dense(784, activation='relu', input_dim=300)(self.z)
        
        binary_cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.reconstruction, labels=self.input), reduction_indices=1)

        self.loss = tf.reduce_mean(kl_divergence + binary_cross_entropy)

        tf.print(self.loss)

        return self.reconstruction, self.loss
    
    #TODO create loss function method for future inheritance / class extension
    # def loss_function(self, reconstructed_x, x):
    #     """
    #     Loss function as defined in "Autoencoding Variational Bayes" : Kingma, et al.

    #     http://arxiv.org/abs/1312.6114
                
    #     Arguments:
    #         reconstructed_x {np.ndarray} -- reconstruction of input x
    #         x {np.ndarray} -- original input x
    #     """

    #     binary_cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(reconstructed_x, x), reduction_indices=1)

    #     #kl_divergence = -0.5 * tf.reduce_sum(1. + self.Encoder.)
