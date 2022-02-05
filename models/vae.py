"""
References for implementation:
    Generative Deep Learning by David Foster, Chapter 3
    https://keras.io/examples/generative/vae/
    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#going_lower-level
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def construct_encoder(input_dim, latent_dim, conv_filters, kernel_sizes):
    """
    Construct an encoder network according to the given specifications.
    input_dim, conv_filters and kernel_sizes are lists of the same length.
    """
    encoder_input = layers.Input(shape=input_dim)
    x = encoder_input

    for i in range(len(conv_filters)):
        conv_layer = layers.Conv2D(
            filters=conv_filters[i],
            kernel_size=kernel_sizes[i],
            strides=2,
            padding='same'
        )
        x = conv_layer(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        # x = layers.Dropout(0.25)(x)

    shape_before_flattening = x.shape[1:]

    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_sigma = layers.Dense(latent_dim, name='z_log_sigma')(x)

    encoder = tf.keras.Model(
        encoder_input, [z_mean, z_log_sigma], name='Encoder'
    )
    return encoder, shape_before_flattening


def construct_decoder(
        shape_before_flattening, latent_dim, conv_filters, kernel_sizes
        ):
    """
    Construct a decoder network according to the given specifications.
    shape_before_flattening is the second
    output of the construct_encoder method.
    """
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)

    for i in reversed(range(len(conv_filters))):
        conv_t_layer = layers.Conv2DTranspose(
            filters=conv_filters[i],
            kernel_size=kernel_sizes[i],
            strides=2,
            padding='same'
        )
        x = conv_t_layer(x)

        if i != 0:
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            # x = layers.Dropout(0.25)(x)
        else:
            x = layers.Activation('sigmoid')(x)

    decoder = tf.keras.Model(decoder_input, x, name='Decoder')

    return decoder


class VAE(tf.keras.Model):
    def __init__(
            self,
            input_dim=(128, 128, 3),
            latent_dim=200,
            enc_conv_filters=[32, 64, 64, 64],
            dec_conv_filters=[3, 32, 64, 64],
            kernel_sizes=[3, 3, 3, 3],
            kl_weight=0.0001
            ):
        super().__init__()

        self.encoder, self.flat_shape = construct_encoder(
            input_dim, latent_dim, enc_conv_filters, kernel_sizes
        )
        self.decoder = construct_decoder(
            self.flat_shape, latent_dim, dec_conv_filters, kernel_sizes
        )
        self.kl_weight = kl_weight
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_div_loss')
        self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.kl_loss_tracker,
            self.recon_loss_tracker
        ]

    @staticmethod
    def sample_z(z_mean, z_log_sigma):
        """
        Sample from a Gaussian distribution.
        Arguments: z_mean, z_log_sigma (tensor): mean and log
        of standard deviation of latent distribution (Q(z|X)).
        Returns: z (tensor): sampled latent vector.
        """
        batch = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        # Reparameterize using epsilon.
        epsilon = tf.random.normal(shape=(batch, latent_dim))

        z = z_mean + tf.math.exp(0.5 * z_log_sigma) * epsilon
        return z

    @staticmethod
    def loss_function(x, x_recon, z_mean, z_log_sigma, kl_weight):
        """
        The loss function for VAE.
        Arguments: An input x, reconstructed output x_recon,
        encoded mean z_mean, encoded log of standard deviation z_log_sigma,
        and weight parameter for the latent loss kl_weight.
        """
        # The reconstruction loss (L2).
        recon_loss = tf.reduce_mean(tf.square(x - x_recon), axis=(1, 2, 3))

        # The KL-divergence loss.
        kl_div_loss = 0.5 * tf.reduce_sum(
            tf.exp(z_log_sigma) + tf.square(z_mean) - 1.0 - z_log_sigma, axis=1
        )

        vae_loss = recon_loss + kl_weight * kl_div_loss
        return vae_loss, kl_div_loss, recon_loss

    def call(self, x):
        """
        Forward pass of the model.
        """
        # Given an input x, find mean and std of the latent variables.
        z_mean, z_log_sigma = self.encoder(x)
        # Sample from the latent variable distributions.
        z = VAE.sample_z(z_mean, z_log_sigma)
        # Reconstruct the output from the sampled value z.
        x_recon = self.decoder(z)

        return z_mean, z_log_sigma, x_recon

    def train_step(self, data):
        """
        Custom training step.
        """
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            z_mean, z_log_sigma, x_recon = self(x, training=True)
            total_loss, kl_div_loss, recon_loss = VAE.loss_function(
                x, x_recon, z_mean, z_log_sigma, self.kl_weight)

        # Compute and update gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_div_loss)
        self.recon_loss_tracker.update_state(recon_loss)

        return {
            'total_loss': self.total_loss_tracker.result(),
            'kl_div_loss': self.kl_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result()
        }
