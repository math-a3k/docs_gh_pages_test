# -*- coding: utf-8 -*-
HELP="""Keras VQ-VAE-2.ipynb

Original file is located at
    https://colab.research.google.com/drive/1dsRcxGDtLIY_ZVeiyPyF_HQQrZlp19TP
#!pip install -q tensorflow-probability

"""
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
from keras.layers.merge import concatenate




############################################################################    
def test_vqvae2():
    """Loading/ processing the dataset and then training and plotting"""

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #since we dont need the labels, we will discard them

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    data_variance = np.var(x_train / 255.0)

    #creating instances of the VQ-VAE and compiling the model
    vq_vae_trainer = VQ_VAE_Trainer_2(data_variance, latent_dim=16, number_of_embeddings=128)
    vq_vae_trainer.compile(optimizer=keras.optimizers.Adam())
    vq_vae_trainer.fit(x_train_scaled, epochs=30, batch_size=128)

    #ploting the reconstruced images
    trained_vqvae_model = vq_vae_trainer.vqvae
    idx = np.random.choice(len(x_test_scaled), 10)
    test_images = x_test_scaled[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        plot_original_reconstructed(test_image, reconstructed_image)

    vq_vae_trainer.vqvae.summary()

    encoder = vq_vae_trainer.vqvae.get_layer("encoder")
    quantizer = vq_vae_trainer.vqvae.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")
    
    
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vq_vae_trainer.number_of_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
    )(ohe)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    out = keras.layers.Conv2D(
        filters=vq_vae_trainer.number_of_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")

    # Generating the codebook indices.
    encoded_outputs = encoder.predict(x_train_scaled)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
    )

    # Creating a sampler 
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    x = pixel_cnn(inputs, training=False)
    dist = tfp.distributions.Categorical(logits=x)
    sampled = dist.sample()
    sampler = keras.Model(inputs, sampled)

    # Empty array representing priors.
    batch = 10
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    print(f"Prior shape: {priors.shape}")

    # Perform an embedding lookup.
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vq_vae_trainer.number_of_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(28, 28, 1)))
    # Generate novel images.
    decoder = vq_vae_trainer.vqvae
    generated_samples = decoder.predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()






#################################################################################################
#################################################################################################
class Quantizer(layers.Layer):
    def __init__(self, number_of_embeddings, embedding_dimensions, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dimensions = embedding_dimensions
        self.number_of_embeddings = number_of_embeddings
        self.beta = (
            beta  
            # This parameter are set as described int the paper
        )

        # Initializing the embeddings for quantization
        initializer  = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=initializer (
                shape=(self.embedding_dimensions, self.number_of_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_for_vq_vae",
        )

    def call(self, x):
        # flattening the input
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dimensions])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.number_of_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Computing quantization loss 
        
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        # adding it to the layer. 

        self.add_loss(commitment_loss)
        self.add_loss(codebook_loss)

        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

def encoder_Base(latent_dim):
  encoder_A_inputs = keras.Input(shape=(28, 28, 1))
  Encoder_A = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_A_inputs)
  Encoder_A = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(Encoder_A)
  encoder_A_outputs = layers.Conv2D(latent_dim, 1, padding="same")(Encoder_A)
  return keras.Model(encoder_A_inputs, encoder_A_outputs, name="encoder")


def get_vqvae_layer_hierarchical(latent_dim=16, num_embeddings=64):
  vq_layer = Quantizer(num_embeddings, latent_dim, name="vector_quantizer")

  
  encoder_A_inputs = keras.Input(shape=(28, 28, 1))
  encoder = encoder_Base(latent_dim)
  encoder_A_outputs = encoder(encoder_A_inputs)

  
  Encoder_B = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(encoder_A_outputs)
  Encoder_B = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(Encoder_B)
  encoder_B_outputs = layers.Conv2D(latent_dim, 3, padding="same", name="encoder_B")(Encoder_B)

  quantized_latents_b = vq_layer(encoder_A_outputs)
  quantized_latents_t = vq_layer(encoder_B_outputs)


  
  Decoder_T = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(quantized_latents_t)
  Decoder_T = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(Decoder_T)
  decoder_T_outputs = layers.Conv2DTranspose(1, 3, padding="same", name="decoder_B")(Decoder_T)

  Decoder_B = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(quantized_latents_b)
  Decoder_B = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(Decoder_B)
  decoder_B_outputs = layers.Conv2DTranspose(1, 3, padding="same", name="decoder_A")(Decoder_B)
  
  reconstructions_of_t = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(decoder_T_outputs)
  reconstructions_of_t = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(reconstructions_of_t)

  concat = tf.keras.layers.Concatenate(axis=-1)([reconstructions_of_t,quantized_latents_b])
  Decoder_B = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(concat)
  Decoder_B = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(Decoder_B)
  decoder_B_outputs = layers.Conv2DTranspose(1, 3, padding="same")(Decoder_B)
  
  return keras.Model(encoder_A_inputs, decoder_B_outputs, name="decoder")


def plot_original_reconst_img(orig, rec):   #name changed
    plt.subplot(1, 2, 1)
    plt.imshow(orig.squeeze() + 0.5)
    plt.title("Real Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rec.squeeze() + 0.5)
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.show()

    
    
class VQ_VAE_Trainer_2(keras.models.Model):
    def __init__(self, train_variance, latent_dim=16, number_of_embeddings=128, **kwargs):
        super(VQ_VAE_Trainer_2, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.number_of_embeddings = number_of_embeddings

        self.vqvae = get_vqvae_layer_hierarchical(self.latent_dim, self.number_of_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            commitment_loss = self.vqvae.losses[0:2]
            codebook_loss =  self.vqvae.losses[2:4]
        
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(commitment_loss+codebook_loss)
            
        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    
    

# PixelCNN layer..
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# residual block layer is based upon PixelConvLayer.
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


