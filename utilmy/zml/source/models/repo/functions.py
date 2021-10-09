from keras.layers import Lambda, Input, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch, 1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_dataset(state_num=10, time_len=50000, signal_dimension=15, CNR=1, window_len=11, half_window_len=5):
    a = np.ones(shape=(state_num, state_num))
    alpha = np.ones(10)*10
    alpha[5:] = 1
    base_prob = np.random.dirichlet(alpha) * 0.1
    for t in range(state_num):
        a[t, :] = base_prob
        a[t, t] += 0.9

    # simulate states
    state = np.zeros(time_len, dtype=np.uint8)
    p = np.random.uniform()
    state[0] = np.floor(p*state_num)
    for t in range(0, time_len-1):
        p = np.random.uniform()
        for s in range(state_num):
            if (p <= np.sum(a[state[t], :s+1])):
                state[t+1] = s
                break

    freq = np.zeros(state_num)
    for t in range(state_num):
        freq[t] = np.sum(state == t)
    loading = np.random.randint(-1, 2, size=(state_num, signal_dimension))

    cov = np.zeros((state_num, signal_dimension, signal_dimension))
    for t in range(state_num):
        cov[t, :, :] = np.matmul(np.transpose(
            [loading[t, :]]), [loading[t, :]])

    # generate BOLD signal
    signal = np.zeros((time_len, signal_dimension))
    for t in range(0, time_len):
        signal[t, :] = np.random.multivariate_normal(
            np.zeros((signal_dimension)), cov[state[t], :, :])
    signal += np.random.normal(size=signal.shape)/CNR
    original_dim = np.uint32(signal_dimension*(signal_dimension-1)/2)

    x_train = np.zeros(
        shape=(time_len-window_len*2, np.uint32(original_dim)))
    sum_corr = np.zeros(shape=(state_num, original_dim))
    occupancy = np.zeros(state_num)

    for t in range(window_len, time_len-window_len):
        corr_matrix = np.corrcoef(np.transpose(
            signal[t-half_window_len:t+half_window_len+1, :]))
        upper = corr_matrix[np.triu_indices(signal_dimension, k=1)]
        x_train[t-window_len, :] = np.squeeze(upper)
        if (np.sum(state[t-half_window_len:t+half_window_len+1] == state[t]) == window_len):
            sum_corr[state[t], :] += x_train[t-window_len, :]
            occupancy[state[t]] += 1

    return original_dim, x_train


def get_model(original_dim, class_num=5, intermediate_dim=64, intermediate_dim_2=16, latent_dim=3,
                                                    batch_size=256, Lambda1=1, Lambda2=200, Alpha=0.075):
    input_shape = (original_dim, )
    inputs = Input(shape=input_shape, name='encoder_input')
    inter_x1 = Dense(intermediate_dim, activation='tanh',
                        name='encoder_intermediate')(inputs)
    inter_x2 = Dense(intermediate_dim_2, activation='tanh',
                        name='encoder_intermediate_2')(inter_x1)
    inter_x3 = Dense(intermediate_dim_2, activation='tanh',
                        name='encoder_intermediate_3')(inter_x1)
    # add 3 means as additional parameters
    dummy = Input(shape=(1,), name='dummy')
    mu_vector = Dense(class_num*latent_dim, name='mu_vector',
                        use_bias=False)(dummy)
    mu = Reshape((class_num, latent_dim), name='mu')(mu_vector)

    # prior categorical distribution
    pi = Dense(class_num, activation='softmax', name='pi')(dummy)

    # posterior categorical distribution
    c = Dense(class_num, activation='softmax', name='c')(inter_x2)

    # outlier/non-outlier classification (Posterior Beta)
    # inter_outlier = Dense(128, activation='relu', name='inter_outlier')(x)
    c_outlier = Dense(2, activation='softmax', name='c_outlier')(inter_x3)

    # q(z|x)
    z_mean = Dense(latent_dim, name='z_mean')(inter_x2)
    z_log_var = Dense(latent_dim, name='z_log_var')(inter_x2)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,),
                name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model([inputs, dummy], [z_mean, z_log_var, z,
                                            mu, c, c_outlier, pi], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    inter_y1 = Dense(intermediate_dim_2, activation='tanh')(latent_inputs)
    inter_y2 = Dense(intermediate_dim, activation='tanh')(inter_y1)
    outputs = Dense(original_dim, activation='tanh')(inter_y2)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder([inputs, dummy])[2])
    vae = Model([inputs, dummy], outputs, name='vae_mlp')
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss = K.tf.multiply(
        reconstruction_loss, c_outlier[:, 0])
    reconstruction_loss *= original_dim
    kl_loss_all = K.tf.get_variable("kl_loss_all", [batch_size, 1],
                                    dtype=K.tf.float32, initializer=K.tf.zeros_initializer)
    kl_cat_all = K.tf.get_variable("kl_cat_all", [batch_size, 1],
                                   dtype=K.tf.float32, initializer=K.tf.zeros_initializer)
    dir_prior_all = K.tf.get_variable("dir_prior_all", [batch_size, 1],
                                      dtype=K.tf.float32, initializer=K.tf.zeros_initializer)
    for i in range(0, class_num):
        c_inlier = K.tf.multiply(c[:, i], c_outlier[:, 0])

        # kl-divergence between q(z|x) and p(z|c)
        kl_loss = 1 + z_log_var - \
            K.square(z_mean-mu[:, i, :]) - K.exp(z_log_var)
        kl_loss = K.tf.multiply(K.sum(kl_loss, axis=-1), c_inlier)
        kl_loss *= -0.5
        kl_loss_all += kl_loss

        # kl-divergence between q(c|x) and p(c) (not including outlier class)
        mc = K.mean(c[:, i])
        mpi = K.mean(pi[:, i])
        kl_cat = mc * K.log(mc) - mc * K.log(mpi)
        kl_cat_all += kl_cat

        # Dir prior: Dir(3, 3, ..., 3)
        dir_prior = -0.1*K.log(pi[:, i])
        dir_prior_all += dir_prior
    mco1 = K.mean(c_outlier[:, 0])
    mco2 = K.mean(c_outlier[:, 1])
    mpo1 = 1-Alpha
    mpo2 = Alpha
    kl_cat_outlier = (mco1 * K.log(mco1) - mco1 * np.log(mpo1) +
                      mco2 * K.log(mco2) - mco2 * np.log(mpo2))

    # total loss
    vae_loss = K.mean(reconstruction_loss +
                      kl_loss_all +
                      dir_prior_all +
                      Lambda1*kl_cat_all)+Lambda2*kl_cat_outlier

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    return vae


def fit(vae, x_train, epochs=1, batch_size=256):
    dummy_train = np.ones((x_train.shape[0], 1))
    vae.fit([x_train, dummy_train],
                    epochs=epochs,
                    batch_size=batch_size)
    return vae


def save(model):
    model.save_weights('vae_mlp_mnist.h5')

def load(model,path):
    model.load_weights(path)

def test(self,encoder,x_train,dummy_train,class_num=5,batch_size=256):
    [z_mean, z_log_var, z, mu, c, c_outlier, pi] = encoder.predict(
        [x_train, dummy_train], batch_size= batch_size)

    labels = np.zeros(x_train.shape[0])
    for i in range(0,x_train.shape[0]):
        max_prob = np.max(np.multiply(c[i, :], c_outlier[i, 0]))
        idx = np.argmax(np.multiply(c[i, :], c_outlier[i, 0]))
        if (max_prob > c_outlier[i, 1]):
            labels[i] = idx
        else:
            labels[i] = class_num
    return labels
