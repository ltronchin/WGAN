from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, AveragePooling2D, Add
from keras.layers.merge import _Merge
from keras.preprocessing.image import array_to_img

from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, RMSprop

from keras.utils import plot_model
from keras.initializers import RandomNormal, he_uniform

from functools import partial

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt



class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):

        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGAN_ResNet():

    def __init__(self,
                 input_dim,
                 critic_conv_filters,
                 critic_conv_kernel_size,
                 critic_conv_strides,
                 critic_batch_norm_momentum,
                 critic_activation,
                 critic_dropout_rate,
                 critic_learning_rate,
                 generator_initial_dense_layer_size,
                 generator_upsample,
                 generator_conv_filters,
                 generator_conv_kernel_size,
                 generator_conv_strides,
                 generator_batch_norm_momentum,
                 generator_activation,
                 generator_dropout_rate,
                 generator_learning_rate,
                 optimiser,
                 grad_weight,
                 z_dim,
                 batch_size):

        self.name = 'gan'

        self.input_dim = input_dim
        self.critic_conv_filters = critic_conv_filters
        self.critic_conv_kernel_size = critic_conv_kernel_size
        self.critic_conv_strides = critic_conv_strides
        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        self.critic_activation = critic_activation
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_learning_rate = critic_learning_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate

        self.optimiser = optimiser

        self.z_dim = z_dim

        self.n_layers_critic = len(critic_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)  # 'he_normal'
        self.weight_init = he_uniform(seed = 42)
        self.grad_weight = grad_weight
        self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self._build_critic()
        self._build_generator()

        self._build_adversarial()

    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        # -------------------------------
        # Gradiente penalty loss -> termine nella loss function che penalizza il modello se la norma del gradiente del
        # critico devia da 1. La penalizzazione del gradiente è calcolata sulla base delle predizioni del critico su campioni
        # creati interpolando immagini reali e false.
        # -------------------------------
        gradients = K.gradients(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    # Definizione della Wasserstein Loss
    def wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    # funzione per definire la funzione di attivazione
    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha=0.2)
        else:
            layer = Activation(activation)
        return layer

    def _residual_block(self, x, number_filter_output, resample, kernel_size=[3, 3]):

        input_shape = x.shape
        print(input_shape)
        number_filter_input = input_shape[-1]
        if resample == 'down':  # Downsample
            shortcut = AveragePooling2D([2,2])(x)
            shortcut = Conv2D(filters = number_filter_output,
                              kernel_size=[1, 1],
                              kernel_initializer=self.weight_init,
                              activation = None)(shortcut)

            net = Conv2D(number_filter_input,
                         padding = 'same',
                         kernel_initializer=self.weight_init,
                         kernel_size=kernel_size)(x)
            net = self.get_activation(self.critic_activation)(net)
            net = Conv2D(number_filter_output,
                         padding='same',
                         kernel_initializer=self.weight_init,
                         kernel_size=kernel_size)(net)
            net = AveragePooling2D([2, 2])(net)

            return Add()([net, shortcut])

        elif resample == 'up':  # Upsample
            shortcut = UpSampling2D()(x)
            shortcut = Conv2D(number_filter_output,
                              kernel_size=[1, 1],
                              kernel_initializer=self.weight_init,
                              activation=None)(shortcut)

            net = BatchNormalization()(x)
            net = self.get_activation(self.generator_activation)(net)
            net = UpSampling2D()(net)
            net = Conv2D(number_filter_output,
                         padding = 'same',
                         kernel_initializer=self.weight_init,
                         kernel_size=kernel_size)(net)
            net = BatchNormalization()(net)
            net = self.get_activation(self.generator_activation)(net)
            net = Conv2D(number_filter_output,
                         padding = 'same',
                         kernel_initializer=self.weight_init,
                         kernel_size=kernel_size)(net)

            return Add()([net, shortcut])
        else:
            raise Exception('invalid resample value')

    def _build_generator(self):

        generator_input = Input(shape=(self.z_dim,))

        net = generator_input

        number_of_filter = 64
        net = Dense(5 * 5 * 8 * number_of_filter, kernel_initializer=self.weight_init, activation=None)(net)  # 5x5x512, lineare layer
        net = Reshape([5, 5, 8 * number_of_filter])(net)
        net = self._residual_block(net, 8 * number_of_filter, resample='up')  # 10x10x512
        net = self._residual_block(net, 4 * number_of_filter, resample='up')  # 20x20x256
        net = self._residual_block(net, 2 * number_of_filter, resample='up')  # 40x40x128
        net = self._residual_block(net, 1 * number_of_filter, resample='up')  # 80x80x64

        net = BatchNormalization()(net)
        net = self.get_activation(self.generator_activation)(net)
        net = Conv2D(1,kernel_size=[3, 3], kernel_initializer=self.weight_init, padding = 'same', activation='tanh')(net)

        generator_output = net
        self.generator = Model(generator_input, generator_output)


    def _build_critic(self):

        critic_input = Input(shape = self.input_dim)

        filter_size = 64
        net = Conv2D(filter_size,
                     kernel_size = [3, 3],
                     padding = 'same',
                     kernel_initializer=self.weight_init,
                     activation=None)(critic_input)  # 80x80x64
        net = self._residual_block(net, 2 * filter_size, resample='down')  # 40x40x128
        net = self._residual_block(net, 4 * filter_size, resample='down')  # 20x20x256
        net = self._residual_block(net, 8 * filter_size, resample='down')  # 10x10x512
        net = self._residual_block(net, 8 * filter_size, resample='down')  # 5x5x512
        net = Flatten()(net)
        critic_output = Dense(1,kernel_initializer=self.weight_init, activation =None)(net)

        self.critic = Model(critic_input, critic_output)
        self.critic.summary()

    def get_opti(self, lr):
        if self.optimiser == 'adam':
            opti = Adam(lr=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)
        return opti

    def set_trainable(self, model, val):
        model.trainable = val
        for layer in model.layers:
            layer.trainable = val

    def _build_adversarial(self):

        # -------------------------------
        # Costruzione della rete per il training del critico
        # -------------------------------

        # Congelamento dei layer del generatore: il generatore fa parte del modello utilizzato per il training del critico
        # (le immagini interpolate sono coinvolte nella loss function), quindi è necessario congelare i pesi del generatore
        self.set_trainable(self.generator, False)

        # Il modello per il training del critico ha due input:
        # -- Batch di immagini reali
        real_img = Input(shape=self.input_dim)
        # -- Batch di immagini fake -> diamo in input al generatore rumore campionato da una distribuzione gaussiana e
        # generiamo immagini false
        z_disc = Input(shape=(self.z_dim,))
        fake_img = self.generator(z_disc)

        # Le immagini reali e fake passano attaverso il critico, il quale le classifica come false o reali
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'interpolated_samples' argument
        # Keras si aspetta una loss function con due input (predizioni e verità), per questo è necessario definire
        # una loss function personalizzata che permette
        partial_gp_loss = partial(self.gradient_penalty_loss, interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        # Definizione del modello per il training del critico: due input, batch di immagini reali e l'input random che permette
        # di generare immagini false. Tre output, 1 per le immagini reali, -1 per le immagini false, e un tensore "fantoccio"
        # di zeri necessaria per la loss function per il gradient penalty
        self.critic_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])

        # Compilazione del modello del critico cone le tre funzioni di perdiata: two Wasserstein Loss per le immagini reali
        # e false e la gradient Penalty Loss. La loss complessiva del critico è la somma delle singole loss con la gradiente
        # penalty loss pesata di un fattore 10 rispetto alle altre due (vedi paper originale)
        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein, partial_gp_loss],
            optimizer=self.get_opti(self.critic_learning_rate),
            loss_weights=[1, 1, self.grad_weight])

        # -------------------------------
        # Costruzione della rete per il training del generatore
        # -------------------------------

        # Si  congelano i pesi del critico e si scongelano quelli del generatore
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Input del modello: rumore
        model_input = Input(shape=(self.z_dim,))
        # Output del modello: immagine fake
        img = self.generator(model_input)
        model_output = self.critic(img)

        self.model = Model(model_input, model_output)

        # Compilazione del modello
        self.model.compile(optimizer=self.get_opti(self.generator_learning_rate),
                           loss=self.wasserstein)

        # Si scongelano i pesi del critico
        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size, using_generator):

        # -------------------------------
        # Per il training del critico D calcolo la perdita comparando la predizione per l'immagine reale p_i = D(x_i)
        # rispetto y_i = 1 e la predizione per l'immagine generata p_i = D(G(z_i)) rispetto y_i = -1
        # -------------------------------

        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32)  # etichette "fantoccio" per il gradient penalty

        # se using_generator è True significa che si sta utilizzando un generatore per selezionare batch di immagini reali
        if using_generator:
            true_imgs = next(x_train)
            # E' necessario che al discriminatore arrivino sempre tensori di dimensione [batch_size, rows, cols, channels],
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        a = [true_imgs, noise]
        d_loss = self.critic_model.train_on_batch([true_imgs, noise], [valid, fake, dummy])
        return d_loss

    def train_generator(self, batch_size):

        # -------------------------------
        # Per il training del generatore G calcolo la perdita comparando la predizione per l'immagine
        # generata p_i = D(G(z_i)) rispetto y_i = 1
        # -------------------------------

        # valid -> etichette per il generatore, è un tensore di dimensione [batch_size, 1] di 1
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches, n_critic, using_generator):

        # Per ogni epoca passano attraverso la rete del critico 5 batch di immagini
        for epoch in range(self.epoch, self.epoch + epochs):

            # E' necessario portare il critico ad essere confidente sulle predizioni che dà sulle immagini, questo
            # per garantire che il gradiente per l'update del generatore sia accurato. Ciò è in contrasto con una GAN standard
            # per la quale era importante non rendere il discriminatore troppo forte in modo da evitare la scomparsa
            # del gradiente. Per questo motivo si aggiornano i pesi del critico più volte prima di aggiornare nuovamente
            # il generatore (rapporto di 5 a 1)
            for _ in range(n_critic):
                d_loss = self.train_critic(x_train, batch_size, using_generator)

            # Training del generatore
            g_loss = self.train_generator(batch_size)

            print("%d (%d, %d) [D loss: (%.1f)(R %.1f, F %.1f, G %.1f)] [G loss: %.1f]" % (
                epoch, n_critic, 1, d_loss[0], d_loss[1], d_loss[2], d_loss[3], g_loss))

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # Se la divisione tra epoch e print_every_n_batches è pari a 1 si salvano le immagini generate in modo da
            # monitorare le performance del generatore
            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

            if epoch % 100 == 0:
                self.turing_test(x_train, run_folder, using_generator)

            if epoch % 1000 == 0:
                self.plot(run_folder)

            self.epoch += 1

        self.plot(run_folder)

    def turing_test(self, x_train, run_folder, using_generator):

        rows, cols = 1, 10
        noise = np.random.normal(0, 1, (rows * cols, self.z_dim))
        imgs_fake = self.generator.predict(noise)
        # scale from [-1 1] to [0 1]
        imgs_fake = 0.5 * (imgs_fake + 1)
        imgs_fake = np.clip(imgs_fake, 0, 1)

        if using_generator:
            imgs_real = next(x_train)
            if imgs_real.shape[0] != self.batch_size:
                imgs_real = next(x_train)
        else:
            idx = np.random.randint(0, x_train.shape[0], 10)
            imgs_real = x_train[idx]

        # scale from [-1 1] to [0 1]
        imgs_real = 0.5 * (imgs_real + 1)
        imgs_real = np.clip(imgs_real, 0, 1)

        fig = plt.figure()
        for i in range(1, (rows * cols)):
            img_fake = imgs_fake[i]
            ax = fig.add_subplot(rows, cols, i)
            ax.imshow(np.squeeze(img_fake), cmap='gray')
            ax.axis('off')

        for i in range(1, rows * cols):
            img_real = imgs_real[i]
            ax = fig.add_subplot(2, 10, i)
            ax.imshow(array_to_img(img_real), cmap='gray')
            ax.axis('off')
        plt.savefig(os.path.join(run_folder, "plot/Visual_Turing_test_epoch%d.png"%(self.epoch)), dpi=1200, format='png')
        plt.close()

    def sample_images(self, run_folder):
        rows, cols = 10, 10

        # Restituisce rows*cols tensori di dimensione 100x1, in cui ciascun elemento del tensore è un
        # random campionato da una distribuzione Gaussiana di media 0 e deviazione standard 1: restituisce un tensore di
        # dimensione [rows*cols, 100]
        noise = np.random.normal(0, 1, (rows * cols, self.z_dim))
        # Generazione di rows*cols immagini: gen_imgs ha dimensione [rows*cols, 64, 64, 3]
        gen_imgs = self.generator.predict(noise)

        # Rescale dell'immagine dall'intervallo [-1 1] all'intervallo [0 1]
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        idx = 0

        for i in range(rows):
            for j in range(cols):
                # squeeze rimuove una dimensione al tensore in input, es:
                # gen_imgs ->  [25, 64, 64, 3], np.squeeze(gen_imgs) -> [64, 64, 3]
                axs[i, j].imshow(np.squeeze(gen_imgs[idx, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                idx += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'plot/model.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.critic, to_file=os.path.join(run_folder, 'plot/critic.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.generator, to_file=os.path.join(run_folder, 'plot/generator.png'), show_shapes=True,
                   show_layer_names=True)

    def save(self, folder):

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([self.input_dim,
                         self.critic_conv_filters,
                         self.critic_conv_kernel_size,
                         self.critic_conv_strides,
                         self.critic_batch_norm_momentum,
                         self.critic_activation,
                         self.critic_dropout_rate,
                         self.critic_learning_rate,
                         self.generator_initial_dense_layer_size,
                         self.generator_upsample,
                         self.generator_conv_filters,
                         self.generator_conv_kernel_size,
                         self.generator_conv_strides,
                         self.generator_batch_norm_momentum,
                         self.generator_activation,
                         self.generator_dropout_rate,
                         self.generator_learning_rate,
                         self.optimiser,
                         self.grad_weight,
                         self.z_dim,
                         self.batch_size], f)

        self.plot_model(folder)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'models/model.h5'))
        self.critic.save(os.path.join(run_folder, 'models/critic.h5'))
        self.generator.save(os.path.join(run_folder, 'models/generator.h5'))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def plot(self, run_folder):
        # -- PLOT risultati finali --
        fig = plt.figure()
        plt.plot([x[0] for x in self.d_losses], label='critic average of real and fake', color='black', linewidth=0.35)
        plt.plot([x[1] for x in self.d_losses], label='critic real', color='green', linewidth=0.35)
        plt.plot([x[2] for x in self.d_losses], label='critic fake', color='red', linewidth=0.35)
        plt.plot(self.g_losses, label='generator', color='orange', linewidth=0.35)

        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Wasserstein loss', fontsize=18)
        plt.legend()
        plt.savefig(os.path.join(run_folder, "plot/loss.png"), dpi=1200, format='png')
        plt.show()