# Importo le librerie necessarie
import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Load import Load
from Models.WGANGP import WGANGP

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Parametri della RUN
section = 'gan'
run_id = '004_resnet_adaptive_ct_images_continue'
run_folder = 'D:/Documenti/Tesi/Run/run/{}/'.format(section)
model = 'WGAN-resnet'
data = 'Data augmentation'
idd = '004'
run_folder += '/'.join([model, data, idd, run_id])
print(run_folder)

if not os.path.exists(run_folder):
    os.makedirs(run_folder)
    os.makedirs(os.path.join(run_folder, 'plot'))
    os.makedirs(os.path.join(run_folder, 'images'))
    os.makedirs(os.path.join(run_folder, 'weights'))
    os.makedirs(os.path.join(run_folder, 'models'))

mode = 'load' # 'build'

path_slice = 'D:/Download/data/ID_RUN/ID8/Slices_data/layer/slices_padding_layer_adaptive.mat'
# -- DATA --

load= Load()

batch_size = 64
image_size = 80
input_dim = (image_size, image_size, 1)
data_flow = load.load_ctslice(path_slice,
                              batch_size,
                              augmentation = True,
                              acgan = False)

# tupla che contiene all'indice 0 una batch di immagini (imgs_real[0]) e all'
# indice 1 le label associate
real_data = next(data_flow)
print(real_data.shape)

# -- ARCHITETTURA --
gan = WGANGP(input_dim = input_dim,
             critic_conv_filters = [128, 256, 512, 1024],
             critic_conv_kernel_size = [5, 5, 5, 5],
             critic_conv_strides = [2, 2, 2, 2],
             critic_batch_norm_momentum = None,
             critic_activation = 'leaky_relu',
             critic_dropout_rate = None,
             critic_learning_rate = 0.0002, #0.0001
             generator_initial_dense_layer_size = (5, 5, 1024),
             generator_upsample = [2, 2, 2, 2],
             generator_conv_filters = [512, 256, 128, 1],
             generator_conv_kernel_size = [5, 5, 5, 5],
             generator_conv_strides = [2, 2, 2, 2],
             generator_batch_norm_momentum = 0.9, #0.99
             generator_activation = 'leaky_relu',
             generator_dropout_rate = None,
             generator_learning_rate = 0.0002,
             optimiser = 'adam',
             grad_weight = 10,
             z_dim = 128,
             batch_size = batch_size,
             use_resnet = True,
             number_of_filters_generator  = 64,
             number_of_filters_critic = 64)

if mode == 'build':
    #gan.save(run_folder)
    print('RUN')
else:
    gan.load_weights(os.path.join(run_folder, 'weights/weights.h5'))

gan.critic.summary()
gan.generator.summary()

# -- TRAINING --
epochs = 10000
print_every_n_batch = 5
n_critic = 5

gan.train(data_flow,
          batch_size = batch_size,
          epochs = epochs,
          run_folder = run_folder,
          print_every_n_batches = print_every_n_batch,
          n_critic = n_critic,
          using_generator = True)

# -- PLOT risultati finali --
fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], label='critic average of real and fake', color='black', linewidth=0.35)
plt.plot([x[1] for x in gan.d_losses], label='critic real', color='green', linewidth=0.35)
plt.plot([x[2] for x in gan.d_losses], label='critic fake', color='red', linewidth=0.35)
plt.plot(gan.g_losses, label='generator', color='orange', linewidth=0.35)

plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Wasserstein loss', fontsize=18)
plt.legend()
plt.savefig(os.path.join(run_folder, "plot/loss.png"), dpi=1200, format='png')
plt.show()

gan.turing_test(data_flow, run_folder)