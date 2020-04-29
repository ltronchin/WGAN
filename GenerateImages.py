from tensorflow.keras import models
import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Load import Load
from keras.preprocessing.image import array_to_img

def gen_img(batch, generator):
    # funzione per la generazione di una batch di immagini
    noise = np.random.normal(0, 1, (batch, 128))
    return generator.predict(noise)

def plot_images(imgs, true, path_model,  n_image):

    imgs = 0.5 * (imgs + 1)
    imgs = np.clip(imgs, 0, 1)

    fig, axs = plt.subplots(8, 8, figsize=(15, 15))
    idx = 0

    for i in range(8):
        for j in range(8):
            #print(i,j)
            axs[i, j].imshow(np.squeeze(imgs[idx, :, :, :]), cmap='gray')
            axs[i, j].axis('off')
            idx += 1

    plt.savefig(os.path.join(path_model,"{}_{}.png".format(true,n_image)), dpi=1200, format='png')
    plt.show()

# Caricamento del modello trainato
path_model = 'D:/Documenti/Tesi/Run/run/gan/WGAN-resnet/002_resnet_adaptive_continue/'
generator = models.load_model(os.path.join(path_model, 'models/generator.h5'))
load= Load()
batch = 64
images = 5
path_slice = 'D:/Download/data/ID_RUN/ID8/Slices_data/layer/slices_padding_layer_adaptive.mat'
true_data_flow = load.load_ctslice(path_slice , batch, augmentation = True)

for i in range(images):
    # Immagini false
    gen_imgs = gen_img(batch, generator)
    plot_images(imgs = gen_imgs, true = 'fake', path_model = path_model, n_image = i)

    # Immagini reali
    true_imgs = next(true_data_flow)
    plot_images(imgs = true_imgs, true = 'real', path_model = path_model, n_image = i)