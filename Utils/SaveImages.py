# Tramite questo script si vogliono creare due cartelle:
# -- cartella "real" composta da 100 slice reali
# -- cartella "fake" composta da 100 slice generate dalla GAN

from tensorflow.keras import models
import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Load import Load
from PIL import Image
from keras.preprocessing.image import array_to_img

def gen_img(batch, generator):
    # funzione per la generazione di una batch di immagini
    noise = np.random.normal(0, 1, (batch, 128))
    return generator.predict(noise)

def plot_single_images(imgs, label, path):

    imgs = 0.5 * (imgs + 1)
    imgs = np.clip(imgs, 0, 1)
    imgs = (imgs * 255).astype(np.uint8)
    for idx in range(100):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1,bottom=0,right=1, left=0,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(np.reshape(imgs[idx], (80,80)), Interpolation = None,  cmap='gray', vmin=0, vmax=255)
        plt.savefig(os.path.join(path, "img_{}_{}.png".format(idx, label)), format='png', bbox_inches = 'tight',
                    pad_inches = 0, dpi='figure')
        plt.show()

        #im = Image.fromarray(np.reshape(imgs[idx], (80,80)))
        #im.save(os.path.join(path, "imgtiff_{}_{}.tiff".format(idx, label)), vmin=0, vmax = 255)


run_folder = 'GUI_images/003'
print(run_folder)
if not os.path.exists(run_folder):
    os.makedirs(run_folder)
    os.makedirs(os.path.join(run_folder, 'real'))
    os.makedirs(os.path.join(run_folder, 'fake'))

batch = 100

# Caricamento del modello trainato
path_model= 'D:/Documenti/Tesi/Run/run/gan/WGAN-resnet/Data augmentation/003/all/survey/'
generator = models.load_model(os.path.join(path_model, 'models/generator.h5'))

# Generazione immagini
gen_imgs = gen_img(batch, generator)

# Caricamento immagini reali
load= Load()
path_slice = 'D:/Download/data/ID_RUN/ID8/Slices_data/layer/slices_padding_layer.mat'
true_data_generator = load.load_ctslice(path_slice , batch, augmentation=True, acgan= False,
                                                 file_mat='slices_padding_layer')
true_imgs = next(true_data_generator)

# Salvataggio immagini
plot_single_images(imgs = gen_imgs, label = 'fake', path=os.path.join(run_folder, 'fake/'))

plot_single_images(imgs = true_imgs, label = 'real', path=os.path.join(run_folder, 'real/'))
