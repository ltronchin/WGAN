# Tramite questo script si vogliono creare due cartelle:
# -- cartella "real" a sua volta composta da
#   -- cartella "adaptive": 25 slice reali adaptive
#   -- cartella "not_adaptive": 25 slice reali non adaptive

# -- cartella "fake" a sua volta composta da
#   -- cartella "adaptive": 25 slice sintetiche adaptive
#   -- cartella "not_adaptive": 25 slice sintetiche non adaptive

# In totale saranno salvate 100 slice, 50 reali e 50 sintetiche.
# adaptive -- label 1
# not adaptive -- label 0

from tensorflow.keras import models
import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Load import Load
import cv2
from PIL import Image
import scipy.misc

from keras.preprocessing.image import array_to_img

def gen_img(batch, generator):
    # funzione per la generazione di una batch di immagini
    noise = np.random.normal(0, 1, (batch, 128))
    return generator.predict(noise)

def plot_single_images(imgs, label, path):

    imgs = 0.5 * (imgs + 1)
    imgs = np.clip(imgs, 0, 1)
    #imgs = (imgs * 255).astype(np.uint8)
    for idx in range(30):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1,bottom=0,right=1, left=0,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(array_to_img(imgs[idx]), cmap='gray')
        plt.savefig(os.path.join(path, "img_{}_{}.png".format(idx, label)), format='png', bbox_inches = 'tight',
                    pad_inches = 0)
        plt.show()

run_folder = 'GUI_images'
print(run_folder)
if not os.path.exists(run_folder):
    os.makedirs(run_folder)
    os.makedirs(os.path.join(run_folder, 'real'))
    os.makedirs(os.path.join(run_folder, 'real/adaptive'))
    os.makedirs(os.path.join(run_folder, 'real/not_adaptive'))
    os.makedirs(os.path.join(run_folder, 'fake'))
    os.makedirs(os.path.join(run_folder, 'fake/adaptive'))
    os.makedirs(os.path.join(run_folder, 'fake/not_adaptive'))

batch = 64

# Caricamento del modello trainato con immagini adaptive
path_model= 'C:/Users/User/Desktop/'
generator_adaptive = models.load_model(os.path.join(path_model, 'generator_003.h5'))
# Caricamento del modello trainato con immagini non adaptive
generator_not_adaptive = models.load_model(os.path.join(path_model, 'generator_not_adaptive.h5'))

# Generazione immagini
gen_imgs_adaptive = gen_img(batch, generator_adaptive)
gen_imgs_not_adaptive = gen_img(batch, generator_not_adaptive)

# Caricamento immagini reali
load= Load()
path_slice_adaptive = 'C:/Users/User/Desktop/Tesi/Matlab/data/ID_RUN/ID8/Slices_data/layer/slices_padding_layer_adaptive.mat'
path_slice_not_adaptive = 'C:/Users/User/Desktop/Tesi/Matlab/data/ID_RUN/ID8/Slices_data/layer/slices_padding_layer_not_adaptive.mat'

true_data_generator_adaptive = load.load_ctslice(path_slice_adaptive , batch, file_mat='slices_padding_layer_adaptive')
true_data_generator_not_adaptive = load.load_ctslice(path_slice_not_adaptive , batch, file_mat='slices_padding_layer_not_adaptive')

true_imgs_adaptive = next(true_data_generator_adaptive)
true_imgs_not_adaptive = next(true_data_generator_not_adaptive)

# Salvataggio immagini
plot_single_images(imgs = gen_imgs_adaptive, label = 'fake_adaptive', path=os.path.join(run_folder, 'fake/adaptive'))
plot_single_images(imgs = gen_imgs_not_adaptive, label = 'fake_not_adaptive', path=os.path.join(run_folder, 'fake/not_adaptive'))

plot_single_images(imgs = true_imgs_adaptive, label = 'real_adaptive', path=os.path.join(run_folder, 'real/adaptive'))
plot_single_images(imgs = true_imgs_not_adaptive, label = 'real_not_adaptive', path=os.path.join(run_folder, 'real/not_adaptive'))
