from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

class Load():
    # Caricamento e preparazione delle immagini dal dataset CIFAR10
    def load_cifar10(self, label):
        # label = 5  # selezione della sola classe con label 5 (cani ad esempio)
        # Download dei dati
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        train_mask = [Y[0] == label for Y in Y_train]
        test_mask = [Y[0] == label for Y in Y_test]

        # Unione del training set e del test set: il focus è generare nuovi campioni, quindi NON
        # c'è la necessità di un set di test
        X = np.concatenate((X_train[train_mask], X_test[test_mask]), axis=0)
        Y = np.concatenate((Y_train[train_mask], Y_test[test_mask]), axis=0)

        X = X.astype('float32')
        # Normalizzazione dati tra -1 e 1
        X = (X - 127.5) / 127.5
        return X

    def load_celeb(self, data_folder, image_size, batch_size):

        data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5)

        data_flow = data_gen.flow_from_directory(data_folder,
                                                 target_size=(image_size, image_size),
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 class_mode=None)

        return data_flow

    def load_ctslice(self, path_slices, batch_size, augmentation):

        load = sio.loadmat(path_slices)
        print(load.keys())
        data = load['slices_padding_layer_adaptive'][0]

        slices = []
        for idx in range(data.shape[0]):
            slices.append(img_to_array(data[idx][0], dtype='double'))

        slices = np.array(slices)

        print("[INFO]-- Numero e dimensione slice {}".format(slices.shape))
        if augmentation:
            data_gen = ImageDataGenerator(rotation_range = 175,
                                          width_shift_range = (-7, +7),
                                          height_shift_range = (-7, +7),
                                          horizontal_flip = 'true',
                                          vertical_flip = 'true',
                                          fill_mode = 'constant',
                                          cval = 0,
                                          preprocessing_function=lambda x: (x.astype('double') - 0.5) / 0.5)
        else:
            # in python double = float64
            data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('double') - 0.5) / 0.5)

        data_flow = data_gen.flow(slices,
                                  batch_size=batch_size,
                                  shuffle=True)

        true_imgs = next(data_flow)
        self.plot_images(imgs=true_imgs)

        return data_flow

    def plot_images(self, imgs):

        imgs = 0.5 * (imgs + 1)
        imgs = np.clip(imgs, 0, 1)

        fig, axs = plt.subplots(8, 8, figsize=(15, 15))
        idx = 0

        for i in range(8):
            for j in range(8):
                axs[i, j].imshow(np.squeeze(imgs[idx, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                idx += 1

        plt.show()