from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import scipy.io as sio

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

    def load_ctslice(self, path_slices, batch_size):

        load = sio.loadmat(path_slices)
        print(load.keys())
        data = load['slices_padding_layer'][0]

        slices = []
        for idx in range(data.shape[0]):
            slices.append(img_to_array(data[idx][0]))

        slices = np.array(slices)

        print("[INFO]-- Numero e dimensione slice {}".format(slices.shape))

        # in python double = float64
        data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('double') - 0.5) / 0.5)
        data_flow = data_gen.flow(slices,
                                  batch_size=batch_size,
                                  shuffle=True)

        return data_flow