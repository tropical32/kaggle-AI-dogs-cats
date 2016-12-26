import os
import re

import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

DATA_PATH = '/home/kamil/Documents/kaggle/kagglecatsdogs/data/'
TEST_DIR = DATA_PATH + 'test/'
TRAIN_DIR = DATA_PATH + 'train/'
VALID_DIR = DATA_PATH + 'validation/'


def my_list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, dirs, files in os.walk(directory) for f in
            sorted(files, key=lambda name: int(name.split('.')[-2].split('/')[-1]))
            if re.match('^.*\.(' + ext + ')', f)]


class ModelController:
    def __init__(self):
        self.BATCH_SIZE = 300
        self.VALID_SIZE = 250
        self.IMAGE_SIZE = (32, 32)
        self.IMAGE_SIZE_CHANNELS = (32, 32, 3)
        self.__imgdata_generator_distorted = ImageDataGenerator(
            rescale=1. / 255.,
            rotation_range=20,
            shear_range=.05,
            zoom_range=.05,
            fill_mode='constant',
            horizontal_flip=True,
            vertical_flip=True,
        )

    def get_image_generator(self):
        try:
            return self.__img_generator_distorted
        except:
            pass  # not initialized

        self.__img_generator_distorted = self.__imgdata_generator_distorted.flow_from_directory(
            TRAIN_DIR,
            target_size=self.IMAGE_SIZE,
            batch_size=30,
            class_mode='binary'
        )

        return self.__img_generator_distorted

    def get_validation_data(self):
        images = []
        labels = []

        X = []
        for picture in my_list_pictures(VALID_DIR, ext='jpg'):
            img = img_to_array(load_img(picture, target_size=self.IMAGE_SIZE_CHANNELS))
            img = img * (1. / 255.)  # rescale the image
            X.append(img)
        X = np.asarray(X)

        assert len(X) % 2 == 0

        labels_cats = np.zeros(len(X) // 2)
        labels_dogs = np.ones(len(X) // 2)

        return X, np.append(labels_cats, labels_dogs)

    def __test_data_generator(self):
        for picture in my_list_pictures(TEST_DIR, ext='jpg'):
            img = img_to_array(load_img(picture, target_size=self.IMAGE_SIZE_CHANNELS))
            img = img * (1. / 255.)
            yield img

    def get_test_data(self, batch_size=125):
        try:
            self.__test_data_generator_object
        except AttributeError:
            self.__test_data_generator_object = self.__test_data_generator()

        imgs = []
        for i in range(batch_size):
            imgs.append(next(self.__test_data_generator_object))
        return np.asarray(imgs)

    def get_sample_img(self):
        return next(self.get_image_generator())

    def show_sample(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.get_sample_img()[0][0])
        plt.ion()
        plt.show()

    def get_model(self):
        try:
            return self.__model
        except AttributeError:
            pass  # not initialized

        try:
            self.__model = load_model('./model_cifar.hdf5')  # TODO: replace with regex
            print('Found an existing model.')
            return self.__model
        except:
            print('No models saved. Creating a new model...')

        model = Sequential()
        model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=self.IMAGE_SIZE_CHANNELS))
        model.add(MaxPooling2D())
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adadelta()
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        self.__model = model

        return self.__model
