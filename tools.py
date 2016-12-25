import os
import re

import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


DATA_PATH = '/home/kamil/Documents/kaggle/kagglecatsdogs/data/'
TEST_DIR = DATA_PATH + 'test/'
TRAIN_DIR = DATA_PATH + 'train/'
VALID_DIR = DATA_PATH + 'validation/'


def my_list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, dirs, files in os.walk(directory) for f in files
            if re.match('^.*\.(' + ext + ')', f)]


class ModelController:
    def __init__(self):
        self.BATCH_SIZE = 300
        self.VALID_SIZE = 250
        self.IMAGE_SIZE = (150, 150)
        self.IMAGE_SIZE_CHANNELS = (150, 150, 3)
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
        img_generator = self.__imgdata_generator_distorted.flow_from_directory(
            TRAIN_DIR,
            target_size=self.IMAGE_SIZE,
            batch_size=30,
            class_mode='binary'
        )

        return img_generator

    def get_validation_data(self):
        images = []
        labels = []

        imgs_path = VALID_DIR

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

    def get_sample_img(self):
        return next(self.get_image_generator())

    def show_sample(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.get_sample_img()[0][0])
        plt.ion()
        plt.show()

    def get_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu', input_shape=self.IMAGE_SIZE_CHANNELS))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(.25))
        model.add(Dense(64, activation='tanh'))
        model.add(Dropout(.25))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(decay=1e-6)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

        return model
