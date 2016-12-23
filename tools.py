from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

DATA_PATH = '/home/kamil/Documents/kaggle/kagglecatsdogs/data/'
TEST_DIR = DATA_PATH + 'test/'
TRAIN_DIR = DATA_PATH + 'train/'
VALID_DIR = DATA_PATH + 'validation/'


class ModelController:
    def __init__(self):
        self.BATCH_SIZE = 200
        self.VALID_SIZE = 80
        self.IMAGE_SIZE = (150, 150)
        self.IMAGE_SIZE_CHANNELS = (150, 150, 3)
        self.__imgdata_generator = ImageDataGenerator(
            # samplewise_std_normalization=True,
            rescale=1. / 255.,
            rotation_range=20,
            shear_range=.05,
            zoom_range=.05,
            fill_mode='constant',
            horizontal_flip=True,
            vertical_flip=True,
        )

    def get_image_generator(self, mode='train'):
        if mode == 'train':
            path = TRAIN_DIR
        elif mode == 'test':
            path = TEST_DIR
        elif mode == 'valid':
            path = VALID_DIR
        else:
            raise Exception('Invalid mode; available modes: "train", "test", "valid".')

        img_generator = self.__imgdata_generator.flow_from_directory(
            path,
            target_size=self.IMAGE_SIZE,
            batch_size=32,
            class_mode='binary'
        )

        return img_generator

    def get_sample_img(self):
        return next(self.get_image_generator())

    def show_sample(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.get_sample_img()[0][0])
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
