from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

DATA_PATH = '/home/kamil/Documents/kaggle/kagglecatsdogs/'
TEST_DIR = DATA_PATH + 'test/'
TRAIN_DIR = DATA_PATH + 'train/'
VALID_DIR = DATA_PATH + 'valid/'


class ImageGeneration:
    def __init__(self):
        self.IMAGE_SIZE = (150, 150)
        self.IMAGE_SIZE_CHANNELS = (150, 150, 3)
        self.__imgdata_generator = ImageDataGenerator(
            # rescale=1./255.,
            zca_whitening=True,
            rotation_range=40,
            width_shift_range=.1,
            height_shift_range=.1,
            shear_range=.1,
            zoom_range=.05,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            rescale=.05)

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
            class_mode='sparse'  # binary?
        )

        return img_generator

    def get_model(self):
        model = Sequential()
        model.add(Convolution2D(64, 6, 6, subsample=(2, 2), activation='relu', input_shape=self.IMAGE_SIZE_CHANNELS))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        model.add(Convolution2D(256, 2, 2, activation='relu', border_mode='same'))
        model.add(Convolution2D(256, 2, 2, activation='relu', border_mode='same'))
        model.add(Convolution2D(128, 2, 2, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        model.add(Flatten())
        model.add(Dense(1024, activation='tanh'))
        model.add(Dropout(.5))
        model.add(Dense(1024, activation='tanh'))
        model.add(Dropout(.5))
        model.add(Dense(1, activation='softmax'))
        adam_optimizer = Adam(decay=1e-6)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam_optimizer)

        return model
