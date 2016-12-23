from keras.preprocessing.image import ImageDataGenerator

DATA_PATH = '/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/training_data/kagglecatsdogs/'
TEST_DIR = DATA_PATH + 'test/'
TRAIN_DIR = DATA_PATH + 'train/'
VALID_DIR = DATA_PATH + 'valid/'


class ImageGeneration:
    def __init__(self):
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
            target_size=(150, 150),
            batch_size=32,
            class_mode='sparse'  # binary?
        )

        return img_generator
