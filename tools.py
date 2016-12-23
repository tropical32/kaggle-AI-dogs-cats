from keras.preprocessing.image import ImageDataGenerator

DATA_PATH = '/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/training_data/kagglecatsdogs/'


def next_batch():
    generator = ImageDataGenerator(
        # featurewise_center=True,
        # samplewise_center=True,
        # featurewise_std_normalization=True,
        # samplewise_std_normalization=True,
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
