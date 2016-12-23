# coding=utf-8

# test sample images
from tools import ModelController

# get the model controller
model_controller = ModelController()
# model_controller.show_sample()

# get the model
model = model_controller.get_model()

# train the model
model.fit_generator(
    model_controller.get_image_generator(),
    samples_per_epoch=model_controller.BATCH_SIZE,
    nb_epoch=50,
    validation_data=model_controller.get_image_generator(mode='valid'),
    nb_val_samples=model_controller.VALID_SIZE
)