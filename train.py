# coding=utf-8
from keras.callbacks import Callback, ModelCheckpoint
from tools import ModelController

# get the model controller
model_controller = ModelController()
model_controller.show_sample()

# get the model
model = model_controller.get_model()

# get validation data
valid_X, valid_y = model_controller.get_validation_data()

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('val_acc'))


# log the loss and accuracy
history = LossHistory()
checkpointer = ModelCheckpoint('./model.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                               save_weights_only=False, mode='auto')

# train the model
model.fit_generator(
    model_controller.get_image_generator(),
    samples_per_epoch=model_controller.BATCH_SIZE,
    nb_epoch=2000,
    validation_data=(valid_X, valid_y),
    nb_val_samples=model_controller.VALID_SIZE,
    callbacks=[history, checkpointer]
)

# save the loss and accuracy
with open('./losses.txt', 'a') as f:
    f.write(str(history.losses))
    f.write(str(history.accuracy))
