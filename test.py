# coding=utf-8
import os

import numpy as np
import pandas as pd

from tools import ModelController, DATA_PATH

model_controller = ModelController(
    os.path.join('/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/kagglecatsdogs/saves',
                 'model.32238-0.19.hdf5'))
model = model_controller.get_model()
predictions = []

while True:
    try:
        test = model_controller.get_test_data()
        prediction = model.predict(test, verbose=1)
        predictions.append(prediction)
    except StopIteration:
        break

predictions = np.array(predictions).ravel()
df = pd.DataFrame(predictions)
df.columns = ['label']
df.index += 1
df.to_csv(DATA_PATH + 'my_submission.csv', index_label='id')
