# coding=utf-8
from tools import ModelController, DATA_PATH
import pandas as pd
import numpy as np

model_controller = ModelController()
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
df.to_csv(DATA_PATH + 'submission.csv', index_label='id')
