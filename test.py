# coding=utf-8
from tools import ModelController
import pandas as pd

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

