# coding=utf-8
from tools import TEST_DIR, ModelController
import numpy as np

model_controller = ModelController()
model = model_controller.get_model()

while True:
    try:
        test = model_controller.get_test_data()
        break
    except StopIteration:
        break

predictions = model.predict(test)
print(predictions)
