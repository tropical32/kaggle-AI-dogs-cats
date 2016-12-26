# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

with open("./losses.txt", "r") as myfile:
    all_data = myfile.read()

SMOOTH = True

all_data_list = all_data.split('][')
all_data_list[0] = all_data_list[0][1:]
all_data_list[-1] = all_data_list[-1][:-1]

losses = np.array(all_data_list[::2])
accuracies = np.array(all_data_list[1::2])

# losses = [item for sublist in losses for item in sublist]
losses = np.array([np.array(loss.split(','), dtype=np.float32) for loss in losses]).ravel()
accuracies = np.array([np.array(accuracy.split(','), dtype=np.float32) for accuracy in accuracies]).ravel()

x = np.arange(0, len(accuracies))

if SMOOTH:
    x_smooth = np.linspace(x.min(), x.max(), 100)
    accuracy_smooth = spline(x, accuracies, x_smooth)
    loss_smooth = spline(x, losses, x_smooth)
    plt.plot(x_smooth, loss_smooth, color='r', label='loss')
    plt.plot(x_smooth, accuracy_smooth, color='g', label='accuracy')
else:
    plt.plot(x, losses, color='r', label='loss')
    plt.plot(x, accuracies, color='g', label='accuracy')

plt.legend(loc='upper right')
plt.grid(True)
plt.ylim(0, 2)
plt.savefig('plot.png')