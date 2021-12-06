import cv2
import numpy as np


img = cv2.imread('kit.jpg')
filter_values = [5, 3, 3, 3]


width, height, channels = img.shape


filters = np.random.uniform(size=(filter_values[0], filter_values[1], filter_values[2], filter_values[3]))
after_conv = np.zeros(shape=(filter_values[0], height, width))

for m in range(filter_values[0]):
    for x in range(width):
        for y in range(height):
            for i in range(filter_values[2]):
                for j in range(filter_values[3]):
                    for k in range(filter_values[1]):
                        xx = min(x + i, height - 1)
                        yy = min(y + j, channels - 1)
                        after_conv[m][x][y] += img[k, xx, yy] * filters[m, k, i, j]

epsilon = 1e-9
gamma = 1
beta = 0
for i in range(filter_values[0]):
    mean = after_conv[i].mean(axis=0)
    mean2 = ((after_conv[i] - mean) ** 2).mean(axis=0)
    after_conv[i] = ((after_conv[i] - mean) / np.sqrt(mean2 + epsilon)) * gamma + beta


for i in range(filter_values[0]):
    for x in range(width):
        for y in range(height):
            after_conv[i, x, y] = np.maximum(0, after_conv[i, x, y])


pool_size = 2
channels, width, height  = after_conv.shape
width = width // pool_size
height = height // pool_size

maxpool = np.empty(shape=(channels, width, height))

for m in range(channels):
    for x in range(width):
        for y in range(height):
            default = -1
            for xx in range(x * pool_size, (x + 1) * pool_size):
                for yy in range(y * pool_size, (y + 1) * pool_size):
                    default = np.maximum(default, after_conv[m, xx, yy])
            maxpool[m, x, y] = default

softmax = maxpool
for i in range(filter_values[0]):
    softmax = np.exp(maxpool[i]) / sum(np.exp(maxpool[i]))

print(softmax)
