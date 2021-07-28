

import numpy as np
from matplotlib import pyplot as plt
import cv2
from numba import njit
from PIL import Image
import PIL


def sigmoid( x):
    # return 1/(1+np.exp(-x))
    a = np.array([])
    for i in x[0]:
        if i < 0:
            a = np.append(a, 0)
        if i >= 0:
            a = np.append(a, i)
    return np.array([a])

def softmax(x):
    a = np.array([])
    for i in x[0]:
        a = np.append(a,np.exp(i)/np.sum(np.exp(x)))
    return np.array([a])

train_input = []

um = ["0.png","1.png","2.png","3.jpg","4.png","5.jpg","6.jpg","7.jpg","8.png","9.png"]
for i in um:
    im = Image.open(i).convert('L')
    im.thumbnail((25,25))
    print(np.array(im.getdata()).shape)
#     train_input.append([x/255 for x in im.getdata()])
# train_input = np.array(train_input)

#

#
# train_output = np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])
#
#
#
#
#
# weights1 = (2 * np.random.random((200,625)) - 1)      *0.01
# weights2 = (2 * np.random.random((50,200)) -1)        *0.01
# weights3 = (2* np.random.random((10,50)) -1)           *0.01
#
#
#
# biass1 = np.array([np.ones(200)])*0.01
# biass2 = np.array([np.ones(50)])*0.01
# biass3 = np.array([np.ones(10)])*0.01
#
#
#
# for i in range(1):
#     for stp in range(len(train_input)):
#         train_i = np.array([train_input[stp]])
#         train_o = np.array([train_output[stp]])
#
#         break
#         h1 = sigmoid(np.dot(train_i, weights1.T) + biass1)
#         h2 = sigmoid(np.dot(h1, weights2.T) + biass2)
#         o1 = softmax(np.dot(h2, weights3.T) + biass3)
#
#         error_o1 = train_o - o1
#         error_h2 = np.dot(error_o1, weights3)
#         error_h1 = np.dot(error_h2, weights2)
#
#         weights1 += np.dot(np.array(0.01*error_h1 * h1 * (np.array(1) - h1)).T, np.array(train_i))
#         weights2 += np.dot(np.array(0.01*error_h2 * h2 * (np.array(1) - h2)).T, np.array(h1))
#         weights3 += np.dot(np.array(0.01*error_o1 * o1 * (np.array(1) - o1)).T, np.array(h2))
#
#         biass1 += 0.01*error_h1 * h1 * (np.array(1) - h1)
#         biass2 += 0.01*error_h2 * h2 * (np.array(1) - h2)
#         biass3 += 0.01*error_o1 * o1 * (np.array(1) - o1)



# print(o1)
# h1 = sigmoid(np.dot(train_input[0], weights1.T) + biass1)
# h2 = sigmoid(np.dot(h1, weights2.T) + biass2)
# o1 = softmax(np.dot(h2, weights3.T) + biass3)
# print(o1)
# h1 = sigmoid(np.dot(train_input[1], weights1.T) + biass1)
# h2 = sigmoid(np.dot(h1, weights2.T) + biass2)
# o1 = softmax(np.dot(h2, weights3.T) + biass3)
# print(o1)
# h1 = sigmoid(np.dot(train_input[2], weights1.T) + biass1)
# h2 = sigmoid(np.dot(h1, weights2.T) + biass2)
# o1 = softmax(np.dot(h2, weights3.T) + biass3)
# print(o1)