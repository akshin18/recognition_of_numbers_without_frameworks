import numpy as np
from matplotlib import pyplot as plt
import cv2


# image2 = cv2.imread('1.png')
# image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
# image2 = cv2.resize(image2,(20,20))
# sharp2 = np.array([image2.reshape(400)/255])
#
#
#
#
# image3 = cv2.imread('2.png')
# image3 = cv2.cvtColor(image3,cv2.COLOR_RGB2GRAY)
# image3 = cv2.resize(image3,(20,20))
# sharp3 = np.array([image3.reshape(400)/255])
#
#
# image = cv2.imread('0.png')
# image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
# image = cv2.resize(image,(20,20))
# sharp = np.array([image.reshape(400)/255])
#
#
# def sigmoid(x):
#
#     return (1/(1+np.exp(-x)))
#
#
#
# train_input = np.append(sharp,sharp2,axis=0)
#
# train_input = np.append(train_input,sharp3,axis=0)
#
#
# train_outpu = np.array([[1,0,0],[0,1,0],[0,0,1]])
#
#
# weights1 = 2 * np.random.random((100,400)) - 1
# weights2 = 2 * np.random.random((30,100)) - 1
# weights3 = 2 * np.random.random((3,30)) - 1
#
#
# biass1 = np.array([np.ones(100)])
# biass2 = np.array([np.ones(30)])
# biass3 = np.array([np.ones(3)])
#
# for i in range(10000):
#
#     for tes in range(len(train_input)):
#         # break
#
#         train = np.array([train_input[tes]])
#         traino = np.array(train_outpu[tes])
#
#         h1 = sigmoid(np.dot(train , weights1.T) + biass1)
#
#         h2 = sigmoid(np.dot(h1 , weights2.T) + biass2)
#         o1 = sigmoid(np.dot(h2 , weights3.T) + biass3)
#
#         o1 = np.array(o1)
#         h1 = np.array(h1)
#         h2 = np.array(h2)
#
#
#         error_o = traino - o1
#         error_h2 = np.dot(error_o , weights3)
#
#         error_h1 = np.dot(error_h2 , weights2)
#
#         weights1 += np.dot((np.array(error_h1*h1*(np.array(1)-h1))).T,np.array(train))
#         weights2 += np.dot(np.array(error_h2*h2*(np.array(1)-h2)).T,np.array(h1) )
#         weights3 += np.dot((np.array(error_o*o1*(np.array(1)-o1))).T,np.array(h2))
#
#
#
#         biass1 += error_h1*h1*(np.array(1)-h1)
#         biass2 += error_h2*h2*(np.array(1)-h2)
#         biass3 += error_o * o1 * (np.array(1) - o1)
#
#
# print(o1)


# train_input = sharp
# h1 = sigmoid(np.dot(train_input , weights1) + biass1)
# h2 = sigmoid(np.dot(h1 , weights2) + biass2)
# o1 = np.dot(h2 , weights3) + biass3
# print(o1)

############################################################################################3

####################################################3 НИЖЕ НЕ ЧИТАЙ ЭТО МОЙ ТЕСТ
################################################################################################3
#
# from time import time
# def sigmoid(x):
#     ret = []
#     for w in x:
#
#         ret.append(1/(1+np.exp(-w)))
#     return ret
# bads = []
#
# train_input = np.array([1,0,1])
#
# weights1 = 2*np.random.random((3,2))-1
# weights1 = np.array([[0.1,0.2,0.3],[0.4,0.2,0.3]])
# # weights2 = 2*np.random.random((2,3))-1
# weights2 = np.array([[0.3,0.4],[0.1,0.1],[ 0.5, 0.1]])
# # weights3 = 2*np.random.random((3,1))-1
# weights3 = np.array([[0.1,0.2,0.3]])
# # print(weights1.shape,weights2.shape,weights3.shape)
#
# train_output = np.array([1])
#
# biass = np.array([1.0,1.0])
# biass2 = np.array([1.0,1.0,1.0])
# biass3 = np.array([float(1)])
#
#
#
# t1 =time()
# for i in range(1000):
#
#     h1 = sigmoid(np.dot(train_input,weights1.T)+biass)
#
#     h2 = sigmoid(np.dot(h1,weights2.T)+biass2)
#
#     o = sigmoid(np.dot(h2,weights3.T)+biass3)
#
#     error_o = train_output - o
#
#
#     error_h2 = (np.dot(error_o,weights3))
#
#     error_h1 = np.dot(error_h2,weights2)
#
#
#

#     weights1 += np.dot((np.array([error_h1*(h1*(np.array([1])-h1))])).T,np.array([train_input]))
#
#
#     weights2 += np.dot(np.array([(error_h2*(h2*(np.array(1)-h2)))]).T,np.array([h1]))
#
#     # print([error_h2[0][::-1]*(h2*(np.array([1])-h2))][0][0])
#     # print([error_h2[0][::-1] * (h2 * (np.array([1]) - h2))][0][1])
#     # print([error_h2[0][::-1] * (h2 * (np.array([1]) - h2))][0][2])
#
#
#     weights3 += np.dot((error_o*o*(np.array(1)-o)),np.array([h2]))
#
#
#     biass += error_h1*(h1*(np.array([1])-h1))
#     biass2 +=   (error_h2 * h2 * (np.array(1) - h2))
#     biass3 += (error_o*o*(np.array(1)-o))
#     print(biass3)
#     break
# # print(time()-t1)
# #
# print(o)

####################################################################################################









# def relu(x):
#     a = np.array([])
#
#     for i in x:
#         if i < 0:
#             a = np.append(a,i)
#         if i >= 0:
#             a = np.append(a,i)
#
#     return np.array([a])
#
# ka = np.array([np.random.randint(-3,5,10)])
#
#
#
# print(relu(ka[0]))

print(np.exp(1000))









