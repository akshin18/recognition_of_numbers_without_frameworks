

import numpy as np
from matplotlib import pyplot as plt
import cv2
from numba import njit




image2 = cv2.imread('1.png')
image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
image2 = cv2.resize(image2,(20,20))
sharp2 = np.array([image2.reshape(400)/255])
print(sharp2.shape)




image3 = cv2.imread('2.png')
image3 = cv2.cvtColor(image3,cv2.COLOR_RGB2GRAY)
image3 = cv2.resize(image3,(20,20))
sharp3 = np.array([image3.reshape(400)/255])


image = cv2.imread('0.png')
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
image = cv2.resize(image,(20,20))
sharp = np.array([image.reshape(400)/255])


def sigmoid(x):
    shap = x.shape
    a = np.array([])
    for i in x:

        a = np.append(a,(1/(1+np.exp(-i))))
        # print(a)
    return a.reshape(shap)
njit()
def cart():
    def softmax(x):
        a = np.array([])

        for i in x[0]:


            a = np.append(a,(np.exp(i/100))/(np.sum(np.exp(x/100))))

        return np.array([a])

    def relu(x):
        a = np.array([])

        for i in x[0]:
            if i < 0:
                a = np.append(a,0)
            if i >= 0 and i <300:
                # print(i)
                a = np.append(a,i)
            if i >= 300:
                a = np.append(a,300)

        return np.array([a])

    train_input = np.append(sharp,sharp2,axis=0)

    train_input = np.append(train_input,sharp3,axis=0)


    train_outpu = np.array([[1,0,0],[0,1,0],[0,0,1]])


    weights1 = 2 * np.random.random((200,400)) - 1
    weights2 = 2 * np.random.random((100,200)) - 1
    weights3=  2 * np.random.random((30,100)) - 1
    weights4 = 2 * np.random.random((3,30)) - 10.


    biass1 = np.array([np.ones(200)])
    biass2 = np.array([np.ones(100)])
    biass3 = np.array([np.ones(30)])
    biass4 = np.array([np.ones(3)])




    for i in range(1000):

        for tes in range(len(train_input)):


            train = np.array([train_input[tes]])
            traino = np.array(train_outpu[tes])
            h1 = relu(np.dot(train , weights1.T) + biass1)

            h2 = relu(np.dot(h1 , weights2.T) + biass2)

            h3 = relu(np.dot(h2, weights3.T) + biass3)
            o1 = softmax(np.dot(h3 , weights4.T) + biass4)



            o1 = np.array(o1)
            h1 = np.array(h1)
            h2 = np.array(h2)
            h3 = np.array(h3)

            #
            # print(traino.shape , o1.shape)
            error_o = traino - o1

            error_h3 = np.dot(error_o , weights4)

            error_h2 = np.dot(error_h3, weights3)
            error_h1 = np.dot(error_h2, weights2)

            weights1 += np.dot((0.0001*np.array(error_h1*h1*(np.array(1)-h1))).T,np.array(train))
            weights2 += np.dot( 0.0001*np.array(error_h2*h2*(np.array(1)-h2)).T,np.array(h1) )
            weights3 += np.dot((0.0001*np.array(error_h3*h3*(np.array(1)-h3))).T,np.array(h2))
            weights4 += np.dot((0.0001*np.array(error_o * o1 * (np.array(1) - o1))).T, np.array(h3))


            biass1 += error_h1*h1*(np.array(1)-h1)
            biass2 += error_h2*h2*(np.array(1)-h2)
            biass3 += error_h3 * h3 * (np.array(1) - h3)
            biass4 += error_o * o1 * (np.array(1) - o1)

    # with open("0w1.txt", 'w')as f:
    #     f.write('')
    # with open("0w1.txt", 'a')as f:
    #     for i in weights1:
    #         for x in i:
    #             f.write(str(x) + '\n')
    #
    # with open("0w2.txt", 'w')as f:
    #     f.write('')
    # with open("0w2.txt", 'a')as f:
    #     for i in weights2:
    #         for x in i:
    #             f.write(str(x) + '\n')
    # with open("0w3.txt", 'w')as f:
    #     f.write('')
    # with open("0w3.txt", 'a')as f:
    #     for i in weights2:
    #         for x in i:
    #             f.write(str(x) + '\n')
    # with open("0w4.txt", 'w')as f:
    #     f.write('')
    # with open("0w4.txt", 'a')as f:
    #     for i in weights2:
    #         for x in i:
    #             f.write(str(x) + '\n')

    print(o1[0][0])
    print(o1[0][1])
    print(o1[0][2])
    train = sharp
    h1 = relu(np.dot(train, weights1.T) + biass1)

    h2 = relu(np.dot(h1, weights2.T) + biass2)

    h3 = relu(np.dot(h2, weights3.T) + biass3)
    o1 = softmax(np.dot(h3, weights4.T) + biass4)
    print(o1[0][0])
    print(o1[0][1])
    print(o1[0][2])

cart()