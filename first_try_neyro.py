import numpy as np
from matplotlib import pyplot as plt
import cv2
from time import time
#
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
#
#
# training_inputs = np.array([[0,0,0],
#                             [1,1,1],
#                             [1,0,1],
#                             [0,1,1]])
#
# training_outputs = np.array([[0,1,1,0]]).T
# weights = 2* np.random.random((3,1)) -1
#
#
#
# for i in range(2000):
#     input_layer = training_inputs
#     outputs = sigmoid(np.dot(input_layer,weights))
#
#     err = training_outputs - outputs
#     adj = np.dot(input_layer.T,err(outputs(1-outputs)))
#     weights += adj
#
#
#
# print(outputs)


#
# def sigmoid(x):
#     return 1/(1+np.exp(-x))


# training_inputs = np.array([[0,1,0,1],
#                             [1,0,0,0],
#                             [1,0,1,1],
#                             [0,1,0,0]])
#
#
# training_outputs = np.array([[0,1,1,0]]).T
#
# weights = 2* np.random.random((4,1))-1
#
# biass = 1
#
#
# for i in range(10):
#     input_layer = training_inputs
#
#     output = sigmoid(np.dot(input_layer,weights)+1)
#     err = training_outputs - output
#     print(err)
#     break
#     # adj = np.dot(input_layer.T,err*(output(1-output)))
#     # weights += adj




# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# train_inputs = np.array([[1,0,1],
#                          [0,0,0],
#                          [1,0,0],
#                          [1,0,1]])
#
# train_outputs = np.array([[1,0,1,1]]).T
#
# weights = 2*np.random.random((3,1))-1
# weights2 = 2*np.random.random((3,1))-1
# weights3 = 2*np.random.random((1,1))-1
# weights4 = 2*np.random.random((1,1))-1
#
# b = 1
#
# for i in range(20000):
#     houtputs1 = sigmoid(np.dot(train_inputs,weights)+b)
#     houtputs2 = sigmoid(np.dot(train_inputs, weights2) + b)
#     output = sigmoid(np.dot(houtputs1,weights3)+np.dot(houtputs2,weights4)+b)
#     weights += np.dot(train_inputs.T,(train_outputs-outputs)*(outputs*(1-outputs)))
# print(output)

################################################################################################3

# input_train1 = 1
# input_train2 = 0
# input_train3 = 1
#
# output_train1 = 1
# output_train2 = 0
#
#
# weight = 2*np.random.random() - 1
# weight2 = 2*np.random.random() - 1
# weight3 = 2*np.random.random() - 1
# weight4 = 2*np.random.random() - 1
# weight5 = 2*np.random.random() - 1
# weight6 = 2*np.random.random() - 1
# b1 = 1
# b2 = 1
#
#
# for i in range(1000):
#
#     o1 = sigmoid(input_train1*weight+input_train2*weight2+input_train3*weight3+b1)
#     o2 = sigmoid(input_train1 * weight4 + input_train2 * weight5 + input_train3 * weight6 + b2)
#     derivative1 = o1*(1-o1)
#     derivative2 = o1 * (1 - o1)
#     error1 = output_train1  - o1
#     error2 = output_train2 - o2
#     weight = weight + error1*derivative1*input_train1
#     weight2 = weight2 + error1 * derivative1 * input_train2
#     weight3 = weight3 +  error1 * derivative1 * input_train3
#     b1 = b1 + error1 *derivative1
#     weight4 = weight4 + error2 * derivative2 * input_train1
#     weight5 = weight5 + error2 * derivative2 * input_train2
#     weight6 = weight6 + error2 * derivative2 * input_train3
#     b1 = b1 + error2 * derivative2
#
#     print(o1)
#     print(o2)

#############################################################################################   успешная


############################################################################################################
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
#
#
# input_train1 = 1
# input_train2 = 0
# input_train3 = 1
#
# output_train1 = 0
# output_train2 = 1
#
# weight1 = 2*np.random.random() - 1
# weight2 = 2*np.random.random() - 1
# weight3 = 2*np.random.random() - 1
# weight4 = 2*np.random.random() - 1
# weight5 = 2*np.random.random() - 1
#
# b1 = 1
# b2 = 1
# b3 = 1
#
# for i in range(10000):
#     h1 = sigmoid(input_train1*weight1+input_train2*weight2+input_train3*weight3+b1)
#     o1 = sigmoid(h1*weight4+b2)
#     o2 = sigmoid(h1*weight5+b3)
#     error1 = output_train1 - o1
#     error2 = output_train2 - o2
#
#     errorh1 = weight4*(output_train1 - o1) + weight5*(output_train2 - o2)
#     derivative_h1 =h1*(1-h1)
#     derivative_o1 = o1*(1-o1)
#     derivative_o2 = o2*(1-o2)
#     weight1 = weight1 + errorh1*derivative_h1*input_train1
#     weight2 = weight2 + errorh1 * derivative_h1 * input_train2
#     weight3 = weight3 + errorh1 * derivative_h1 * input_train3
#
#     weight4 = weight4 + error1*derivative_o1*h1
#     weight5 = weight5 + error2*derivative_o2*h1
#
#     b1 = b1 + errorh1*derivative_h1
#     b2 = b2 + error1*derivative_o1
#     b3 = b3 + error2*derivative_o2
#
# print(o1,o2)
################################################################################################ еще одна успешная

##################################################################################################
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
#
#
# input_train1 = 1
# input_train2 = 0
# input_train3 = 1
#
# output_train1 =1
#
#
#
# weight1 = 2*np.random.random() - 1
# weight2 = 2*np.random.random() - 1
# weight3 = 2*np.random.random() - 1
# weight4 = 2*np.random.random() - 1
# weight5 = 2*np.random.random() - 1
# weight6 = 2*np.random.random() - 1
#
# weight7 = 2*np.random.random() - 1
# weight8 = 2*np.random.random() - 1
# weight9 = 2*np.random.random() - 1
# weight10 = 2*np.random.random() - 1
# weight11 = 2*np.random.random() - 1
# weight12 = 2*np.random.random() - 1
#
# weight13 = 2*np.random.random() - 1
# weight14 = 2*np.random.random() - 1
# weight15 = 2*np.random.random() - 1
#
# b1 = 1
# b2 = 1
# b3 = 1
# b4 = 1
# b5 = 1
# b6 = 1
#
#
# for i in range(1000):
#     h1 = sigmoid(input_train1*weight1+input_train2*weight2+input_train3*weight3+b1)
#     h2 = sigmoid(input_train1 * weight4 + input_train2 * weight5 + input_train3 * weight6 + b2)
#
#     h3 = sigmoid(h1 * weight7 + h2 * weight8  + b3)
#     h4 = sigmoid(h1 * weight9 + h2 * weight10  + b4)
#     h5 = sigmoid(h1 * weight11 + h2 * weight12 + b5)
#
#     o1 = sigmoid(h3*weight13+h4*weight14+h5*weight15+b6)
#     error_o1 = output_train1-o1
#     error_h5 = error_o1*weight15
#     error_h4 = error_o1*weight14
#     error_h3 = error_o1*weight13
#     error_h2 = error_h3*weight8+error_h4*weight10+error_h5*weight12
#     error_h1 = error_h3*weight7+error_h4*weight9*error_h5*weight11
#
#     derivate_o1 = o1*(1-o1)
#     derivate_h5 = h5*(1-h5)
#     derivate_h4 = h4 * (1 - h4)
#     derivate_h3 = h3 * (1 - h3)
#     derivate_h2 = h2 * (1 - h2)
#     derivate_h1 = h1 * (1 - h1)
#
#
#     weight1 += error_h1*derivate_h1*input_train1
#     weight2 += error_h1*derivate_h1*input_train2
#     weight3 += error_h1 * derivate_h1 * input_train3
#     weight4 += error_h2 * derivate_h2 * input_train1
#     weight5 += error_h2 * derivate_h2 * input_train2
#     weight6 += error_h2 * derivate_h2 * input_train3
#
#     weight7 += error_h3*derivate_h3*h1
#     weight8 += error_h3*derivate_h3*h2
#     weight9 += error_h4*derivate_h4*h1
#     weight10 += error_h4 * derivate_h4 * h2
#     weight11 += error_h5*derivate_h5*h1
#     weight12 += error_h5 * derivate_h5 * h2
#
#     weight13 += error_o1*derivate_o1*h3
#     weight14 += error_o1*derivate_o1*h4
#     weight15 += error_o1*derivate_o1*h5
#
#     b1 += error_h1*derivate_h1
#     b2 += error_h2*derivate_h2
#     b3 += error_h3*derivate_h3
#     b4 += error_h4*derivate_h4
#     b5 += error_h5*derivate_h5
#     b6 += error_o1*derivate_o1
#
#     print(o1)

###################################################################################### успешно)

# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# train_input = np.array([[1,0,1]])
#
# weights = 2*np.random.random((3,2))-1
#
# train_output = np.array([1,0])
#
# biass = np.array([1,1])
#
# for i in range(10000):
#     output = sigmoid(np.dot(train_input,weights))
#     error = train_output-output
#     weights += np.dot(train_input.T,error*(output*(1-output)))
# print(output)

###################################################################################### успешно)



# def sigmoid(x):
#     for i in x:
#         return 1/(1+np.exp(-i))
#
# train_input = np.array([[1,0,1]])
#
# weights1 = 2*np.random.random((3,1))-1
# weights2 = 2*np.random.random((1,2))-1
#
# train_output = np.array([0,0])
#
# biass = np.array([1])
# biass2 = np.array([1,1])
#
# for i in range(20000):
#     h1 = sigmoid(np.dot(train_input,weights1)+biass)
#     o = sigmoid((h1*weights2)+biass2)
#     error = train_output - o
#     error_h1 = np.dot(error,weights2.T)
#     weights1 += train_input.T*error_h1*h1*(1-h1)
#     weights2 += h1*error*o*(1-o)
# print(o)

################################################################## Успеххххххх




# def sigmoid(x):
#     ret = []
#     for w in x:
#
#         ret.append(1/(1+np.exp(-w)))
#     return ret
# bads = []
#
# train_input = np.array([[1,0,1],
#                         [0,1,1],
#                         [0,0,0],
#                         [1,0,0]])
#
# weights1 = 2*np.random.random((3,2))-1
# weights2 = 2*np.random.random((2,3))-1
# weights3 = 2*np.random.random((3,1))-1
#
#
# train_output = np.array([1,0,0,1])
#
# biass = np.array([1,1])
# biass2 = np.array([1,1,1])
# biass3 = np.array([1])
#
#
#
#
# for i in range(10000):
#     for tes in range(len(train_input)):
#         h1 = sigmoid(np.dot(train_input[tes],weights1)+biass)
#         h2 = sigmoid(np.dot(h1,weights2)+biass2)
#         o = sigmoid(np.dot(h2,weights3)+biass3)
#         error_o = train_output[tes] - o
#
#         error_h2 = np.dot(error_o,weights3.T)
#         error_h1 = np.dot(error_h2,weights2.T)
#         weights1 += np.array([train_input[tes]]).T*error_h1*h1*(np.array([1])-h1)
#
#
#         weights2 += np.array([h1]).T*error_h2*h2*(np.array([1])-h2)
#         weights3 += np.array([h2]).T*error_o*o*(np.array([1])-o)
#     bads.append(np.fabs(error_o))
# y = [x for x in range(len(bads))]
# plt.plot(y,bads,'-r')
# plt.show()
# h1 = sigmoid(np.dot([0,0,0],weights1)+biass)
# h2 = sigmoid(np.dot(h1,weights2)+biass2)
# o = sigmoid(np.dot(h2,weights3)+biass3)
# print(o)
##########################################33    3######################33
#
# image2 = cv2.imread('1.png')
# image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
# image2 = cv2.resize(image2,(20,20))
# sharp2 = np.array([image2.reshape(400)/255])
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
#     ret = []
#     for w in x:
#
#         ret.append(1/(1+np.exp(-w)))
#     return ret
#
# train_input = np.append(sharp,sharp2,axis=0)
# train_input = np.append(train_input,sharp3,axis=0)
#
#
#
# weights1 = 2*np.random.random((400,10),)-1
# weights2 = 2*np.random.random((10,3))-1
# # weights1 = np.array([float(x) for x in open('0w1.txt','r').read().split('\n') if x != ""]).reshape(400,10)
# # weights2 = np.array([float(x) for x in open('0w2.txt','r').read().split('\n') if x != ""]).reshape(10,2)
#
# train_output = np.array([[1,0,0],[0,1,0],[0,0,1]])
#
# # biass = np.array([1,1,1,1,1,1,1,1,1,1])
# # biass2 = np.array([1,1,1])
# #
# for i in range(1000):
#     for tes in range(len(train_input)):
#         train = np.array([train_input[tes]])
#         traino = np.array([train_output[tes]])
#         h = sigmoid(np.dot(train,weights1))
#
#         o = sigmoid(np.dot(h,weights2))
#
#         error_o = traino-o
#         error_h = np.dot(error_o,weights2.T)
#
#         weights1 += np.dot(train.T,error_h*h*(np.array([1])-h))
#
#
#         weights2 += np.dot(np.array(h).T,error_o*o*(np.array([1])-o))

#############################################  тест
# image2 = cv2.imread('0.png')
# image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
# image2 = cv2.resize(image2,(20,20))
# sharp2 = np.array([image2.reshape(400)/255])
# train = sharp2
# h = sigmoid(np.dot(train,weights1)+biass)
#
# o = sigmoid(np.dot(h,weights2)+biass2)
# print(o)
#############################################  тест


#
#
# with open("0w1.txt",'w')as f:
#     f.write('')
# with open("0w1.txt",'a')as f:
#     for i in weights1:
#         for x in i:
#
#
#             f.write(str(x)+'\n')
#
#
# with open("0w2.txt",'w')as f:
#     f.write('')
# with open("0w2.txt",'a')as f:
#     for i in weights2:
#         for x in i:
#
#
#             f.write(str(x)+'\n')

###################################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))



input_train1 = 1
input_train2 = 0
input_train3 = 1

output_train1 =1



weight1 = 0.1
weight2 = 0.2
weight3 = 0.3
weight4 = 0.4
weight5 = 0.2
weight6 = 0.3
weight7 = 0.3
weight8 = 0.4
weight9 = 0.1
weight10 = 0.1
weight11 = 0.5
weight12 = 0.1
weight13 = 0.1
weight14 = 0.2
weight15 = 0.3

b1 = 1
b2 = 1
b3 = 1
b4 = 1
b5 = 1
b6 = 1


for i in range(1000):
    h1 = sigmoid(input_train1*weight1+input_train2*weight2+input_train3*weight3+b1)
    h2 = sigmoid(input_train1 * weight4 + input_train2 * weight5 + input_train3 * weight6 + b2)

    h3 = sigmoid(h1 * weight7 + h2 * weight8  + b3)
    h4 = sigmoid(h1 * weight9 + h2 * weight10  + b4)
    h5 = sigmoid(h1 * weight11 + h2 * weight12 + b5)

    o1 = sigmoid(h3*weight13+h4*weight14+h5*weight15+b6)
    error_o1 = output_train1-o1
    error_h5 = error_o1*weight15
    error_h4 = error_o1*weight14
    error_h3 = error_o1*weight13
    error_h2 = error_h3*weight8+error_h4*weight10+error_h5*weight12
    error_h1 = error_h3*weight7+error_h4*weight9+error_h5*weight11
    # print(error_h1,error_h2)
    # break

    derivate_o1 = o1*(1-o1)
    derivate_h5 = h5*(1-h5)
    derivate_h4 = h4 * (1 - h4)
    derivate_h3 = h3 * (1 - h3)
    derivate_h2 = h2 * (1 - h2)
    derivate_h1 = h1 * (1 - h1)


    weight1 += error_h1*derivate_h1*input_train1
    weight2 += error_h1*derivate_h1*input_train2
    weight3 += error_h1 * derivate_h1 * input_train3
    weight4 += error_h2 * derivate_h2 * input_train1
    weight5 += error_h2 * derivate_h2 * input_train2
    weight6 += error_h2 * derivate_h2 * input_train3
    # print(weight1,weight2,weight3,weight4,weight5,weight6)
    # break
    weight7 += error_h3*derivate_h3*h1
    weight8 += error_h3*derivate_h3*h2
    weight9 += error_h4*derivate_h4*h1
    weight10 += error_h4 * derivate_h4 * h2
    weight11 += error_h5*derivate_h5*h1
    weight12 += error_h5 * derivate_h5 * h2
    # print(weight7 ,weight8 ,weight9 ,weight10,weight11,weight12)
    # break
    weight13 += error_o1*derivate_o1*h3
    weight14 += error_o1*derivate_o1*h4
    weight15 += error_o1*derivate_o1*h5
    # print(weight13,weight14,weight15)

    b1 += error_h1*derivate_h1
    b2 += error_h2*derivate_h2
    b3 += error_h3*derivate_h3
    b4 += error_h4*derivate_h4
    b5 += error_h5*derivate_h5
    b6 += error_o1*derivate_o1
    print(b6)
    break
print(o1)