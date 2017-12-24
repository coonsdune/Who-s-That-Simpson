import tarfile,os
import sys
from PIL import Image
import re
import numpy as np
import scipy.optimize as opt



#CREATE RE-SIZED IMAGES
tar = tarfile.open("simpsons_dataset.tar.gz")
os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\training_set")
members = tar.getnames()
m = len(members)
print m
# for member in members:
#     try:
#         os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\training_set")
#         member = member.replace("/", "\\")
#         image = Image.open(member)
#         image = image.convert('L')
#         img = image.resize((50, 50), Image.ANTIALIAS)
#         os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics")
#         img.save(member)
#         print "resized image %s" %(member)
#
#     except:
#         continue

#BEGIN NEURAL NET IMPLEMENTATION

###REQUIRED FUNCTIONS###
def charList_to_y(charList, char):
    n = len(charList)
    y = np.empty([n, 1])
    for i in range(n):
        name = charList[i]
        if (name == char):
            y[i] = 1
        else:
            y[i] = 0
    return y

def sigmoid(z):
    hX = 1/(1 + np.exp(-z))
    return hX

def sigmoidGrad(z):
    Z = np.append(np.ones((1, 1), dtype = float), z)
    sig = sigmoid(Z)
    sigGrad = np.multiply(sig, 1-sig)
    return sigGrad


def nnCostFunction(INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, nnParams):
    #Unrolling Thetas
    theta1 = nnParams[0:HIDDEN_LAYER1*(INPUT_LAYER+1)]
    theta1 = np.reshape(theta1, (HIDDEN_LAYER1, INPUT_LAYER + 1))
    theta2 = nnParams[HIDDEN_LAYER1*(INPUT_LAYER+1):]
    theta2 = np.reshape(theta2, (OUTPUT_LAYER, HIDDEN_LAYER1 + 1,))

    #Switch to Correct Directory
    os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics")
    #input_layer = np.zeros((INPUT_LAYER, 1))
    cost = 0
    picNum = 0

    for member in members:
        member = member.replace("/", "\\")

        try:
            #Generate X and y (Input and expected Output)
            img = Image.open(member)
            im_as_np = np.asarray(img)
            X = np.resize(im_as_np, (INPUT_LAYER,1))
            X = np.append(np.ones((1, 1), dtype = float), X) #Input layer

            char = member.split('\\')
            char = char[0]
            y = charList_to_y(char_list, char)
            y = y.flatten()

            #Forward Propogation
            z2 = np.dot(theta1, X)
            a2 = sigmoid(z2)
            a2 = np.append(np.ones((1, 1), dtype = float), a2) #Hidden layer 1

            # z3 = np.dot(theta2, a2)
            # a3 = sigmoid(z3)
            # a3 = np.append(np.ones((1, 1), dtype = float), a3) #Hidden layer 2
            #
            # z4 = np.dot(theta3, a3)
            # hX = sigmoid(z4) #Output layer

            z3 = np.dot(theta2, a2)
            hX = sigmoid(z3)

            #Calculate cost
            curCost = -np.multiply(y, np.log(hX)) - np.multiply((1-y), np.log(1-hX))
            curCost = np.nan_to_num(curCost)
            curCost = np.sum(curCost.flatten())
            cost = cost + curCost
            picNum = picNum + 1


            #Backward Propogation
            # delta4 = hX - y
            # temp3 = np.dot(np.transpose(theta3), delta4)
            # delta3 = np.multiply(temp3, sigmoidGrad(z3))
            # temp2 = np.dot(np.transpose(theta2), delta3[1:,...])
            # delta2 = np.multiply(temp2, sigmoidGrad(z2))
            delta3 = hX - y
            temp2 = np.dot(np.transpose(theta2), delta3)
            delta2 = np.multiply(temp2, sigmoidGrad(z2))

            X = np.reshape(X, (len(X), 1))
            delta2 = np.reshape(delta2, (len(delta2), 1))
            DELTA1 = np.dot(delta2[1:,...], np.transpose(X))
            try:
                theta1_grad = theta1_grad + DELTA1
            except:
                theta1_grad = DELTA1

            a2 = np.reshape(a2, (len(a2), 1))
            delta3 = np.reshape(delta3, (len(delta3), 1))
            DELTA2 = np.dot(delta3, np.transpose(a2))
            try:
                theta2_grad = theta2_grad + DELTA2
            except:
                theta2_grad = DELTA2

            print picNum

        except:
            continue

    J = cost/picNum
    theta1_grad = theta1_grad/picNum
    theta2_grad = theta2_grad/picNum
    return J, theta1_grad, theta2_grad

################################################################################

#Get Name list from directory
os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics")
directory_list = list()
char_list = list()
for root, dirs, files in os.walk("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics", topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))
        char_list.append(name)
n = len(char_list)

#initialize the parameters for the architecture of the neural network
INPUT_LAYER = 50 * 50
HIDDEN_LAYER1 = 10 * 10
#HIDDEN_LAYER2 = 10 * 10
OUTPUT_LAYER = len(char_list) # = 20 as of now

theta1 = np.random.random((HIDDEN_LAYER1, INPUT_LAYER+1))
#theta2 = np.random.random((HIDDEN_LAYER2, HIDDEN_LAYER1+1))
#theta3 = np.random.random((OUTPUT_LAYER, HIDDEN_LAYER2+1))
theta2 = np.random.random((OUTPUT_LAYER, HIDDEN_LAYER1+1))
nnParams = np.hstack([theta1.flatten(), theta2.flatten()])

J, theta1_grad, theta2_grad = nnCostFunction(INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, nnParams)
print J, theta1_grad.shape, theta2_grad.shape

# ROLLING AND UNROLLING GRADIENTS
# a = theta1_grad.flatten()
# b = theta2_grad.flatten()
# c = np.hstack([a, b])
# p = c[0:HIDDEN_LAYER1*(INPUT_LAYER+1)]
# p = np.reshape(p, (HIDDEN_LAYER1, INPUT_LAYER + 1))
# q = c[HIDDEN_LAYER1*(INPUT_LAYER+1):]
# q = np.reshape(q, (OUTPUT_LAYER, HIDDEN_LAYER1 + 1,))




tar.close()
