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

def nn_cost(nnParams, INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, regConst):
    theta1 = nnParams[0:HIDDEN_LAYER1*(INPUT_LAYER+1)]
    theta1 = np.reshape(theta1, (HIDDEN_LAYER1, INPUT_LAYER + 1))
    theta2 = nnParams[HIDDEN_LAYER1*(INPUT_LAYER+1):]
    theta2 = np.reshape(theta2, (OUTPUT_LAYER, HIDDEN_LAYER1 + 1,))
    m = len(members)

    #Switch to Correct Directory
    os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics")
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

            z3 = np.dot(theta2, a2)
            hX = sigmoid(z3)

            #Calculate cost
            curCost = -np.multiply(y, np.log(hX)) - np.multiply((1-y), np.log(1-hX))
            curCost = np.nan_to_num(curCost)
            curCost = np.sum(curCost.flatten())
            cost = cost + curCost
            picNum = picNum + 1

            if picNum == 1:
                break
        except:
            continue

    J = cost/picNum
    return J


def nnCostFunction(nnParams, INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, regConst):
    #Unrolling Thetas
    print "started cost fn"
    theta1 = nnParams[0:HIDDEN_LAYER1*(INPUT_LAYER+1)]
    theta1 = np.reshape(theta1, (HIDDEN_LAYER1, INPUT_LAYER + 1))
    theta2 = nnParams[HIDDEN_LAYER1*(INPUT_LAYER+1):]
    theta2 = np.reshape(theta2, (OUTPUT_LAYER, HIDDEN_LAYER1 + 1,))
    m = len(members)
    print 'thetas determined'

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

            z3 = np.dot(theta2, a2)
            hX = sigmoid(z3)

            #Calculate cost
            curCost = -np.multiply(y, np.log(hX)) - np.multiply((1-y), np.log(1-hX))
            curCost = np.nan_to_num(curCost)
            curCost = np.sum(curCost.flatten())
            cost = cost + curCost
            picNum = picNum + 1

            #Backward Propogation
            delta3 = hX - y
            temp2 = np.dot(np.transpose(theta2), delta3)
            delta2 = np.multiply(temp2, sigmoidGrad(z2))

            X = np.reshape(X, (len(X), 1))
            delta2 = np.reshape(delta2, (len(delta2), 1))
            DELTA1 = np.dot(delta2[1:,...], np.transpose(X))
            try:
                theta1_grad = theta1_grad + DELTA1
            except:
                theta1_grad = DELTA1.copy()

            a2 = np.reshape(a2, (len(a2), 1))
            delta3 = np.reshape(delta3, (len(delta3), 1))
            DELTA2 = np.dot(delta3, np.transpose(a2))
            try:
                theta2_grad = theta2_grad + DELTA2
            except:
                theta2_grad = DELTA2.copy()

            # if picNum == 1:
            #     break
            #print (m -picNum)
        except:
            continue

    J = cost/picNum# + regConst/(2*picNum)*(np.sum(np.square(theta1[..., 1:])) + np.sum(np.square(theta2[..., 1:])))
    theta1_grad = theta1_grad/picNum# + regConst*theta1
    #theta1_grad[...,1:] = (regConst/m)*theta1[..., 1:]
    theta2_grad = theta2_grad/picNum# + regConst*theta2
    #theta2_grad[...,1:] = (regConst/m)*theta2[..., 1:]
    grad = np.hstack([theta1_grad.flatten(), theta2_grad.flatten()])

    ##Gradient Checker
    # eps = 1e-4
    # gradCheck = grad.copy()
    #
    # numThetas =  len(nnParams)
    # for i in range(len(nnParams)):
    #     print numThetas - i
    #     tempPlus = nnParams
    #     tempMinus = nnParams
    #     tempPlus[i] = tempPlus[i] + eps
    #     tempMinus[i] = tempMinus[i] - eps
    #     costPlus = nn_cost(tempPlus, INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, regConst)
    #     costMinus = nn_cost(tempMinus, INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, regConst)
    #     gradCheck[i] = (costPlus - costMinus)/(2*eps)
    # print 'Gradient Error:',np.linalg.norm(gradCheck-grad)/np.linalg.norm(grad+gradCheck)
    print 'Cost:',J
    return J, grad

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
OUTPUT_LAYER = len(char_list) # = 20 as of now

os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2")
nnParams = np.loadtxt('thetaVals', delimiter = ',')
nnParams = nnParams/100
print nnParams.shape
theta1 = np.random.random((HIDDEN_LAYER1, INPUT_LAYER+1))
theta2 = np.random.random((OUTPUT_LAYER, HIDDEN_LAYER1+1))
os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics")
#nnParams = np.hstack([theta1.flatten(), theta2.flatten()])
regConst = 1

J, grad = nnCostFunction(nnParams, INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, regConst)
#print J, grad.shape

theta,minError,Info = opt.fmin_l_bfgs_b(nnCostFunction, x0=nnParams, args=(INPUT_LAYER, HIDDEN_LAYER1, OUTPUT_LAYER, members, char_list, regConst), fprime = None, maxfun = 5)
os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2")
np.savetxt('thetaVals', theta, delimiter=',')
print minError
print Info

#Unrolling Thetas
theta1 = nnParams[0:HIDDEN_LAYER1*(INPUT_LAYER+1)]
theta1 = np.reshape(theta1, (HIDDEN_LAYER1, INPUT_LAYER + 1))
theta2 = nnParams[HIDDEN_LAYER1*(INPUT_LAYER+1):]
theta2 = np.reshape(theta2, (OUTPUT_LAYER, HIDDEN_LAYER1 + 1,))



tar.close()
