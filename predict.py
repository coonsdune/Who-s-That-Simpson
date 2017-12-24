import tarfile,os
import sys
from PIL import Image
import re
import numpy as np
import scipy.optimize as opt

def sigmoid(z):
    hX = 1/(1 + np.exp(-z))
    return hX

os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2")
nnParams = np.loadtxt('thetaVals', delimiter = ',')

#Get Name list from directory
os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics")
directory_list = list()
char_list = list()
for root, dirs, files in os.walk("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\\resized pics", topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))
        char_list.append(name)
n = len(char_list)

INPUT_LAYER = 50 * 50
HIDDEN_LAYER1 = 10 * 10
OUTPUT_LAYER = len(char_list) # = 20 as of now

theta1 = nnParams[0:HIDDEN_LAYER1*(INPUT_LAYER+1)]
theta1 = np.reshape(theta1, (HIDDEN_LAYER1, INPUT_LAYER + 1))
theta2 = nnParams[HIDDEN_LAYER1*(INPUT_LAYER+1):]
theta2 = np.reshape(theta2, (OUTPUT_LAYER, HIDDEN_LAYER1 + 1,))

os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\kaggle_simpson_testset")
img = Image.open('chief_wiggum_2.jpg')
img = img.convert('L')
img = img.resize((50, 50), Image.ANTIALIAS)
os.chdir("C:\Users\Gautam\Desktop\Python-Coursera\ml\simpsons identification 2\kaggle_simpson_testset_gs")
img.save('chief_wiggum_2.jpg')
img = Image.open('chief_wiggum_2.jpg')
im_as_np = np.asarray(img)
X = np.resize(im_as_np, (INPUT_LAYER,1))
X = np.append(np.ones((1, 1), dtype = float), X)
print 'X',X
print 'theta1:',theta1

z2 = np.dot(theta1, X)
a2 = sigmoid(z2)
a2 = np.append(np.ones((1, 1), dtype = float), a2) #Hidden layer 1
print 'a2:',a2

z3 = np.dot(theta2, a2)
hX = sigmoid(z3)

print hX
print char_list
