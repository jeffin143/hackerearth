#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:10:33 2017

@author: jeffin
"""


import scipy.io
import numpy as np
from scipy.optimize import minimize
import csv
import cv2
import os



def sigmoidGradient(z):
    
    g = 1.0 / (1.0 + np.exp(-z))
    g = g*(1-g)

    return g

    


def sigmoid(z):


    from scipy.special import expit 
    
     
    g = np.zeros(z.shape)



    # g = 1/(1 + np.exp(-z))
    g = expit(z)

    return g

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
	num_labels, X, y, lambda_reg):
    #separating theta1 and theta2 so that i can do computation
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                     (hidden_layer_size, input_layer_size + 1), order='F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                     (num_labels, hidden_layer_size + 1), order='F')

    
    m = len(X)
             
    
    J = 0;
    Theta1_grad = np.zeros( Theta1.shape )
    Theta2_grad = np.zeros( Theta2.shape )


    X = np.column_stack((np.ones((m,1)), X)) # = a1
    a2 = sigmoid( np.dot(X,Theta1.T) )

    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
    a3 = sigmoid( np.dot(a2,Theta2.T) )


    labels = y
    y = np.zeros((m,num_labels),int)
    for i in range(m):
    	y[i, int(labels[i])] = 1

    #calcualting cost    

    cost = 0
    for i in range(m):
    	cost += np.sum( y[i] * np.log( a3[i] ) + (1 - y[i]) * np.log( 1 - a3[i] ) )

    J = -(1.0/m)*cost

    # REGULARIZED COST FUNCTION
    

    sumOfTheta1 = np.sum(np.sum(Theta1[:,1:]**2))
    sumOfTheta2 = np.sum(np.sum(Theta2[:,1:]**2))

    J = J + ( (lambda_reg/(2.0*m))*(sumOfTheta1+sumOfTheta2) )

    # BACKPROPAGATION

    bigDelta1 = 0
    bigDelta2 = 0

    # for each training example
    for t in range(m):


    	x = X[t]
    	a2=sigmoid(np.dot(x,Theta1.T))
    	a2=np.concatenate((np.array([1]),a2))
    	a3=sigmoid(np.dot(a2,Theta2.T))
    	
    	delta3 = np.zeros((num_labels))

    	for k in range(num_labels):
            y_k = y[t, k]
            delta3[k] = a3[k] - y_k

    	
    	delta2 = (np.dot(Theta2[:,1:].T, delta3).T) * sg.sigmoidGradient( np.dot(x, Theta1.T) )

    	
    	bigDelta1 += np.outer(delta2, x)
    	bigDelta2 += np.outer(delta3, a2)


    
    Theta1_grad = bigDelta1 / m
    Theta2_grad = bigDelta2 / m

    # REGULARIZATION FOR GRADIENT
    # only regularize for j >= 1, so skip the first column
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1_grad += (float(lambda_reg)/m)*Theta1
    Theta2_grad += (float(lambda_reg)/m)*Theta2
    Theta1_grad[:,0] = Theta1_grad_unregularized[:,0]
    Theta2_grad[:,0] = Theta2_grad_unregularized[:,0]

    

    

    
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad





def randInitializeWeights(L_in, L_out):
     
    W = np.zeros((L_out, 1 + L_in))

    

    # Randomly initialize the weights to small values
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init) - epsilon_init
    W=W*0.001

    return W

    

def predict(Theta1, Theta2, X):

    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))

    
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros((m,1))

    h1 = sigmoid( np.dot( np.column_stack( ( np.ones((m,1)), X ) ) , Theta1.T ) )
    h2 =sigmoid( np.dot( np.column_stack( ( np.ones((m,1)), h1) ) , Theta2.T ) )

    p = np.argmax(h2, axis=1)

    
    return p + 1 # offsets python's zero notation



## Seting up the parameters i will use for this exercise
input_layer_size  = 65536 # 256x256 pixels features of images
hidden_layer_size = 1000   # 1000 hidden units
num_labels = 25          # 25 labels(categories), from 0 to 24   

# Load Training Data
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
        	images.append(img)
    #for i in range(1,200):       cv2.imshow(i,images[i]) (Not working)

    X=np.reshape(images, (3215,-1))  
    print("images unrolled in a vector of  gey scale features", len(X[0]))
    return X

X=load_images_from_folder("train_img")
print("no of images  ", len(X)) #total no of images unrolled in a matrix


#truth values for all the images
"""with open('train.csv', 'r') as f:  #actual truth values
  reader = csv.reader(f)
  truth = list(reader)
print("no of truth values  ", len(truth))
print('Loading and Visualizing Data ...')
"""
with open('truth1.csv', 'r') as f:  #since working in string is difficult so converted categories 
  reader = csv.reader(f)			#to number
  testing = list(reader)
print("no of truth values  ", len(testing))


#number of unique categories
print("unique characteres",np.unique(testing))


m = X.shape[0]


# changes the dimension from (m,1) to (m,)
# otherwise the minimization and further step is difficult since ['x'] input will be taken

#y=np.testing.flatten() 
testing=np.array(testing)
y=testing.flatten('C')


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'), initial_Theta2.reshape(initial_Theta2.size, order='F')))
print('done ')



print('Training Neural Network... do wait')

#using bfgs algorithim
maxiter = 5
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
results = minimize(nnCostFunction, x0=initial_nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Obtaining Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                 (hidden_layer_size, input_layer_size + 1), order='F')

Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                 (num_labels, hidden_layer_size + 1), order='F')

input('Program paused. Press enter to continue.\n')
#computing the actual thing i mean algorithim testing
pred = predict(Theta1, Theta2, X)


print('Training Set Accuracy: {:f}'.format( ( np.mean(pred == y)*100 ) ) )




