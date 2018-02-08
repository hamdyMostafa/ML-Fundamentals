# -*- coding: utf-8 -*-
"""
Author: Hamdy Mostafa
"""
import numpy as np
import matplotlib.pyplot as plt
import random


def F(x,W):
    return W[0]+W[1]*x

# Target_ function
def target_line():
    
    # I create two random points and take th line passing through them X = [-1,1] * [-1,1]
    point_1 = np.random.uniform(low=-1.0, high=1.0, size=2)
    point_2 = np.random.uniform(low=-1.0, high=1.0, size=2)

    # A W = b --> W = inv(A) * b
    A = np.array([[1,point_1[0]],[1,point_2[0]]])
    b = np.array([point_1[1],point_2[1]])
    W = np.linalg.solve(A,b)
    plt.plot([point_1[0],point_2[0]],[point_1[1],point_2[1]],'ro')
    return (np.array([-1,-W[1]/W[0], 1/W[0]]),W)
        
## Creating the dataSet
def prediction(features,W):
    '''classify points '''
    labels = np.sign(features.dot(W))
    return labels  


def generate_data(N,W):
    ''' generate data , x0 = 1, x1 & x2 random , label '''
    data_coor = np.random.uniform(-1., 1., size=(N,2))
    features =  np.column_stack([np.ones(N), data_coor])
    labels = prediction(features,W)
    labels = np.reshape(labels, (N,1))
    data = np.concatenate((features,labels),axis = 1)
    
    return data


# Perceptron learning Algorithm
def perceptron(data):
    W = np.zeros(3)  
    features = data[:,:-1]
    T_labels = data[:,-1]
    while True: 
        H_labels = prediction(features,W)
        
        # choose a misclassified random point
        misClassified = list(np.where(np.not_equal(T_labels, H_labels))[0])
        if len(misClassified) == 0:
            break
        i =  random.choice(misClassified)
        
        # Update the weight vector
        W = W + T_labels[i]*features[i]
    
    return W
        


def plot_perceptron(data,W_t , W_h):
    plt.scatter(data[:,1],data[:,2],c=data[:,3])
    xlim=[-1.1,1.1]
    plt.xlim(xlim)
    plt.ylim([-1.1,1.1])
    
    plt.plot(xlim,[F(xlim[0],W_t),F(xlim[1],W_t)])
    plt.plot(xlim,[F(xlim[0],W_h),F(xlim[1],W_h)])
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid()
    plt.show()
    

## Implementation
W,W_t = target_line() 
data = generate_data(1000,W) 

W_h = perceptron(data) 
W_h_p = (1/W_h[2],-W_h[1]/W_h[2] )

 
plot_perceptron(data,W_t , W_h_p)










