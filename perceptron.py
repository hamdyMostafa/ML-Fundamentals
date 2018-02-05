# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:42:21 2018
Author: Hamdy Mostafa
"""


import numpy as np
import matplotlib.pyplot as plt


def F(x,W):
    return W[0]+W[1]*x


##  to Create the Target function , I create two random points and take th line passing through them

point_1 = np.random.uniform(low=-1.0, high=1.0, size=2)
point_2 = np.random.uniform(low=-1.0, high=1.0, size=2)


# A W = b --> W = inv(A) * b

A = np.array([[1,point_1[0]],[1,point_2[0]]])
b = np.array([point_1[1],point_2[1]])
W = np.linalg.solve(A,b)

# plot the two points and the line connecting them

plt.plot([point_1[0],point_2[0]],[point_1[1],point_2[0]],'ro')



xlim=[-1.1,1.1]
plt.xlim(xlim)
plt.ylim([-1.1,1.1])
plt.plot(xlim,[F(xlim[0],W),F(xlim[1],W)])



plt.xlabel('X1')
plt.ylabel('X2')
plt.grid()
plt.title('plotting the line going through two points')
plt.show()

