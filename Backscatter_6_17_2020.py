# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:29:12 2020

@author: tjmayer
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image  
from sklearn.metrics import r2_score
import seaborn as sns; sns.set(color_codes=True)
import imageio
from scipy.interpolate import *
from scipy.stats import *
from scipy import ndimage, misc
import matplotlib.lines as lines

path = r'C:\Users\tjmayer\Desktop\GEDI_Work\BackScatter'

##Tropical Dry Broadleaf
A = 0.021563
B = 0.042324
C = 0.027519
##########################################################################

def func(x, a, b, c):
    return a * (1 - np.exp(-b * x * (c )))
##########################################################################
########### x data  ###LIDAR
im = Image.open(path + r'\hmean_degree_setnull.tif')
print("im", im.size)

xdata = np.array(im)  

#print("xdata Nan?", np.isnan(xdata).any())   #this prints False
#print("xdata inf", np.isinf(xdata).any())
#
#print("xdata:", xdata)
#print("xdata size:", xdata.size)
#print("xdata shape:", xdata.shape)
#
#print("xdata:", xdata)
#print("xdata.ravel() size:", xdata.ravel().size)
#print("xdata.ravel() shape:", xdata.ravel().shape)

x_ravel = xdata.ravel()

##########################################################################
########## y data  ###SAR
im_sar = Image.open(path + r'\Gamma_pw_clip_snap.tif')

y_sar = np.array(im_sar)

#print("y_sar Nan?", np.isnan(y_sar).any())   #this prints False
#print("y_sar inf", np.isinf(y_sar).any())
#
#print("y_sar:", y_sar)
#print("y_sar size:", y_sar.size)
#print("y_sar shape:", y_sar.shape)

ydata = y_sar
y_ravel = ydata.ravel()

#########################################################################
#ignore nan values in both inputs by creating a mask called "bad"  (potentially add inf into this)
bad = ~np.logical_or(np.isnan(x_ravel), np.isnan(y_ravel))
print("bad nan", bad.size)
# compress the ravelled x and y data with the 'bad' mask
#to create two new datasets xnew and ynew that should match in size and shape
xnew = np.compress(bad, xdata)
ynew = np.compress(bad, ydata)

print("y new size:", ynew.size)
print("y new shape:", ynew.shape)

print("x new size:", xnew.size)
print("x new shape:", xnew.shape)

############## manual filter 
filter_arr = ynew > -1
y = ynew[filter_arr]
print(filter_arr)
print("y", y.size)

print("y Nan?", np.isnan(y).any())   #this prints False
print("y inf", np.isinf(y).any())
###
filter_arr_x = xnew > -10
x = xnew[filter_arr_x]
print(filter_arr_x)
print("x", x.size)

print("x Nan?", np.isnan(x).any())   #this prints False
print("x inf", np.isinf(x).any())
#########################################################################
#jointhist = sns.jointplot(xnew, ynew); #kind="hex"; kind="kde"

#########################################################################
###curvefit--parameters
##func = function name
##xnew = what is being used to predict the value
##ynew = measuered values that We will try and fit it to
##c = covarince
##pcov = paramter values

#########
c, pcov = curve_fit(func, xnew, ynew, maxfev=50000000) #maxfev=50000000
print("covarince values:", c)

#length = len(xnew)
#y_empty = np.empty(length)
#for i in range (length):
#    y_empty[i] = func(xnew[i],c[0],c[1],c[2])

length = len(ynew)
y_empty = np.empty(length)
for i in range (length):
    y_empty[i] = func(ynew[i],c[0],c[1],c[2])


fig = plt.figure()
plt.scatter(xnew, ynew) #, c=colors, alpha=0.5
plt.plot(xnew, y_empty,'ro', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(c))
plt.plot( [0,1],[0,1], 'g', transform=fig.transFigure, figure=fig )
plt.xlabel('x LIDAR')
plt.ylabel('y SAR inverted height')
plt.legend()
plt.show()

R2 = r2_score(y_empty,ynew)
print("R^2", R2)

######################################################################
print("end script")