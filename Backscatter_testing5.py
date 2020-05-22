# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:25:38 2020

@author: tjmayer
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
#from PIL import Image  ### uncomment if reading in tif for heights


#https://stackoverflow.com/questions/40265960/curve-fit-using-python
## example
#def func(x, a, b, c):
#    return a * np.exp(-b * x) + c

#######################
#Lei, Y., Siqueira, P., Torbick, N., Ducey, M., Chowdhury, D., & Salas, W. (2018). Generation of Large-Scale Moderate-Resolution Forest Height Mosaic With Spaceborne Repeat-Pass SAR Interferometry and Lidar. IEEE Transactions on Geoscience and Remote Sensing, 57(2), 770-787.
## y0=A(1-e-BhvC)
# fix this equation c should be in the exp ?

def func(x, a, b, c):
    return a * (1 - np.exp(-b * x* np.exp(c)))

################3
###Get hv tree heights; read in a tif image file and  show, 
              
#im = Image.open('Height.tif') ### update as needed, move the .tif file in the same location as the script
###uncomment to show the tif file as a png
#im.show()
    
##################
    
##Working (not functioning because tif isn't editable) to replace NAN with mean value
#Obtain mean of columns as you need, nanmean is convenient.
#col_mean = np.nanmean(im, axis=0)
#print("col_mean", col_mean)
#
##Find indices that you need to replace
#inds = np.where(np.isnan(im))
#
##Place column means in the indices. Align the arrays using take
#im[inds] = np.take(col_mean, inds[1])
#
#print(im)

###################
##function variables
#hv = np.array(im) ### code used for pulling values from a tif

hv = np.linspace(0.1,1.1,101) ### dummy data


#A,B,C are asuumed constant for a generic type of forest over a wide area
#Lei, Y., Siqueira, P., Torbick, N., Ducey, M., Chowdhury, D., & Salas, W. (2018). Generation of Large-Scale Moderate-Resolution Forest Height Mosaic With Spaceborne Repeat-Pass SAR Interferometry and Lidar. IEEE Transactions on Geoscience and Remote Sensing, 57(2), 770-787.
A = 2.5
B = 1.3
C = 0.5

y = func(hv, A, B, C)
#plt.plot(y, 'o')

ydata = y + 0.2 * np.random.normal(size=len(hv))
#plt.plot(ydata,'o')

##
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
popt, pcov = curve_fit(func, hv, ydata, p0=[2,1,1])

plt.ion()
plt.plot(hv,y,'o')
xplot = np.linspace(0,4,100)
plt.plot(xplot,func(xplot,*popt))









##############
#def func(x, a, b, c):
#    return a * np.exp(-b * x) + c
##############
#xdata = np.linspace(0, 4, 50)
##plt.plot(xdata)
#y = func(xdata, 2.5, 1.3, 0.5)
##plt.plot(y)
#np.random.seed(1729)
#y_noise = 0.2 * np.random.normal(size=xdata.size)
##plt.plot(y_noise)
#ydata = y + y_noise
##plt.plot(ydata)
#plt.plot(xdata, ydata, 'b-', label='data')
##############
#popt, pcov = curve_fit(func, xdata, ydata)
#popt
#print(popt)
#
#plt.plot(xdata, func(xdata, *popt), 'r-',
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
##############
#popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
#popt
#
##plt.plot(xdata, func(xdata, *popt), 'g--',
##        label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
##############
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend()
#plt.show()