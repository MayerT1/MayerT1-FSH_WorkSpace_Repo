# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:26:59 2020

@author: tjmayer
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image  
from sklearn.metrics import r2_score

#A,B,C are asuumed constant for a generic type of forest over a wide area
#Lei, Y., Siqueira, P., Torbick, N., Ducey, M., Chowdhury, D., & Salas, W. (2018). Generation of Large-Scale Moderate-Resolution Forest Height Mosaic With Spaceborne Repeat-Pass SAR Interferometry and Lidar. IEEE Transactions on Geoscience and Remote Sensing, 57(2), 770-787.
#Asian Tropical Moist
#A = 0.045409
#B = 0.060518
#C = 0.00
#alpha = 0.060518

##Tropical Dry Broadleaf
A = 0.021563
B = 0.042324
C = 0.027519
Q = 0.1117  #q=alpha

#def func(x, a, b, c, q):
#    return a + q * np.exp(-b * x) + c

####after discussing with Helen 5/29/2020
def func(x, a, b, c, q):
    return (a * x ** q * (1 - np.exp(-b * x )) + c)

##dummy equations
##def func(x, a, b, c, q):
##    return a * (1 - np.exp(-b * x * (c * q)))
##
#def dummy(x, q, a, b, c,):
#    return (q + x) + (a + b) * (x * c)


########### x data
im = Image.open('Gamma_pw-v3.tif')
xdata = np.array(im)  

#xdata = xdata[:1000,:1000] #test first n numbers---limiter

#xdata = np.linspace(0.1,5.1,101)  ####### dummy data
#xdata = np.linspace(0, 4, 50)   ####### dummy data

print("xdata Nan?", np.isnan(xdata).any())   #this prints False
print("xdata inf", np.isinf(xdata).any())

print("xdata:", xdata)
print("xdata size:", xdata.size)
print("xdata shape:", xdata.shape)

#print("xdata:", xdata)
print("xdata.ravel() size:", xdata.ravel().size)
print("xdata.ravel() shape:", xdata.ravel().shape)

########## y data
yout = func(xdata, A, B, C, Q)
y = np.nan_to_num(yout)

#dummy y data
#y_noise = 0.2 * np.random.normal(size=xdata.size)
#ydata = np.linspace(0, -27, 50)
#y = ydata + y_noise

print("y Nan?", np.isnan(y).any())   #this prints False
print("y inf", np.isinf(y).any())

print("y:", y)
print("y size:", y.size)
print("y shape:", y.shape)

#############curve fit
popt, pcov = curve_fit(func, xdata.ravel(), y.ravel(), maxfev=50000000)
print("pcov:", pcov)
print("pcov size:", pcov.size)
print("pcov shape:", pcov.shape)

print("popt:", popt)
print("popt size:", popt.size)
print("popt shape:", popt.shape)

plt.ion()
plt.plot(xdata, y,'o')

curve_out = func(xdata,*popt)
plt.plot(xdata,curve_out, 'r-',ls='--') #, label="curvefit"
plt.legend()


######evaluation
####https://stackoverflow.com/questions/41635448/how-can-i-draw-scatter-trend-line-on-matplot-python-pandas/41635626
print("plot 2 started")
fig = plt.figure(3, figsize=(5, 5))
fig.clf()

plt.plot(xdata,y,"+", ms=10, mec="k")   #label='input Data'
#z = np.polyfit(xdata.ravel(),y, 1)
z = np.polyfit(xdata.ravel(),y.ravel(), 1)
y_hat = np.poly1d(z)(xdata)

plt.plot(xdata, y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')
plt.legend()

######################################
##output-->save the image
print("save y output")

#Image.save(fp, format=None, **params)
#y = y.save("inverted_Backscatter_ytestdata.tiff", format = tiff)
print("end script")