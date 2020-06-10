# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:26:59 2020

@author: tjmayer
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image  
from sklearn.metrics import r2_score
import seaborn as sns; sns.set(color_codes=True)

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


###after discussing with Yang and Helen 6/2/2020 #simplified version of equation 2 from Yifan’s paper. A, B and C aren’t equivalent. Yielded similar results to the equation from Yifan’s paper but with fewer variables. A, B and C are simply for fitting
def func(x, a, b, c):
    return a * (1 - np.exp(-b * x * (c )))
   
#####after discussing with Helen 5/29/2020
#def func(x, a, b, c, q):
#    return (a * x ** q * (1 - np.exp(-b * x )) + c)


##dummy equations
##def func(x, a, b, c, q):
##    return a * (1 - np.exp(-b * x * (c * q)))
##
#def dummy(x, q, a, b, c,):
#    return (q + x) + (a + b) * (x * c)


########### x data  ###SAR
im = Image.open('Gamma_pw-v3.tif')
xdata = np.array(im)  

#xdata = xdata[:10,:10] #test first n numbers---limiter

print("xdata Nan?", np.isnan(xdata).any())   #this prints False
print("xdata inf", np.isinf(xdata).any())

print("xdata:", xdata)
print("xdata size:", xdata.size)
print("xdata shape:", xdata.shape)

print("xdata:", xdata)
print("xdata.ravel() size:", xdata.ravel().size)
print("xdata.ravel() shape:", xdata.ravel().shape)
###################### Rolling window #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
###https://www.geeksforgeeks.org/numpy-cumsum-in-python/
###https://numpy.org/doc/stable/reference/generated/numpy.convolve.html

def movingaverage (values,window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values,weights,'valid')
    return smas

convolve_xdata = movingaverage(xdata.ravel(),20)
convolve_xdata = convolve_xdata[:100000] #test first n numbers---limiter
print("convolve_xdata", convolve_xdata)
print("convolve_xdatasize:", convolve_xdata.size)
print("convolve_xdata shape:", convolve_xdata.shape)
#plt.ion()
#plt.subplot(2,2,1)
#plt.plot(convolve_xdata,'o', label='convolve_xdata')
#print(f'The equation of regression line is y0={A}(1-e^(-{B}y^C)')
convolve_xdata = func(convolve_xdata, A, B, C)

########## y data  ###Lidar
im_lidar = Image.open('Savannakhet_hmean_degree.tif')

#im_lidar.show()
#Image.new("RGB",(300, 300), (50, 255, 80)).show()
y_out = np.array(im_lidar) 

#y_out = y_out[:10,:10]

##https://www.w3schools.com/python/numpy_array_filter.asp

filter_arr = y_out > 0

y = y_out[filter_arr]

print(filter_arr)
print("y", y.size)


#yout = func(xdata, A, B, C, Q)
y = np.nan_to_num(y)

y = y[:100000]

#y_func = func(y, A, B, C) # Q)

print("y Nan?", np.isnan(y).any())   #this prints False
print("y inf", np.isinf(y).any())

print("y:", y)
print("y size:", y.size)
print("y shape:", y.shape)

###initial plot
plt.ion()
#plt.subplot(2,2,2)
#plt.plot(y,'o', y_func, '*', label='y v. y_func')


######################
popt, pcov = curve_fit(func, convolve_xdata, y, maxfev=50000000)
#popt, pcov = curve_fit(func, y_func, convolve_xdata, maxfev=50000000)
#popt, pcov = curve_fit(func, y.ravel(), convolve_xdata, maxfev=50000000)
#popt, pcov = curve_fit(func, convolve_xdata, y.ravel(), maxfev=50000000)
#popt, pcov = curve_fit(func, xdata.ravel(), y.ravel(), maxfev=50000000)
print("pcov:", pcov)
print("pcov size:", pcov.size)
print("pcov shape:", pcov.shape)

print("popt:", popt)
print("popt size:", popt.size)
print("popt shape:", popt.shape)

plt.ion()
#plt.plot(xdata, y,'o')
#plt.plot(convolve_xdata, y,'o', label="convolve_xdata v. y")
#plt.plot(convolve_xdata, y_func,'o', label="convolve_xdata v. y")
#plt.plot(y, convolve_xdata,'o', label="convolve_xdata v. y")
ax = sns.kdeplot(convolve_xdata, y, cmap="Blues", shade=True, shade_lowest=False)
plt.xlabel('convolve_xdata -- sar')
plt.ylabel('y -- lidar')


############
#curve_out = func(xdata,*popt)
#curve_out = func(convolve_xdata,*popt)
#curve_out = func(y,*popt)
#curve_out = func(y_func,*popt)
curve_out = func(convolve_xdata,*popt)

#plt.plot(xdata,curve_out, 'r-',ls='--') #, label="curvefit"
#plt.plot(convolve_xdata, curve_out, 'r-',ls='--', label="curvefit") #, label="curvefit"

plt.plot(convolve_xdata, curve_out, 'r-',ls='--', label="curvefit") #, label="curvefit"
#plt.plot(curve_out, 'r-',ls='--', label="curve_out") #, label="curvefit"
plt.legend()


######evaluation
####https://stackoverflow.com/questions/41635448/how-can-i-draw-scatter-trend-line-on-matplot-python-pandas/41635626

print("plot 2 started")
fig = plt.figure(3, figsize=(5, 5))
fig.clf()

plt.plot(convolve_xdata,y,"+", ms=10, mec="k")   #label='input Data'
#z = np.polyfit(xdata.ravel(),y, 1)
z = np.polyfit(convolve_xdata,y, 1)
y_hat = np.poly1d(z)(convolve_xdata)

plt.plot(convolve_xdata, y_hat, "r--", lw=1, label='r2' )
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')
plt.legend()
plt.xlabel('convolve_xdata -- sar')
plt.ylabel('y -- lidar')



######################################
##output-->save the image
#print("save y output")

#Image.save(fp, format=None, **params)
#y = y.save("inverted_Backscatter_ytestdata.tiff", format = tiff)
print("end script")