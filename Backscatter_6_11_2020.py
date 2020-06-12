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
##########################################################################
##########https://www.youtube.com/watch?v=4vryPwLtjIY

###after discussing with Yang and Helen 6/2/2020 #simplified version of equation 2 from Yifan’s paper. A, B and C aren’t equivalent. Yielded similar results to the equation from Yifan’s paper but with fewer variables. A, B and C are simply for fitting
def func(x, a, b, c):
    return a * (1 - np.exp(-b * x * (c )))


######after discussing with Helen 5/29/2020
##def func(x, a, b, c, q):
##    return (a * x ** q * (1 - np.exp(-b * x )) + c)
#    
##############dummy func
##def func(x, a, b, c):
##    return (x) + (a + b) * (x * c)
#
#
path = r'C:\Users\tjmayer\Desktop\GEDI_Work\BackScatter'

##################################################################################################
########### x data  ###LIDAR
im = Image.open(path + r'\hmean_degree_setnull.tif')
print("im", im.size)
#im.show()
#Image.new("RGB",(300, 300), (50, 255, 80)).show()

#pic_x = imageio.imread(path + r'\Gamma_pw_clip_snap.tif')
#plt.figure(figsize = (5,5))
#plt.imshow(pic_x, cmap = plt.get_cmap(name = 'Blues'))


xdata = np.array(im)  

xout = func(xdata, A, B, C)

#xout = xout[:-19] #test first n numbers---limiter

print("xdata Nan?", np.isnan(xout).any())   #this prints False
print("xdata inf", np.isinf(xout).any())

print("xdata:", xout)
print("xdata size:", xout.size)
print("xdata shape:", xout.shape)

print("xdata:", xout)
print("xdata.ravel() size:", xout.ravel().size)
print("xdata.ravel() shape:", xout.ravel().shape)


x_ravel = xout.ravel()

#################################################################################################
########## y data  ###SAR
im_sar = Image.open(path + r'\Gamma_pw_clip_snap.tif')

#pic_y = imageio.imread(path + r'\hmean_degree_setnull.tif')
#plt.figure(figsize = (5,5))
#plt.imshow(pic_y, cmap = plt.get_cmap(name = 'Reds'))

y_sar = np.array(im_sar) 
print("y_sar", y_sar.shape)
print("y_sar", y_sar.size)
####################### Rolling window #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
####https://www.geeksforgeeks.org/numpy-cumsum-in-python/
####https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
#
#def movingaverage (values,window):
#    weights = np.repeat(1.0, window)/window
#    smas = np.convolve(values,weights,'valid')
#    return smas
#
#ydata = movingaverage(y_sar.ravel(),20)
##convolve_ydata = convolve_ydata[:100000] #test first n numbers---limiter
#print("convolve_ydata", ydata)
#print("convolve_ydatasize:", ydata.size)
#print("convolve_ydata shape:", ydata.shape)

##############
###https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html#scipy.ndimage.median_filter
#
#ydata = ndimage.median_filter(y_sar, size=20)
#print("convolve_ydata", ydata)
#print("convolve_ydatasize:", ydata.size)
#print("convolve_ydata shape:", ydata.shape)

###################
###https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
#
#def moving_average(a, n=3) :
#    ret = np.cumsum(a, dtype=float)
#    ret[n:] = ret[n:] - ret[:-n]
#    return ret[n - 1:] / n
#
#y_data = moving_average(y_sar, n=20)

#######################

print("ydata Nan?", np.isnan(ydata).any())   #this prints False
print("ydata inf", np.isinf(ydata).any())

print("ydata:", ydata)
print("ydata size:", ydata.size)
print("ydata shape:", ydata.shape)
y_ravel = ydata.ravel()

##################################################################################################
#ignore nan values in both inputs by creating a mask called "bad"  (potentially add inf into this)
bad = ~np.logical_or(np.isnan(x_ravel), np.isnan(y_ravel))
print("bad nan", bad.size)
# compress the ravelled x and y data with the 'bad' mask
#to create two new datasets xnew and ynew that should match in size and shape
xnew = np.compress(bad, xdata)
ynew = np.compress(bad, ydata)

#ynew = ynew[:100] #### older limiter
#xnew = xnew[:100] #### older limiter

print("y new size:", ynew.size)
print("y new shape:", ynew.shape)

print("x new size:", xnew.size)
print("x new shape:", xnew.shape)


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


#sns.distplot(x);
#sns.distplot(xnew);
#sns.distplot(y);
#jointhist = sns.jointplot(x, y); #kind="hex"; kind="kde"


#########################################################################
#g = [0.07964125, -0.42311371, -0.26063897]
#g = [1000000, 1000000, 1000000, 100000]

#
## this is the methodof fitting a curve without using the curve_fit Function
#length = len(x)
#print("length", length)
#empty = np.empty(length)
#for i in range (length):
#    empty[i] = func(x[i],g[0],g[1],g[2])
##plt.plot(x)
#plt.plot(x, empty,'ro')

############################################################################
###curvefit
#func = function name
#x = what is being used to predict the value
# y = measuered values that We will try and fit it to
# g= are intial guess
#c = covarince
#pcov = paramter values

#######

c, pcov = curve_fit(func, x, y) #maxfev=50000000
print("covarince values:", c)

length = len(x)
y_empty = np.empty(length)
for i in range (length):
    y_empty[i] = func(x[i],c[0],c[1],c[2])

#sns.lineplot(x);
#sns.lineplot(x, y_empty,'ro')
    
#plt.subplot (2,2,1)
plt.scatter(x, y) #, c=colors, alpha=0.5
#sns.scatterplot(x, y_empty)
plt.plot(x, y_empty,'ro', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(c))


#sns.lineplot(x,y)
#ax = sns.kdeplot(x, y_empty, cmap="Blues", shade=True, shade_lowest=False, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(c))
    

#plt.subplot (2,2,2)
#plt.plot(x, func(x, *c), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(c))


plt.xlabel('x LIDAR')
plt.ylabel('y SAR')
plt.legend()
plt.show()

R2 = r2_score(y_empty,y)
print("R^2", R2)


######################################
###linear fit

# single degree polynomial
#p1 = np.polyfit(x,y,1)
#print("polyfit", p1)
#plt.plot(x,y,"o")
#plt.plot(x,np.polyval (p1,x))
#
#plt.xlabel('x LIDAR')
#plt.ylabel('y SAR')
#plt.legend()
#plt.show()

# second degree polynomial

p2 = np.polyfit(x,y,2)
print("polyfit2", p2)
plt.plot(x,y,"o")
plt.plot(x,np.polyval (p2,x))


plt.xlabel('x LIDAR')
plt.ylabel('y SAR')
plt.legend()
plt.show()

slope, intercept,r_value,p_value,std_err = linregress(x,y)
print(pow(r_value,2))



########################
###initial plot
##plt.ion()
##plt.subplot(3,3,1)
##plt.hist(xnew, label= 'xnew')
##plt.subplot(3,3,2)
##plt.hist(ynew, label= 'ynew')
##plt.subplot(3,3,3)
##plt.hist(ydata, label= 'ydata')
#
#
###################################################################################################
##broke everything from here down!
#
#popt, pcov = curve_fit(func, xnew, ynew, maxfev=50000000)
#
##popt, pcov = curve_fit(func, convolve_xdata, y, maxfev=50000000)
##popt, pcov = curve_fit(func, y_func, convolve_xdata, maxfev=50000000)
##popt, pcov = curve_fit(func, y.ravel(), convolve_xdata, maxfev=50000000)
##popt, pcov = curve_fit(func, convolve_xdata, y.ravel(), maxfev=50000000)
##popt, pcov = curve_fit(func, xdata.ravel(), y.ravel(), maxfev=50000000)
#print("pcov:", pcov)
#print("pcov size:", pcov.size)
#print("pcov shape:", pcov.shape)
#
#print("popt:", popt)
#print("popt size:", popt.size)
#print("popt shape:", popt.shape)
#
#plt.ion()
##plt.plot(xdata, y,'o')
##plt.plot(convolve_xdata, y,'o', label="convolve_xdata v. y")
##plt.plot(convolve_xdata, y_func,'o', label="convolve_xdata v. y")
##plt.plot(y, convolve_xdata,'o', label="convolve_xdata v. y")
##ax = sns.kdeplot(convolve_xdata, y, cmap="Blues", shade=True, shade_lowest=False)
#ax = sns.kdeplot(xnew, ynew, cmap="Blues", shade=True, shade_lowest=False)
#
#plt.xlabel('convolve_xdata -- sar')
#plt.ylabel('y -- lidar')
#
#
#############
##curve_out = func(xdata,*popt)
##curve_out = func(convolve_xdata,*popt)
##curve_out = func(y,*popt)
##curve_out = func(y_func,*popt)
#curve_out = func(xnew,*popt)
#
##plt.plot(xdata,curve_out, 'r-',ls='--') #, label="curvefit"
##plt.plot(convolve_xdata, curve_out, 'r-',ls='--', label="curvefit") #, label="curvefit"
#
#plt.plot(xnew, curve_out, 'r-',ls='--', label="curvefit") #, label="curvefit"
##plt.plot(curve_out, 'r-',ls='--', label="curve_out") #, label="curvefit"
#plt.legend()
#
#
#######evaluation
#####https://stackoverflow.com/questions/41635448/how-can-i-draw-scatter-trend-line-on-matplot-python-pandas/41635626
#
#print("plot 2 started")
#fig = plt.figure(3, figsize=(5, 5))
#fig.clf()
#
#plt.plot(xnew,y,"+", ms=10, mec="k")   #label='input Data'
##z = np.polyfit(xdata.ravel(),y, 1)
#z = np.polyfit(xnew,ynew, 1)
#y_hat = np.poly1d(z)(xnew)
#
#plt.plot(xnew, y_hat, "r--", lw=1, label='r2' )
#text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
#plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
#     fontsize=14, verticalalignment='top')
#plt.legend()
#plt.xlabel('convolve_xdata -- sar')
#plt.ylabel('y -- lidar')
#
#
#
#######################################
###output-->save the image
##print("save y output")
#
##Image.save(fp, format=None, **params)
##y = y.save("inverted_Backscatter_ytestdata.tiff", format = tiff)
#print("end script")