# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:47:07 2020

@author: tjmayer
"""
#############https://stats.stackexchange.com/questions/407562/goodness-of-fit-measurement-in-python
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

###backscatter
def func(x, a, b, c):
    return a * (1 - np.exp(-b * x**(c)))

A = 2.5
B = 1.3
C = 0.5


#Your data
np.random.seed(0)
N = 1000
x = np.linspace(0.1,1.1,101)
y = func(x, A, B, C)


fig = plt.figure(2, figsize=(5, 5))
fig.clf()

plt.plot(x,y,"+", ms=10, mec="k")
z = np.polyfit(x,y, 1)
y_hat = np.poly1d(z)(x)

plt.plot(x, y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')


#Your model
model=LinearRegression()
model.fit(x.reshape(-1,1),y)

#Your statsitic
r2 = r2_score(y, model.predict(x.reshape(-1,1)))
print("r2", r2)


################
#popt, pcov = curve_fit(func, x, y, p0=[2,1,1])
#
#
#plt.ion()
#plt.plot(x,y,'o')
#xplot = np.linspace(0,4,100)
#
#plt.plot(xplot,func(xplot,*popt))
