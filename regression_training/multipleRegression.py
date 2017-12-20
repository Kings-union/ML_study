from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

datapath = r"Dilivery.csv"
deliveryData = genfromtxt(datapath, delimiter=',')

# print(deliveryData)

X = deliveryData[:, :-1]
Y = deliveryData[:, -1]
# print("X:",X,"Y",Y)
print(deliveryData)
print(X)

regr = linear_model.LinearRegression()
regr.fit(X,Y)

print(regr.coef_, regr.intercept_)

xPred = [[106, 6, 1]]
yPred = regr.predict(xPred)
print(yPred)