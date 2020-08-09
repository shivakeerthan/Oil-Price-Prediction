
import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import xlrd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

c=pd.read_excel(r'E:\Group Project\file.xlsx')
x1=c.iloc[:,:-1]

x2=c['CRUDEIND'].tail(20)
x3=c['CRUDEDUB'].tail(20)
x4=c['DIESELPG'].tail(20)

xtrain=np.array(x1.head(120))
xtest=np.array(x1.tail(20))
y1=c['class']
ytrain=np.array(y1.head(120))
ytest=np.array(y1.tail(20))

dec = SVR(kernel='linear', C=1e3)

y_rbf1 = dec.fit(xtrain,ytrain)
ypred2=y_rbf1.predict(xtest)
ypred1=y_rbf1.predict(xtrain)

c1a=0
c2a=0
accuracy1=0
accuracy2=0
print("Prediction using SVR kernel(linear)")
for i in range(0,120):
    if((round(ypred1[i]))==(round(ytrain[i]))):
        c1a=c1a+1
        accuracy1=(c1a/120)*100
print("Training accuracy:{}".format(accuracy1))


for i in range(0,20):
    if((round(ypred2[i]))==(round(ytest[i]))):
        c2a=c2a+1
        accuracy2=(c2a/20)*100
print("True accuracy:{}".format(accuracy2))

plt.scatter(x2,ytest,color="red")
plt.plot(x2,ypred2,color="blue")
plt.ylabel("Diesel price per litre(in INR)")
plt.xlabel("Crude oil price in India(INR)")
plt.title("Support Vector Regression(Linear Kernel)")
plt.show()

plt.scatter(x3,ytest,color="red")
plt.plot(x3,ypred2,color="blue")
plt.ylabel("Diesel price per litre(in INR)")
plt.xlabel("Crude oil price in DUBAI(INR)")
plt.title("Support Vector Regression(Linear Kernel)")
plt.show()

plt.scatter(x4,ytest,color="red")
plt.plot(x4,ypred2,color="blue")
plt.ylabel("Diesel price per litre(in INR)")
plt.xlabel("Diesel price per Gallon(INR)")
plt.title("Support Vector Regression(Linear Kernel)")

plt.show()





