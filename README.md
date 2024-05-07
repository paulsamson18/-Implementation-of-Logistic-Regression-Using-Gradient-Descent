# EX.NO:5 Implementation-of-Logistic-Regression-Using-Gradient-Descent
# DATE:
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Paul Samson.S
RegisterNumber: 212222230104

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
*/
```

## Output:
Array Value of x:


![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/eff918c5-2bcd-49e4-b978-48b389efc6a2)


## Array Value of y:

![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/13ac1c01-59e0-4241-bde9-55d935387586)

## Exam 1 - score graph:




![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/4290a194-5acd-43d0-a049-f9650aab2602)


## Sigmoid function graph:


![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/e3084e4f-dccb-4efc-b34a-3cfe2350367e)


## X_train_grad value:


![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/811052aa-cd7d-4fa8-96ae-304e11575ea3)


## Y_train_grad value:


![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/51f87878-a179-4987-9698-da15e145289a)


## Print res.x:


![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/2f8c6e05-3452-4dba-8fd0-759a324c038c)


## Decision boundary - graph for exam score:


![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/e5385c03-3aa3-4fab-bb32-d7f22bc54850)


## Proability value:

![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/d31f0dd0-bbf9-49f3-b401-aef7c7642448)


## Prediction value of mean:

![image](https://github.com/haritha-venkat/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121285701/127f330c-4ad6-4022-924e-6f23e062918b)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
