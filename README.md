# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Farhana H
RegisterNumber:  212223230057


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse) 
*/
```

## Output:
## Contents in the data file (head, tail):
![358854172-3ad5ed58-0065-4e57-9a00-024277758f05](https://github.com/user-attachments/assets/4b5d0b04-73b7-405d-8802-98874e368f01)
## X and Y datasets from original dataset:
![358854194-000a0ee0-b2aa-4398-a56b-75c2095bc241](https://github.com/user-attachments/assets/36491569-71fc-436e-82eb-6769799fbf48)
## Predicted and Test values of Y:
![358854208-d2a852a2-8402-49fd-ac25-935769facc36](https://github.com/user-attachments/assets/5e2191dd-800d-4674-98f6-c1ac553f6bc7)
## Graph for Training Data:
![358854230-f13752f3-ad2b-4ccd-9d2c-6d4e342aae48](https://github.com/user-attachments/assets/f9a59243-d7f3-43ec-98b1-71e45947a928)
## Graph for Test Data:
![358854264-95215c63-fb67-4242-b281-02188acde5e9](https://github.com/user-attachments/assets/7f870661-7dca-4098-a646-4eefa380b196)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
