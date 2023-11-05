# Implementation-of-SVM-For-Spam-Mail-Detection

## Aim:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Roghith K
RegisterNumber: 212222040135
*/
```
```
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
```
```
import pandas as pd
data= pd.read_csv("/content/spam.csv",encoding='Windows-1252')
```
```
data.head()
```
```
data.info()
```
```
x=data["v1"].values
```
```
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Result output
![279719379-aaa8e951-a92a-40f4-8e7e-a8d9dddaa9c1](https://github.com/RoghithKrishnamoorthy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475474/87d89370-0471-4d5a-9da7-21d6c54b3f65)

### data.head()
![279719568-30bda476-48ec-4f39-96fb-347e938e0365](https://github.com/RoghithKrishnamoorthy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475474/8235037c-e206-418e-aeb7-97bac866052d)

### data.info()
![279719719-1fa10c39-8276-4d93-8452-6cb0049d862b](https://github.com/RoghithKrishnamoorthy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475474/c926e7a7-6a71-4f96-82f6-83bc5f70de58)

### data.isnull().sum()
![279719840-a619d3e5-5c55-4334-981d-81df1a51d3fc](https://github.com/RoghithKrishnamoorthy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475474/51d65d5c-d919-43a7-ac9a-0c8bf9c15bbd)

### Y_prediction value
![279719938-9a4393cf-f679-4d9e-b798-bb3900a4440c](https://github.com/RoghithKrishnamoorthy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475474/d96cb11e-3eb1-488e-8763-28db509dbb2c)

### Accuracy value
![279720070-7df9b110-0068-449a-9c9f-28e24816adaf](https://github.com/RoghithKrishnamoorthy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475474/45ed95d8-3a05-474d-8700-02211d879488)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
