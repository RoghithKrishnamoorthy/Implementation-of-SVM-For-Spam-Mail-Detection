# EX-09 Implementation of SVM For Spam Mail Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Roghith K
RegisterNumber: 212222240135
```
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
```
## Output:
## Head:
![image](https://github.com/ganeshshanmugavel27/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122046208/747c05d2-ffa0-4d69-9309-e562468d136d)

## Kernel Model:
![image](https://github.com/ganeshshanmugavel27/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122046208/21a53fe4-5c11-4cf4-ac18-5fdc150050b8)

## Accuracy and Classification report:
![image](https://github.com/ganeshshanmugavel27/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122046208/04e74b54-02a5-4dcf-b8b9-67dbc78c0fa1)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
