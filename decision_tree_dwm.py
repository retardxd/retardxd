

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("data_class.csv")

x=data.iloc[:,:-1].values

y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier

classifer= DecisionTreeClassifier(criterion='entropy',random_state=0)

classifer.fit(x_train,y_train)

print(classifer.predict(sc.transform([[50,80700]])))

y_pred=classifer.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm=confusion_matrix(y_pred,y_test)

accuracy_score(y_test, y_pred)

print(cm)