import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('diabetes.csv')

zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,np.nan)
    mean= int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.nan,mean)

X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)

sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=25,p=2,metric='euclidean')

classifier.fit(X_train,y_train)

#KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='euclidean',met  ric_params=None,n_jobs=1,n_neighbors=25,p=2,weights='uniform')

y_pred= classifier.predict(X_test)
print(y_pred)

cm=confusion_matrix(y_test,y_pred)
print(cm)

print(f1_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred))