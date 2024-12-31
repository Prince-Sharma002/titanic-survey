import pandas as pd
import numpy as np
df = pd.read_csv('titanic.csv')
df = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'] , axis='columns')
input = df.drop(['Survived'] , axis='columns')
target = df['Survived']
from sklearn.preprocessing import LabelEncoder
leSex = LabelEncoder()
input['Sex_n'] = leSex.fit_transform(input['Sex'])
input = input.drop(['Sex'] , axis='columns')
input.Age = input.Age.fillna( input.Age.mean() )
from sklearn import tree
model = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input,target,test_size=0.2)
model.fit(X_train,y_train)
model.predict( [ [1,22.0,70.2500,0 ]] )