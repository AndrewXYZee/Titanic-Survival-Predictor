#Load required libraries and data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

#Remove not needed columns
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#Convert text to digits
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

#Fill out empty cells
m_fare = df['Fare'].median() #Calculate median
df['Fare'] = df['Fare'].fillna(m_fare) #Fill out empty 'Fare' cells
m_age = df['Age'].median()
df['Age'] = df['Age'].fillna(m_age)

#Load ready data
X = df.drop('Survived', axis=1)
y = df['Survived']

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

#Full data model
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X, y)

#Load test data
df_test = pd.read_csv("test.csv")

#Repeat same steps to prepare data
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Fare'] = df_test['Fare'].fillna(m_fare)
df_test['Age'] = df_test['Age'].fillna(m_age)

#Use model to predict values
predict = final_model.predict(df_test[X.columns])  

#Submit file
submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": predict
})
submission.to_csv("submission.csv", index=False)

joblib.dump(final_model, 'train_model.joblib')
