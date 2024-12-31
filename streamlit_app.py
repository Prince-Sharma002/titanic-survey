import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

# Load and preprocess the Titanic dataset
@st.cache
def load_data():
    df = pd.read_csv('titanic.csv')
    df = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
    df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill missing Age values
    le_sex = LabelEncoder()
    df['Sex_n'] = le_sex.fit_transform(df['Sex'])
    input_features = df.drop(['Survived', 'Sex'], axis='columns')
    target = df['Survived']
    return input_features, target

@st.cache
def train_model(input_features, target):
    X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2)
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Load data and train the model
input_features, target = load_data()
model = train_model(input_features, target)

# Streamlit app interface
st.title("Titanic Survival Prediction")

st.write("""
This app predicts whether a Titanic passenger would survive based on their features.
Provide the input details below and click 'Predict'.
""")

# User input fields
st.header("Passenger Details")

pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third):", [1, 2, 3], index=2)
age = st.number_input("Age:", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
fare = st.number_input("Fare:", min_value=0.0, max_value=500.0, value=32.0, step=0.5)
sex = st.selectbox("Sex:", ["Male", "Female"])

# Encode the sex value
sex_encoded = 1 if sex == "Male" else 0

# Predict button
if st.button("Predict"):
    # Prepare the input array
    input_data = np.array([[pclass, age, fare, sex_encoded]])
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.success("The passenger is predicted to have survived. 🎉")
    else:
        st.error("The passenger is predicted to have not survived. 😢")
