import streamlit as st
import pandas as pd
import numpy as np

model = joblib.load('titanic.joblib')

#Setup website
st.title("Titanic Survival Predictor")
st.markdown("Setup values for your passenger and find out if he will survive!")
st.caption("Accuracy: 0.775 on Kaggle")

##Input
pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 30)
sibsp = st.sidebar.number_input("Siblings/Spouses aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents/Children aboard", 0, 6, 0)
fare = st.sidebar.number_input("Fare paid", 0.0, 500.0, 30.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])


if st.button("Predict Survival"):
    input_df = pd.DataFrame({
        "PassengerId": [1], "Pclass": [pclass], "Sex": [sex], "Age": [age],
        "SibSp": [sibsp], "Parch": [parch], "Fare": [fare],
        "Embarked": [embarked]
    })
    input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})
    input_df['Embarked'] = input_df['Embarked'].map({'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2})
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Survival probability: **{probability:.1%}**")
    if probability > 0.5:

        st.balloons()

