# Titanic-Survival-Predictor
ML project which based on passenger data predict if he/she gonna survive or not!

### Demo
![Titanic-demo](media/demo.gif)  
Project used for Titanic ML competition on Kaggle: https://www.kaggle.com/competitions/titanic/overview  
New features used:
- RandomForestClassifier as a model
- df.map to reassign values
- df.fillna to fill empty cells
- selectbox and number_input as input
- st.balloons for effect

Model accuracy on Kaggle: 0.77511
  
### To open app locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
