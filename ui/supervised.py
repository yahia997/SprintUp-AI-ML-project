import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from main import feature_selected as df
import plotly.express as px

# load best trained models
log = joblib.load('../models/log_best.pkl')
dst = joblib.load('../models/dst_best.pkl')
rf = joblib.load('../models/rf_best.pkl')
svm = joblib.load('../models/svm_best.pkl')

st.markdown('# Supervised Learning 🤖')
st.write('''
Hyperparameter Tuning:
Used **GridSearch** and **RandomizedSearch** to get the best paramters for models.

- Logistic Regression: **82%** Accuracy
         
- Decision Tree: **80%** Accuracy
         
- Random Forest: **84%** Accuracy
         
- SVM: **84%** Accuracy
''')
st.markdown('## Best Paramters of each model')
st.markdown('### Logistice Regression')
st.write(log.get_params())
st.markdown('### Decision Tree')
st.write(dst.get_params())
st.markdown('### Random Forest')
st.write(rf.get_params())
st.markdown('### SVM')
st.write(svm.get_params())

# Create an input form for user data
st.title("Predict with Your Model")
st.write("you can change paramters in the left side of the page")

features = []
X = df.iloc[:, :-1] # features only
y = df.iloc[:, -1] # target column

for col in X.columns:
  features.append(st.sidebar.slider(f"{col}", min_value=X[col].min(), max_value=X[col].max(), value=X[col].max()))

# Predict when user inputs data
user_input = np.array([features])

# predict output
prediction_log = log.predict_proba(user_input)[0]
prediction_dst = dst.predict_proba(user_input)[0]
prediction_rf = rf.predict_proba(user_input)[0]
prediction_svm = svm.predict(user_input)

# Display the result
df1 = pd.DataFrame({
  'Non Disease': [prediction_log[0], prediction_dst[0], prediction_rf[0], 1 if prediction_svm[0] == '0.0' else 0],
  'Disease': [prediction_log[1], prediction_dst[1], prediction_rf[1], 1 if prediction_svm[0] == '1.0' else 0],
}, index=['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
)

# convert wide format to long format
df_long = df1.reset_index().melt(id_vars='index', var_name='Class', value_name='Probability')
df_long.rename(columns={'index': 'Model'}, inplace=True)

# plot
fig = px.bar(df_long, x='Model', y='Probability', color='Class', barmode='group',
             title='Model Prediction Probabilities')
st.plotly_chart(fig)