import streamlit as st
import pandas as pd

main_df = pd.read_csv('../data/heart_disease.csv', index_col='id')
processed_df = pd.read_csv('../data/cleaned.csv', index_col='id')
feature_selected = pd.read_csv('../data/feature_selected.csv', index_col='Unnamed: 0')
reduced_dimension = pd.read_csv('../data/pca.csv', index_col='Unnamed: 0')

st.markdown('# Dataset 🗂️')
st.write('''
Source: (Heart Disease Dataset)[https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data]

Description: The dataset includes various clinical features such as age, cholesterol levels, blood pressure, and more, aimed at determining the presence of heart disease.
''')


option = st.selectbox('Select Dataset', [
  'Unprocessed Dataset',
  'Processed Dataset',
  'Dataset After Feature Selection',
  'Dataset After Dimensionality Reduction (PCA)'
])

if option == 'Unprocessed Dataset':
  st.write(main_df)
elif option == 'Processed Dataset':
  st.write(processed_df)
elif option == 'Dataset After Feature Selection':
  st.write(feature_selected)
elif option == 'Dataset After Dimensionality Reduction (PCA)':
  st.write(reduced_dimension)