import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from main import main_df as df

st.markdown('# EDA 📊')

fig = px.histogram(df, x='slope', color='num')
st.plotly_chart(fig)


fig = px.histogram(data_frame=df, x='age', color='dataset')
st.plotly_chart(fig)

fig = px.histogram(data_frame=df, x='thal', color='num')
st.plotly_chart(fig)


fig = px.histogram(data_frame=df, x='cp', color='num')
st.plotly_chart(fig)



fig, ax = plt.subplots()
corr = df.select_dtypes(exclude='object').corr()
sns.heatmap(corr, annot=True)
plt.title("Correlations between numeric features")
st.pyplot(fig)



fig, ax = plt.subplots()
sns.histplot(df['age'], kde=True)
plt.title("Correlations between numeric features")
st.pyplot(fig)


fig, ax = plt.subplots()
sns.histplot(df['chol'], kde=True)
plt.title("Correlations between numeric features")
st.pyplot(fig)



fig, ax = plt.subplots()
from sklearn.preprocessing import MinMaxScaler

# scale to make plot visible
scaler = MinMaxScaler()
cols = df.select_dtypes(exclude='object').columns

df_violin = pd.DataFrame(scaler.fit_transform(df.select_dtypes(exclude='object')), columns=cols)
sns.violinplot(df_violin)
st.pyplot(fig)




fig, ax = plt.subplots()
sex = df['sex'].value_counts()
plt.pie(x=sex, labels=sex.index, autopct='%.2f%%', colors=['cyan', 'pink'])
plt.title("Male vs Female")
plt.legend()
st.pyplot(fig)



fig, ax = plt.subplots()
sns.boxenplot(df_violin)
st.pyplot(fig)


import squarify
fig, ax = plt.subplots()
cities = df['dataset'].value_counts()
squarify.plot(sizes=cities, label=cities.index, pad=True)
plt.title("Compare places")
st.pyplot(fig)