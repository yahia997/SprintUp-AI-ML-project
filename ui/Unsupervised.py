import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import joblib
from main import reduced_dimension as df
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# import models
km = joblib.load('../models/kmeans.pkl')
hc = joblib.load('../models/hc.pkl')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

st.markdown('# Best Paramters of each model')
st.markdown('### Kmeans')
st.write(km.get_params())

fig = plt.figure(dpi=400, figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.iloc[:,0], X.iloc[:,1],X.iloc[:,2], c=km.labels_, s=8)
st.pyplot(fig)

st.markdown('### Hierarchical Clustering')
fig, ax = plt.subplots(figsize=(8, 4))
dendrogram(hc, truncate_mode='level', p=5)
st.pyplot(fig)