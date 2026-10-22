import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, clustering

st.title('Clustering App')

file = st.file_uploader('Upload your dataset', type=['csv'])

if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

features = st.multiselect('Select the features for clustering', df.columns.to_list())

if len(features) < 2:
    st.warning('Select at least 2 features')
    st.stop()

st.subheader('Model Selection')
algorithm = st.selectbox('Select a clustering algorithm',
                         ['DBSCAN', 'Hierarchical Clustering'])

st.subheader('Hyperparameter Tuning')
if algorithm == 'DBSCAN':
    eps = st.slider('eps', 0.1, 5.0, 0.5, step=0.1)
    min_samples = st.slider('min_samples', 1, 20, 5)
    params = {'eps': eps, 'min_samples': min_samples}
else:
    n_clusters = st.slider('n_clusters', 2, 10, 3)
    linkage = st.selectbox('linkage', ['ward', 'single', 'complete', 'average'])
    params = {'n_clusters': n_clusters, 'linkage': linkage}

X = preprocessing(df, features)
labels = clustering(X, algorithm, **params)

result = df.copy()
result['cluster'] = labels

st.subheader('Cluster Labels')
st.write(pd.Series(labels, name='cluster').value_counts().sort_index()
         .rename_axis('cluster').reset_index(name='count'))

if algorithm == 'DBSCAN':
    n_noise = int((labels == -1).sum())
    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    st.info(f'Clusters found: {n_found} | Noise points (label -1): {n_noise}')

st.subheader('Clustered Dataset')
st.dataframe(result)

if len(features) == 2:
    fig, ax = plt.subplots()
    ax.scatter(df[features[0]], df[features[1]], c=labels, cmap='viridis')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(algorithm)
    st.pyplot(fig)
else:
    st.info('Select exactly 2 features to see the 2D scatter plot.')