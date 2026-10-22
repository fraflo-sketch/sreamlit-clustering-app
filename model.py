import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


def preprocessing(df, features):
    X = df[features].copy()

    numerical = X.select_dtypes(include=['int64', 'float64']).columns
    categorical = X.select_dtypes(include=['object', 'category']).columns

    if len(numerical) > 0:
        X[numerical] = StandardScaler().fit_transform(X[numerical])

    for col in categorical:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X


def hierarchical_clustering(X, n_clusters, linkage):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(X)


def dbscan_clustering(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)


def clustering(X, algorithm, **params):
    if algorithm == 'DBSCAN':
        return dbscan_clustering(X, params['eps'], params['min_samples'])
    return hierarchical_clustering(X, params['n_clusters'], params['linkage'])


def evaluate(X, labels):
    mask = labels != -1
    X_eval = X[mask]
    labels_eval = labels[mask]

    n_clusters = len(set(labels_eval))
    if n_clusters < 2 or n_clusters >= len(labels_eval):
        return None, None

    return (silhouette_score(X_eval, labels_eval),
            davies_bouldin_score(X_eval, labels_eval))