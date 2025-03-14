# models/unsupervised_models.py

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

def perform_kmeans(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)
    return model, clusters

def perform_agglomerative(X, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(X)
    return model, clusters

def perform_dbscan(X, eps=2.0, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X)
    return model, clusters

def perform_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    return pca, components
