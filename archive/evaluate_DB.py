import json
import numpy as np
from sklearn.metrics import silhouette_score
# from utils import train_cluster_model
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

emb_file = '/Users/Pengyun_Wang/Projects/HSBC/data/gte-embeddings.npy'
response_file = '/Users/Pengyun_Wang/Projects/HSBC/data/openai_responses.json'
with open(response_file, 'r') as f:
    response_data = json.load(f)

embeddings = np.load(emb_file)


def train_cluster_model_DBSCAN(embeddings):
    """
    Train a KMeans clustering model with the given number of clusters
    """
    clustering_model = DBSCAN(eps=10, min_samples=2)
    pipeline = Pipeline([
        ('scaling', StandardScaler()), 
        ('clustering', clustering_model)
    ])
    labels = pipeline.fit_predict(embeddings)
    return labels

# test the number of clusters
silhouette_scores = []
num_outliers = []
num_clusters = []

labels = train_cluster_model_DBSCAN(embeddings)
num_outliers.append(np.sum(labels == -1))
num_clusters.append(len(np.unique(labels)))
silhouette_scores.append(silhouette_score(embeddings[labels != -1], labels[labels != -1]))

print(f'Number of clusters: {num_clusters[-1]}, Number of outliers: {num_outliers[-1]}, Silhouette Score: {silhouette_scores[-1]}')


