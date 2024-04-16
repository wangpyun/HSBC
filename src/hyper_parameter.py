import json
import numpy as np
from sklearn.metrics import silhouette_score
# from utils import train_cluster_model
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def hyper_parameter_search():
    emb_file = '../data/gte-embeddings.npy'
    response_file = '../data/openai_responses.json'

    with open(response_file, 'r') as f:
        response_data = json.load(f)
    embeddings = np.load(emb_file)

    def train_cluster_model(n_clusters, embeddings):
        """
        Train a KMeans clustering model with the given number of clusters
        """
        kmeans = KMeans(n_clusters=n_clusters)
        pipeline = Pipeline([
            ('scaling', StandardScaler()), 
            ('clustering', kmeans)
        ])
        pipeline.fit(embeddings)
        return pipeline


    # test the number of clusters
    silhouette_scores = []
    n_start = 2
    n_end = 20

    for n_clusters in range(n_start, n_end):
        print(f'Evaluating number of clusters: {n_clusters}')
        pipeline = train_cluster_model(n_clusters, embeddings)
        labels = pipeline.predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores
    plt.plot(range(n_start, n_end), silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.savefig('../exp/silhouette_scores.png')

if __name__ == '__main__':
    hyper_parameter_search()


