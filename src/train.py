import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import train_cluster_model

def train():
    emb_file = '../data/gte-embeddings.npy'
    response_file = '../data/overall_summary.json'

    with open(response_file, 'r') as f:
        response_data = json.load(f)
    embeddings = np.load(emb_file)

    def train_cluster_model(n_clusters, embeddings):
        kmeans = KMeans(n_clusters=n_clusters)
        pipeline = Pipeline([
            ('scaling', StandardScaler()), 
            ('clustering', kmeans)
        ])
        pipeline.fit(embeddings)
        return pipeline

    # train
    optimal_n_clusters = 3
    pipeline = train_cluster_model(optimal_n_clusters, embeddings)
    labels = pipeline.predict(embeddings)

    # cluster the URLs
    from collections import defaultdict
    urls = response_data.keys()
    clusters = defaultdict(list)
    for i, r in enumerate(urls):
        clusters[str(labels[i])].append(r)

    # write clusters to file
    cluster_file = '../model/clusters.json'
    with open(cluster_file, 'w') as f:
        json.dump(clusters, f, indent=4)

    # write model to file
    model_file = '../model/model.pkl'
    import joblib
    joblib.dump(pipeline, model_file)

if __name__ == '__main__':
    train()