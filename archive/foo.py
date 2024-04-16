from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

model_path = "/Users/Pengyun_Wang/Projects/HSBC/model/model.pkl"
embeddings_file = "/Users/Pengyun_Wang/Projects/HSBC/data/gte-embeddings.npy"
with open(embeddings_file, "rb") as f:
    embeddings = np.load(f)

model = joblib.load(model_path)
_dist = model.transform(embeddings)
res = cosine_similarity(_dist)

print(f"min similarity: {res.min()}")
