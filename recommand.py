import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load assessment data
with open('data.json') as f:
    assessments = json.load(f)

# Prepare corpus
corpus = [a["description"] + " " + a["test_type"] for a in assessments]
corpus_embeddings = model.encode(corpus)

def recommend(query, top_k=10):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [assessments[i] for i in top_indices]
