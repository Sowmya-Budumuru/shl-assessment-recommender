import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from evaluation import mean_recall_at_k, mean_average_precision_at_k

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load assessment data
with open('data.json') as f:
    assessments = json.load(f)

# Prepare corpus (combining description and test_type as the corpus)
corpus = [a["description"] + " " + a["test_type"] for a in assessments]
corpus_embeddings = model.encode(corpus)

def recommend(query, top_k=10):
    """
    Recommends top_k relevant assessments for the given query.
    """
    query_embedding = model.encode([query])  # Get the embedding of the query
    scores = cosine_similarity(query_embedding, corpus_embeddings)[0]  # Calculate cosine similarity

    top_indices = np.argsort(scores)[::-1][:top_k]  # Get indices of top K recommendations
    return [assessments[i] for i in top_indices]  # Return the top K assessments as a list of dictionaries

def evaluate_recommendations(query, relevant_assessments, k=3):
    """
    Evaluates the recommendations against the relevant assessments for the given query.
    """
    recommended_assessments = recommend(query, top_k=k)
    
    # Get the names of the recommended and relevant assessments
    recommended_names = [rec["name"] for rec in recommended_assessments]
    relevant_names = [rel["name"] for rel in relevant_assessments]
    
    recall = mean_recall_at_k([recommended_names], [relevant_names], k)
    map_score = mean_average_precision_at_k([recommended_names], [relevant_names], k)
    
    return recall, map_score

# Example: Test the evaluation function
query = "software engineer"
relevant_assessments = [{"name": "Cognitive Test 1"}, {"name": "Technical Test 1"}]  # Example relevant assessments

# Evaluate the recommendations for this query
recall, map_score = evaluate_recommendations(query, relevant_assessments, k=3)

print(f"Mean Recall@3: {recall}")
print(f"Mean Average Precision@3: {map_score}")
