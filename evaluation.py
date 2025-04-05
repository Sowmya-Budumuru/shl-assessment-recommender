import numpy as np

def mean_recall_at_k(recs, relevant, k=3):
    """
    Computes Mean Recall@K for a list of recommendations and relevant assessments.
    """
    recalls = []
    
    for rec, rel in zip(recs, relevant):
        top_k_recs = rec[:k]  # Get the top k recommended assessments
        relevant_assessments = set(rel)  # Convert relevant assessments to a set
        retrieved_relevant = sum(1 for rec in top_k_recs if rec in relevant_assessments)  # Count relevant items in top k
        recall = retrieved_relevant / len(relevant_assessments)  # Recall is the ratio of relevant retrieved items
        recalls.append(recall)
    
    return np.mean(recalls)  # Return the mean recall over all queries

def mean_average_precision_at_k(recs, relevant, k=3):
    """
    Computes Mean Average Precision@K for a list of recommendations and relevant assessments.
    """
    aps = []
    
    for rec, rel in zip(recs, relevant):
        relevant_set = set(rel)  # Set of relevant assessments
        relevant_count = len(relevant_set)
        
        precision_at_k = 0
        for i in range(min(k, len(rec))):
            if rec[i] in relevant_set:  # If the recommendation is relevant
                precision_at_k += (i + 1) / (i + 1)  # Precision is 1/(rank) for relevant items
        
        ap_at_k = precision_at_k / relevant_count if relevant_count else 0  # Average Precision
        aps.append(ap_at_k)
    
    return np.mean(aps)  # Return the mean average precision over all queries
