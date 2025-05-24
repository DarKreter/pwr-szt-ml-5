import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, rand_score

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def calculate_clustering_metrics(X, y_true, y_pred):
    metrics = {}
    
    metrics['silhouette_true'] = silhouette_score(X, y_true)
    metrics['silhouette_pred'] = silhouette_score(X, y_pred)
    metrics['davies_bouldin_true'] = davies_bouldin_score(X, y_true)
    metrics['davies_bouldin_pred'] = davies_bouldin_score(X, y_pred)
    metrics['rand_index'] = rand_score(y_true, y_pred)
    metrics['purity'] = purity_score(y_true, y_pred)
    
    return metrics



def print_clustering_metrics(metrics):
    print(f"Silhouette Score (from classes): {metrics['silhouette_true']:.4f}")
    print(f"Silhouette Score (from clusters): {metrics['silhouette_pred']:.4f}")
    print(f"Davies-Bouldin Index (from classes): {metrics['davies_bouldin_true']:.4f}")
    print(f"Davies-Bouldin Index (from clusters): {metrics['davies_bouldin_pred']:.4f}")
    print(f"Rand Index: {metrics['rand_index']:.4f}")
    print(f"Purity score: {metrics['purity']}")