import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    
def plot_clustering(X, y_true, y_pred, cluster_centers):
    n_true_classes = len(np.unique(y_true))
    n_clusters = max(np.unique(y_pred)) + 1
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(n_true_classes):
        plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1],
                    label=f'True Class {i}')
    plt.title('True Classes (PCA)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(n_clusters):
        plt.scatter(X_pca[y_pred == i, 0], X_pca[y_pred == i, 1],
                    label=f'Cluster {i}')

    if cluster_centers is not None:
        centers_pca = pca.transform(cluster_centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                    c='black', marker='x', s=200, linewidths=3, label='Centroids')
    else:
        plt.scatter(X_pca[y_pred == -1, 0], X_pca[y_pred == -1, 1],
                    label=f'Outlier & noise')
    plt.title('Clusters (PCA)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()