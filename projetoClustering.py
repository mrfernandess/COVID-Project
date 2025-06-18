import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as shc

# Find the optimal number of clusters (K-Means)
def find_optimal_k(data):
    distortions = []
    silhouette_scores = []
    K = range(2, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Plot the elbow graph (Distortion) and Silhouette Scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores by k')

    plt.tight_layout()
    plt.show()

    return K[np.argmax(silhouette_scores)]

# Apply K-Means clustering
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

# Apply DBSCAN with parameter search
def find_best_dbscan_params(data, eps_range, min_samples_range):
    best_eps = None
    best_min_samples = None
    best_silhouette = -1

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            # Ignore results with all points in one cluster or as noise
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_silhouette:
                    best_silhouette = score
                    best_eps = eps
                    best_min_samples = min_samples

    return best_eps, best_min_samples

def apply_agglomerative(data):
    linkage_methods = ['ward', 'complete', 'average', 'single']
    cluster_range = range(2, 11)
    best_linkage = None
    best_clusters = None
    best_silhouette = -1
    best_labels = None

    for linkage in linkage_methods:
        for n_clusters in cluster_range:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = model.fit_predict(data)
            if len(set(labels)) > 1:
                silhouette = silhouette_score(data, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_linkage = linkage
                    best_clusters = n_clusters
                    best_labels = labels

    return best_linkage, best_clusters

# Function for Subtractive Clustering
def subtractive_clustering(df, alpha=0.5, beta=0.8, radius=1.0, min_potential=1e-5):
    df = df.to_numpy()
    n_points = df.shape[0]
    potentials = np.zeros(n_points)

    # Calculating the initial potentials for each point
    for i in range(n_points):
        distances = np.linalg.norm(df[i] - df, axis=1)
        potentials[i] = np.sum(np.exp(-alpha * (distances**2)))

    centers = []
    while True:
        # Find the point with the highest potential
        max_potential_idx = np.argmax(potentials)
        max_potential = potentials[max_potential_idx]

        if max_potential < min_potential:
            break

        # Store the cluster center
        centers.append(df[max_potential_idx])

        # Reduce the potentials of nearby points
        distances = np.linalg.norm(df - df[max_potential_idx], axis=1)
        potentials -= max_potential * np.exp(-beta * (distances**2))

    return np.array(centers)

def assign_clusters(df, centers):
    df = df.to_numpy()
    distances = np.linalg.norm(df[:, None] - centers, axis=2)
    return np.argmin(distances, axis=1)

def find_best_subtractive_params(df, alpha_range, beta_range, radius_range, min_potential_range):
    best_alpha = None
    best_beta = None
    best_radius = None
    best_min_potential = None
    best_silhouette = -1

    for alpha in alpha_range:
        for beta in beta_range:
            for radius in radius_range:
                for min_potential in min_potential_range:
                    centers = subtractive_clustering(df, alpha, beta, radius, min_potential)
                    clusters = assign_clusters(df, centers)
                    score = silhouette_score(df, clusters)
                    if score > best_silhouette:
                        best_silhouette = score
                        best_alpha = alpha
                        best_beta = beta
                        best_radius = radius
                        best_min_potential = min_potential

    return best_alpha, best_beta, best_radius, best_min_potential

# Correlation with domain rules and evaluation with Target
def evaluate_clusters(data, labels, original_df, target_column):
    
    if len(original_df) != len(labels):
        raise ValueError("The size of the original data does not match the number of generated clusters.")

    # Add labels to the original dataset
    original_df['Cluster'] = labels

    # Correlation with important variables (domain rules)
    print("Correlation with domain variables:")
    cluster_means = original_df.groupby('Cluster').mean()
    for col in original_df.columns:
        if col != target_column:
            correlation = original_df.groupby('Cluster')[col].mean()
            print(f"Mean of {col} by cluster:\n{correlation}\n")

    # Comparison with TARGET
    if target_column in original_df.columns:
        cross_tab = pd.crosstab(original_df['Cluster'], original_df[target_column])
        print("\nComparison between clusters and TARGET:\n")
        print(cross_tab)
        
    # Plot of means by cluster
    cluster_means_transposed = cluster_means.drop(columns=[target_column]).T
    cluster_means_transposed.plot(kind="bar", figsize=(10, 6), width=0.8)
    plt.title("Means of Variables by Cluster")
    plt.ylabel("Mean")
    plt.xlabel("Variables")
    plt.xticks(rotation=45)
    plt.legend(title="Clusters", loc="best")
    plt.tight_layout()
    plt.show()

def plot_clusters(df, labels, title):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Cluster')
    plt.show()

def plot_dendrogram(X, method):
    plt.figure(figsize=(10, 7))
    dend = shc.dendrogram(shc.linkage(X, method))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

# Function to plot data distribution
def plot_data_distribution(df, title):
    sns.pairplot(df)
    plt.title(title)
    plt.show()

def SSE(df, centers, clusters):
    distances = np.linalg.norm(df.to_numpy() - centers[clusters], axis=1)
    return np.sum(distances ** 2)

def clustering(X_selected, selected_features_with_target):
    
    plot_data_distribution(X_selected, 'Data Distribution')
    
    # Determine the optimal number of clusters for K-Means
    optimal_k = find_optimal_k(X_selected)
    
    print("K-Means")
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Apply K-Means
    kmeans_labels, kmeans_model = apply_kmeans(X_selected, optimal_k)
    
    #plot_clusters(X_selected, kmeans_labels, 'K-Means Clustering')
    
    # Apply DBSCAN
    eps_range = np.arange(0.1, 1.0, 0.1)
    min_samples_range = range(2, 10)
    best_eps, best_min_samples = find_best_dbscan_params(X_selected, eps_range, min_samples_range)
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan.fit_predict(X_selected)
    
    #plot_clusters(X_selected, dbscan_labels, 'DBSCAN Clustering')
    
    # Apply Agglomerative Clustering
    best_linkage, best_clusters  = apply_agglomerative(X_selected.values)
    aggloClustering = AgglomerativeClustering(n_clusters=best_clusters, linkage=best_linkage)
    agglo_labels = aggloClustering.fit_predict(X_selected)
    
    print("Agglomerative Clustering")
    print(f"Best linkage method: {best_linkage}")
    print(f"Best number of clusters: {best_clusters}")
    
    #plot_dendrogram(X_selected, best_linkage)   
     
    # Apply Subtractive Clustering
    alpha_range = np.arange(1, 5, 0.5)
    beta_range = np.arange(1, 5, 0.5)
    radius_range = np.arange(0.5, 2.0, 0.5)
    min_potential_range = np.logspace(-6, -2, 5)
    best_alpha, best_beta, best_radius, best_min_potential = find_best_subtractive_params(X_selected, alpha_range, beta_range, radius_range, min_potential_range)
    centers = subtractive_clustering(X_selected, alpha=best_alpha, beta=best_beta, radius=best_radius, min_potential=best_min_potential)
    clusters = assign_clusters(X_selected, centers)
    
    print("Subtractive Clustering")
    print(f"Best alpha: {best_alpha}")
    print(f"Best beta: {best_beta}")
    print(f"Best radius: {best_radius}")
    print(f"Best min_potential: {best_min_potential}")

    #plot_clusters(X_selected, clusters, 'Subtractive Clustering')
    
    kmeans_silhouette = silhouette_score(X_selected, kmeans_labels)
    dbscan_silhouette = silhouette_score(X_selected, dbscan_labels)
    aggloClustering_silhouette = silhouette_score(X_selected, agglo_labels)
    subtractive_clustering_silhouette = silhouette_score(X_selected, clusters)
    
    print(f"K-Means Silhouette Score: {kmeans_silhouette}")
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
    print(f"Agglomerative Clustering Silhouette Score: {aggloClustering_silhouette}")
    print(f"Subtractive Clustering Silhouette Score: {subtractive_clustering_silhouette}")
    print(f"K-Means SSE: {kmeans_model.inertia_}")
    #print("SSE is not applicable to DBSCAN, as there are no defined centroids.")
    #print("SSE is not applicable to Agglomerative Clustering, as there are no defined centroids.")
    print(f"Subtractive Clustering SSE: {SSE(X_selected, centers, clusters)}")
    
    # Evaluate clusters
    print("K-Means")
    #evaluate_clusters(X_selected, kmeans_labels,selected_features_with_target, 'TARGET') 
    print("DBSCAN")
    evaluate_clusters(X_selected, dbscan_labels, selected_features_with_target, 'TARGET')
    print("Agglomerative Clustering")
    #evaluate_clusters(X_selected, agglo_labels, selected_features_with_target, 'TARGET')
    print("Subtractive Clustering")
    #evaluate_clusters(X_selected, clusters, selected_features_with_target, 'TARGET')