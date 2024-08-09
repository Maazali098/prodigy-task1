import numpy as np
import pandas as pd

class KMeans:
    def _init_(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            
            labels = self._assign_labels(X, self.centroids)
           
            new_centroids = self._calculate_centroids(X, labels)
           
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        
        return labels
    
    def _assign_labels(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        
        return np.argmin(distances, axis=0)
    
    def _calculate_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
        return centroids

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    X = df.select_dtypes(include=np.number).values
    
    return df, X
if 'name' == '_main_':
    file_path = 'Mall_Customers.csv'
    
    df, X = load_data(file_path)
    
    kmeans = KMeans(n_clusters=3)
    
    labels = kmeans.fit(X)
    
    df['Cluster'] = labels
    
    print("Cluster centroids:")
    print(kmeans.centroids)
    
    print("Cluster labels:")
    print(labels)
    
    print(df.head())