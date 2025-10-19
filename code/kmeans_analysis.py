"""
K-Means Clustering Analysis Module
Implementation and evaluation of K-Means on Wine dataset
"""

import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Import custom modules
from data_preprocessing import load_wine_data
from utils import (
    calculate_clustering_metrics, calculate_classification_metrics,
    save_metrics_to_csv, print_metrics, format_execution_time,
    calculate_cluster_stability, get_figure_path
)
from visualizations import (
    plot_elbow_curve, plot_silhouette_scores,
    plot_clusters_2d, plot_pca_2d,
    plot_silhouette_histogram, plot_cluster_stability
)


class KMeansAnalyzer:
    """
    K-Means Clustering Analysis Class for Wine Dataset
    """
    
    def __init__(self, X, y, feature_names, target_names):
        """
        Initialize K-Means Analyzer.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            True labels (for evaluation)
        feature_names : list
            Feature names
        target_names : list
            Target class names
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        self.model = None
        self.optimal_k = None
        self.X_scaled = None
        self.scaler = None
        self.labels = None
    
    def prepare_data(self):
        """
        Prepare data: scale features.
        """
        print("\n" + "="*60)
        print("Data Preparation for K-Means")
        print("="*60)
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"✓ Data scaled using StandardScaler")
        print(f"  Original shape: {self.X.shape}")
        print(f"  Scaled shape: {self.X_scaled.shape}")
    
    def find_optimal_k_elbow(self, k_range=range(2, 11), n_init=10):
        """
        Find optimal K using elbow method.
        
        Parameters:
        -----------
        k_range : range or list
            Range of K values to test
        n_init : int
            Number of times K-Means will run with different centroid seeds
        
        Returns:
        --------
        results : dict
            Dictionary with K values, inertias, and silhouette scores
        """
        print("\n" + "="*60)
        print("Finding Optimal K using Elbow Method")
        print("="*60)
        
        inertias = []
        silhouette_scores = []
        
        start_time = time.time()
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            kmeans.fit(self.X_scaled)
            
            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.X_scaled, kmeans.labels_)
            
            inertias.append(inertia)
            silhouette_scores.append(silhouette)
            
            print(f"K={k:2d} | Inertia: {inertia:8.2f} | Silhouette: {silhouette:.4f}")
        
        elapsed_time = time.time() - start_time
        
        # Find optimal K based on silhouette score
        best_idx = np.argmax(silhouette_scores)
        self.optimal_k = list(k_range)[best_idx]
        
        print(f"\n✓ Optimal K found: {self.optimal_k}")
        print(f"  Best Silhouette Score: {silhouette_scores[best_idx]:.4f}")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")
        
        results = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': self.optimal_k
        }
        
        return results
    
    def train_model(self, n_clusters=None, n_init=10, max_iter=300):
        """
        Train K-Means model.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters (uses optimal_k if not specified)
        n_init : int
            Number of initializations
        max_iter : int
            Maximum iterations
        
        Returns:
        --------
        model : KMeans
            Trained K-Means model
        """
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 3
        
        print("\n" + "="*60)
        print("Training K-Means Model")
        print("="*60)
        print(f"Parameters:")
        print(f"  n_clusters: {n_clusters}")
        print(f"  n_init: {n_init}")
        print(f"  max_iter: {max_iter}")
        
        start_time = time.time()
        
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42
        )
        
        self.labels = self.model.fit_predict(self.X_scaled)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Model trained successfully!")
        print(f"  Training time: {format_execution_time(elapsed_time)}")
        print(f"  Iterations to converge: {self.model.n_iter_}")
        print(f"  Final inertia: {self.model.inertia_:.2f}")
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate K-Means clustering.
        
        Returns:
        --------
        results : dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("\n" + "="*60)
        print("K-Means Model Evaluation")
        print("="*60)
        
        # Calculate clustering metrics
        clustering_metrics = calculate_clustering_metrics(
            self.X_scaled,
            self.labels,
            true_labels=self.y
        )
        
        # Print cluster sizes
        print("\nCluster Sizes:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = 100 * count / len(self.labels)
            print(f"  Cluster {cluster_id}: {count:4d} samples ({percentage:.1f}%)")
        
        # Print metrics
        print_metrics(clustering_metrics, "Clustering Metrics")
        
        # Analyze cluster-class correspondence
        print("\n[Cluster-Class Correspondence Matrix]")
        self._print_cluster_class_matrix()
        
        # If we treat clusters as class predictions, calculate classification metrics
        classification_metrics = calculate_classification_metrics(self.y, self.labels)
        print_metrics(classification_metrics, "Classification Metrics (Clusters as Classes)")
        
        results = {
            'clustering_metrics': clustering_metrics,
            'classification_metrics': classification_metrics,
            'labels': self.labels,
            'centroids': self.model.cluster_centers_
        }
        
        return results
    
    def _print_cluster_class_matrix(self):
        """
        Print matrix showing correspondence between clusters and true classes.
        """
        n_clusters = len(np.unique(self.labels))
        n_classes = len(np.unique(self.y))
        
        matrix = np.zeros((n_clusters, n_classes), dtype=int)
        
        for cluster_id in range(n_clusters):
            cluster_mask = self.labels == cluster_id
            for class_id in range(n_classes):
                class_mask = self.y == class_id
                matrix[cluster_id, class_id] = np.sum(cluster_mask & class_mask)
        
        # Print header
        print(f"\n{'Cluster':<10}", end='')
        for class_id in range(n_classes):
            print(f"Class {class_id:>8}", end='')
        print(f"{'Total':>10}")
        print("-" * (10 + 10 * n_classes + 10))
        
        # Print rows
        for cluster_id in range(n_clusters):
            print(f"Cluster {cluster_id:<2}", end='')
            for class_id in range(n_classes):
                print(f"{matrix[cluster_id, class_id]:>10}", end='')
            print(f"{matrix[cluster_id].sum():>10}")
    
    def analyze_centroids(self):
        """
        Analyze cluster centroids to understand cluster characteristics.

        Returns:
        --------
        centroid_df : DataFrame
            DataFrame with centroid coordinates
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("\n" + "="*60)
        print("Cluster Centroid Analysis")
        print("="*60)

        import pandas as pd

        # Create DataFrame with centroids (in original scale)
        centroids_original = self.scaler.inverse_transform(self.model.cluster_centers_)

        centroid_df = pd.DataFrame(
            centroids_original,
            columns=self.feature_names,
            index=[f'Cluster {i}' for i in range(len(centroids_original))]
        )

        print("\nCentroid Coordinates (Original Scale):")
        print(centroid_df.round(3))

        # Find most distinctive features for each cluster
        print("\nMost Distinctive Features per Cluster:")
        for cluster_id in range(len(centroids_original)):
            centroid = centroids_original[cluster_id]
            mean_distances = np.abs(centroid - np.mean(self.X, axis=0))
            top_features_idx = np.argsort(mean_distances)[-3:][::-1]

            print(f"\n  Cluster {cluster_id}:")
            for idx in top_features_idx:
                feature_name = self.feature_names[idx]
                value = centroid[idx]
                mean_val = np.mean(self.X[:, idx])
                print(f"    - {feature_name}: {value:.3f} (dataset mean: {mean_val:.3f})")

        return centroid_df

    def analyze_cluster_stability(self, n_iterations=10, subsample_ratio=0.8):
        """
        Analyze cluster stability using resampling.

        Parameters:
        -----------
        n_iterations : int
            Number of resampling iterations
        subsample_ratio : float
            Ratio of data to sample in each iteration

        Returns:
        --------
        stability_results : dict
            Stability analysis results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("\n" + "="*60)
        print("Cluster Stability Analysis")
        print("="*60)

        # Get model parameters
        n_clusters = self.model.n_clusters
        n_init = self.model.n_init

        # Create a new model instance for stability testing
        test_model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)

        # Calculate stability
        stability_score, ari_scores = calculate_cluster_stability(
            self.X_scaled,
            test_model,
            n_iterations=n_iterations,
            subsample_ratio=subsample_ratio
        )

        print(f"\nCluster Stability Results:")
        print(f"  Mean ARI: {stability_score:.4f}")
        print(f"  Std ARI: {np.std(ari_scores):.4f}")
        print(f"  Min ARI: {np.min(ari_scores):.4f}")
        print(f"  Max ARI: {np.max(ari_scores):.4f}")

        if stability_score > 0.8:
            print(f"\n✓ Clusters are highly stable (ARI > 0.8)")
        elif stability_score > 0.6:
            print(f"\n✓ Clusters are moderately stable (ARI > 0.6)")
        else:
            print(f"\n⚠️  Clusters show low stability (ARI < 0.6)")

        results = {
            'mean_ari': stability_score,
            'std_ari': np.std(ari_scores),
            'ari_scores': ari_scores
        }

        return results
    
    def visualize_results(self, elbow_results=None, save_figures=True, stability_results=None):
        """
        Generate visualizations for K-Means results.

        Parameters:
        -----------
        elbow_results : dict, optional
            Results from elbow method
        save_figures : bool
            Whether to save figures
        stability_results : dict, optional
            Cluster stability analysis results
        """
        print("\n" + "="*60)
        print("Generating K-Means Visualizations")
        print("="*60)

        # 1. Elbow Curve
        if elbow_results:
            save_path = get_figure_path('kmeans/elbow_curve.png') if save_figures else None
            plot_elbow_curve(
                elbow_results['k_values'],
                elbow_results['inertias'],
                optimal_k=elbow_results['optimal_k'],
                save_path=save_path
            )

        # 2. Silhouette Scores
        if elbow_results:
            save_path = get_figure_path('kmeans/silhouette_scores.png') if save_figures else None
            plot_silhouette_scores(
                elbow_results['k_values'],
                elbow_results['silhouette_scores'],
                optimal_k=elbow_results['optimal_k'],
                save_path=save_path
            )

        # 3. Cluster Visualization (PCA) - WITH CENTROIDS
        save_path = get_figure_path('kmeans/clusters_pca.png') if save_figures else None
        plot_clusters_2d(
            self.X_scaled,
            self.labels,
            centroids=self.model.cluster_centers_,
            method_name='K-Means Clustering Results - PCA Projection',
            save_path=save_path
        )

        # 4. True Classes Visualization (for comparison)
        save_path = get_figure_path('kmeans/true_classes_pca.png') if save_figures else None
        plot_pca_2d(
            self.X_scaled,
            self.y,
            self.target_names,
            'True Class Labels - PCA Projection (Ground Truth)',
            save_path=save_path
        )

        # 5. Detailed cluster analysis visualization
        save_path = get_figure_path('kmeans/cluster_characteristics.png') if save_figures else None
        self._plot_cluster_characteristics(save_path)

        # 6. Feature distribution per cluster
        save_path = get_figure_path('kmeans/feature_distributions.png') if save_figures else None
        self._plot_feature_distributions(save_path)

        # 7. Silhouette histogram
        save_path = get_figure_path('kmeans/silhouette_histogram.png') if save_figures else None
        plot_silhouette_histogram(self.X_scaled, self.labels, save_path=save_path)

        # 8. Cluster stability (if available)
        if stability_results is not None:
            save_path = get_figure_path('kmeans/cluster_stability.png') if save_figures else None
            plot_cluster_stability(stability_results['ari_scores'], save_path=save_path)

        print("\n✓ All visualizations generated!")
    
    def _plot_cluster_characteristics(self, save_path=None):
        """
        Plot detailed characteristics of each cluster.
        """
        import matplotlib.pyplot as plt
        
        n_clusters = len(np.unique(self.labels))
        
        fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 5))
        if n_clusters == 1:
            axes = [axes]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for cluster_id in range(n_clusters):
            ax = axes[cluster_id]
            
            # Get cluster data
            cluster_mask = self.labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Count true classes in this cluster
            cluster_true_labels = self.y[cluster_mask]
            unique_classes, class_counts = np.unique(cluster_true_labels, return_counts=True)
            
            # Create bar chart showing class distribution
            ax.bar(unique_classes, class_counts, color=colors[:len(unique_classes)], 
                   alpha=0.7, edgecolor='black')
            
            # Add labels
            ax.set_xlabel('True Wine Class', fontsize=11)
            ax.set_ylabel('Number of Samples', fontsize=11)
            ax.set_title(f'Cluster {cluster_id}\n({cluster_size} samples)', 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(unique_classes)
            ax.set_xticklabels([self.target_names[i] for i in unique_classes], 
                              rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels on bars
            for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
                percentage = 100 * count / cluster_size
                ax.text(cls, count, f'{percentage:.1f}%', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Cluster Composition Analysis\n(Which True Classes are in Each Cluster?)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cluster characteristics saved: {save_path}")
        
        plt.show()
    
    def _plot_feature_distributions(self, save_path=None):
        """
        Plot feature distributions for each cluster.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Select top 6 most important features
        # Calculate variance of features across clusters
        centroids_original = self.scaler.inverse_transform(self.model.cluster_centers_)
        feature_variance = np.var(centroids_original, axis=0)
        top_features_idx = np.argsort(feature_variance)[-6:][::-1]
        
        n_features = len(top_features_idx)
        n_clusters = len(np.unique(self.labels))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, feature_idx in enumerate(top_features_idx):
            ax = axes[idx]
            feature_name = self.feature_names[feature_idx]
            
            # Plot distribution for each cluster
            for cluster_id in range(n_clusters):
                cluster_mask = self.labels == cluster_id
                feature_values = self.X[cluster_mask, feature_idx]
                
                ax.hist(feature_values, bins=15, alpha=0.5, 
                       label=f'Cluster {cluster_id}',
                       color=colors[cluster_id % len(colors)],
                       edgecolor='black')
            
            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{feature_name} Distribution', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Feature Distributions Across Clusters\n(Top 6 Most Discriminative Features)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature distributions saved: {save_path}")
        
        plt.show()


def run_kmeans_analysis(source='sklearn', wine_type='red', save_results=True, enhanced=True):
    """
    Complete K-Means analysis pipeline.

    Parameters:
    -----------
    source : str
        Dataset source ('sklearn' or 'kaggle')
    wine_type : str
        Wine type for Kaggle dataset ('red' or 'white')
    save_results : bool
        Whether to save results
    enhanced : bool
        Whether to run enhanced analyses (stability, silhouette histogram)

    Returns:
    --------
    analyzer : KMeansAnalyzer
        Trained K-Means analyzer object
    results : dict
        Analysis results
    """
    print("\n" + "🍷"*30)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("🍷"*30)

    # Load data
    X, y, feature_names, target_names = load_wine_data(source=source, wine_type=wine_type)

    # Initialize analyzer
    analyzer = KMeansAnalyzer(X, y, feature_names, target_names)

    # Prepare data
    analyzer.prepare_data()

    # Find optimal K
    elbow_results = analyzer.find_optimal_k_elbow(k_range=range(2, 11))

    # Train model with optimal K
    analyzer.train_model(n_clusters=elbow_results['optimal_k'])

    # Evaluate model
    eval_results = analyzer.evaluate_model()

    # Analyze centroids
    centroid_df = analyzer.analyze_centroids()

    # Enhanced analyses
    stability_results = None
    if enhanced:
        # Analyze cluster stability
        stability_results = analyzer.analyze_cluster_stability(n_iterations=10, subsample_ratio=0.8)

    # Visualize results
    analyzer.visualize_results(elbow_results, save_figures=save_results, stability_results=stability_results)

    # Save metrics
    if save_results:
        # Combine both types of metrics
        combined_metrics = {
            **eval_results['clustering_metrics'],
            **eval_results['classification_metrics']
        }
        # Add stability metrics if available
        if stability_results:
            combined_metrics['stability_mean_ari'] = stability_results['mean_ari']
            combined_metrics['stability_std_ari'] = stability_results['std_ari']

        save_metrics_to_csv(
            combined_metrics,
            'results_summary.csv',
            'K-Means'
        )

    print("\n" + "="*60)
    print("✓ K-Means Analysis Complete!")
    print("="*60)

    results = {
        'model': analyzer.model,
        'optimal_k': analyzer.optimal_k,
        'labels': analyzer.labels,
        'centroids': analyzer.model.cluster_centers_,
        'metrics': eval_results,
        'centroid_analysis': centroid_df,
        'elbow_results': elbow_results,
        'stability_results': stability_results
    }

    return analyzer, results


if __name__ == "__main__":
    # Run complete K-Means analysis
    analyzer, results = run_kmeans_analysis(source='sklearn', save_results=True)
    
    print("\n✓ K-Means analysis completed successfully!")
    print(f"  Optimal K: {results['optimal_k']}")
    print(f"  Silhouette Score: {results['metrics']['clustering_metrics']['silhouette_score']:.4f}")