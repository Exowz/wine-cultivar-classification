"""
CAH (Clustering Agglomératif Hiérarchique) Analysis Module
Hierarchical Clustering Implementation and Evaluation on Wine Dataset
"""

import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Import custom modules
from data_preprocessing import load_wine_data
from utils import (
    calculate_clustering_metrics, calculate_classification_metrics,
    save_metrics_to_csv, print_metrics, format_execution_time,
    calculate_cluster_stability, get_figure_path
)
from visualizations import (
    plot_dendrogram, plot_clusters_2d, plot_pca_2d,
    plot_silhouette_histogram, plot_cluster_stability
)


class CAHAnalyzer:
    """
    Hierarchical Clustering (CAH) Analysis Class for Wine Dataset
    """
    
    def __init__(self, X, y, feature_names, target_names):
        """
        Initialize CAH Analyzer.
        
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
        self.linkage_matrix = None
        self.X_scaled = None
        self.scaler = None
        self.labels = None
    
    def prepare_data(self):
        """
        Prepare data: scale features.
        """
        print("\n" + "="*60)
        print("Data Preparation for CAH")
        print("="*60)
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"✓ Data scaled using StandardScaler")
        print(f"  Original shape: {self.X.shape}")
        print(f"  Scaled shape: {self.X_scaled.shape}")
    
    def find_optimal_clusters(self, k_range=range(2, 11), linkage='ward'):
        """
        Find optimal number of clusters using silhouette analysis.
        
        Parameters:
        -----------
        k_range : range or list
            Range of cluster numbers to test
        linkage : str
            Linkage method
        
        Returns:
        --------
        results : dict
            Dictionary with K values and silhouette scores
        """
        print("\n" + "="*60)
        print("Finding Optimal Number of Clusters for CAH")
        print("="*60)
        
        silhouette_scores = []
        davies_bouldin_scores = []
        
        start_time = time.time()
        
        for k in k_range:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage,
                metric='euclidean'
            )
            labels = model.fit_predict(self.X_scaled)
            
            silhouette = silhouette_score(self.X_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
            
            silhouette_scores.append(silhouette)
            davies_bouldin_scores.append(davies_bouldin)
            
            print(f"K={k:2d} | Silhouette: {silhouette:.4f} | Davies-Bouldin: {davies_bouldin:.4f}")
        
        elapsed_time = time.time() - start_time
        
        # Find optimal K based on silhouette score
        best_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[best_idx]
        
        print(f"\n✓ Optimal K found: {optimal_k}")
        print(f"  Best Silhouette Score: {silhouette_scores[best_idx]:.4f}")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")
        
        results = {
            'k_values': list(k_range),
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'optimal_k': optimal_k
        }
        
        return results
    
    def compute_linkage_matrix(self, method='ward', metric='euclidean'):
        """
        Compute linkage matrix for dendrogram.
        
        Parameters:
        -----------
        method : str
            Linkage method ('ward', 'complete', 'average', 'single')
        metric : str
            Distance metric
        
        Returns:
        --------
        linkage_matrix : array
            Hierarchical clustering linkage matrix
        """
        print("\n" + "="*60)
        print(f"Computing Linkage Matrix")
        print("="*60)
        print(f"  Method: {method}")
        print(f"  Metric: {metric}")
        
        start_time = time.time()
        
        # For ward method, metric must be euclidean
        if method == 'ward':
            metric = 'euclidean'
        
        self.linkage_matrix = linkage(self.X_scaled, method=method, metric=metric)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Linkage matrix computed!")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")
        
        return self.linkage_matrix
    
    def train_model(self, n_clusters=3, linkage='ward', metric='euclidean'):
        """
        Train hierarchical clustering model.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        linkage : str
            Linkage method ('ward', 'complete', 'average', 'single')
        metric : str
            Distance metric
        
        Returns:
        --------
        model : AgglomerativeClustering
            Trained model
        """
        print("\n" + "="*60)
        print("Training CAH Model")
        print("="*60)
        print(f"Parameters:")
        print(f"  n_clusters: {n_clusters}")
        print(f"  linkage: {linkage}")
        print(f"  metric: {metric}")
        
        start_time = time.time()
        
        # For ward linkage, metric must be euclidean
        if linkage == 'ward':
            metric = 'euclidean'
        
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric
        )
        
        self.labels = self.model.fit_predict(self.X_scaled)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Model trained successfully!")
        print(f"  Training time: {format_execution_time(elapsed_time)}")
        
        return self.model
    
    def compare_linkage_methods(self, n_clusters=3):
        """
        Compare different linkage methods.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        
        Returns:
        --------
        results : dict
            Comparison results
        """
        print("\n" + "="*60)
        print("Comparing Different Linkage Methods")
        print("="*60)
        
        linkage_methods = ['ward', 'complete', 'average', 'single']
        results = {}
        
        for method in linkage_methods:
            print(f"\nTesting {method.upper()} linkage...")
            
            # Train model
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=method,
                metric='euclidean'
            )
            labels = model.fit_predict(self.X_scaled)
            
            # Calculate metrics
            metrics = calculate_clustering_metrics(
                self.X_scaled,
                labels,
                true_labels=self.y
            )
            
            results[method] = {
                'labels': labels,
                'metrics': metrics
            }
            
            # Print key metrics
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
            print(f"  Purity: {metrics['purity']:.4f}")
        
        # Find best method based on silhouette score
        best_method = max(
            results.keys(),
            key=lambda x: results[x]['metrics']['silhouette_score']
        )
        
        print(f"\n✓ Best linkage method: {best_method.upper()}")
        print(f"  Silhouette Score: {results[best_method]['metrics']['silhouette_score']:.4f}")
        
        return results
    
    def evaluate_model(self):
        """
        Evaluate hierarchical clustering.
        
        Returns:
        --------
        results : dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("\n" + "="*60)
        print("CAH Model Evaluation")
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
            'labels': self.labels
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
    
    def analyze_cluster_characteristics(self):
        """
        Analyze characteristics of each cluster.

        Returns:
        --------
        cluster_stats : dict
            Statistics for each cluster
        """
        print("\n" + "="*60)
        print("Cluster Characteristics Analysis")
        print("="*60)

        import pandas as pd

        cluster_stats = {}

        for cluster_id in np.unique(self.labels):
            cluster_mask = self.labels == cluster_id
            cluster_data = self.X[cluster_mask]

            # Calculate statistics
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)

            cluster_stats[f'Cluster {cluster_id}'] = {
                'mean': cluster_mean,
                'std': cluster_std,
                'size': np.sum(cluster_mask)
            }

            print(f"\nCluster {cluster_id} (n={np.sum(cluster_mask)}):")
            print(f"  Top 3 distinctive features (by mean):")

            # Find most distinctive features
            overall_mean = np.mean(self.X, axis=0)
            deviations = np.abs(cluster_mean - overall_mean)
            top_features_idx = np.argsort(deviations)[-3:][::-1]

            for idx in top_features_idx:
                feature_name = self.feature_names[idx]
                value = cluster_mean[idx]
                overall = overall_mean[idx]
                print(f"    - {feature_name}: {value:.3f} (overall: {overall:.3f})")

        return cluster_stats

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
        linkage = self.model.linkage
        metric = self.model.metric

        # Create a new model instance for stability testing
        test_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric
        )

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
    
    def visualize_results(self, save_figures=True, stability_results=None):
        """
        Generate visualizations for CAH results.

        Parameters:
        -----------
        save_figures : bool
            Whether to save figures
        stability_results : dict, optional
            Cluster stability analysis results
        """
        print("\n" + "="*60)
        print("Generating CAH Visualizations")
        print("="*60)

        # 1. Dendrogram - Shows hierarchical structure
        if self.linkage_matrix is not None:
            save_path = get_figure_path('cah/dendrogram.png') if save_figures else None
            plot_dendrogram(
                self.linkage_matrix,
                'Hierarchical Clustering Dendrogram\n(Shows how clusters merge at different distances)',
                n_clusters=len(np.unique(self.labels)),
                save_path=save_path
            )

        # 2. Cluster Visualization (PCA)
        save_path = get_figure_path('cah/clusters_pca.png') if save_figures else None
        plot_clusters_2d(
            self.X_scaled,
            self.labels,
            method_name='CAH Clustering Results - PCA Projection',
            save_path=save_path
        )

        # 3. True Classes Visualization (for comparison)
        save_path = get_figure_path('cah/true_classes_pca.png') if save_figures else None
        plot_pca_2d(
            self.X_scaled,
            self.y,
            self.target_names,
            'True Class Labels - PCA Projection (Ground Truth)',
            save_path=save_path
        )

        # 4. Detailed cluster analysis
        save_path = get_figure_path('cah/cluster_characteristics.png') if save_figures else None
        self._plot_cluster_characteristics(save_path)

        # 5. Feature distribution per cluster
        save_path = get_figure_path('cah/feature_distributions.png') if save_figures else None
        self._plot_feature_distributions(save_path)

        # 6. Silhouette histogram
        save_path = get_figure_path('cah/silhouette_histogram.png') if save_figures else None
        plot_silhouette_histogram(self.X_scaled, self.labels, save_path=save_path)

        # 7. Cluster stability (if available)
        if stability_results is not None:
            save_path = get_figure_path('cah/cluster_stability.png') if save_figures else None
            plot_cluster_stability(stability_results['ari_scores'], save_path=save_path)

        print("\n✓ All CAH visualizations generated!")
    
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
        
        plt.suptitle('CAH - Cluster Composition Analysis\n(Which True Classes are in Each Cluster?)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ CAH cluster characteristics saved: {save_path}")
        
        plt.show()
    
    def _plot_feature_distributions(self, save_path=None):
        """
        Plot feature distributions for each cluster.
        """
        import matplotlib.pyplot as plt
        
        # Calculate mean features per cluster to find most discriminative
        cluster_means = []
        for cluster_id in range(len(np.unique(self.labels))):
            cluster_mask = self.labels == cluster_id
            cluster_data = self.X[cluster_mask]
            cluster_means.append(np.mean(cluster_data, axis=0))
        
        cluster_means = np.array(cluster_means)
        feature_variance = np.var(cluster_means, axis=0)
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
        
        plt.suptitle('CAH - Feature Distributions Across Clusters\n(Top 6 Most Discriminative Features)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ CAH feature distributions saved: {save_path}")
        
        plt.show()


def run_cah_analysis(source='sklearn', wine_type='red', n_clusters=3,
                     linkage='ward', compare_methods=True, save_results=True, enhanced=True):
    """
    Complete CAH analysis pipeline.

    Parameters:
    -----------
    source : str
        Dataset source ('sklearn' or 'kaggle')
    wine_type : str
        Wine type for Kaggle dataset ('red' or 'white')
    n_clusters : int
        Number of clusters
    linkage : str
        Linkage method
    compare_methods : bool
        Whether to compare different linkage methods
    save_results : bool
        Whether to save results
    enhanced : bool
        Whether to run enhanced analyses (stability, silhouette histogram)

    Returns:
    --------
    analyzer : CAHAnalyzer
        Trained CAH analyzer object
    results : dict
        Analysis results
    """
    print("\n" + "🍷"*30)
    print("HIERARCHICAL CLUSTERING (CAH) ANALYSIS")
    print("🍷"*30)

    # Load data
    X, y, feature_names, target_names = load_wine_data(source=source, wine_type=wine_type)

    # Initialize analyzer
    analyzer = CAHAnalyzer(X, y, feature_names, target_names)

    # Prepare data
    analyzer.prepare_data()

    # Compare linkage methods if requested
    comparison_results = None
    if compare_methods:
        comparison_results = analyzer.compare_linkage_methods(n_clusters=n_clusters)

    # Compute linkage matrix for dendrogram
    analyzer.compute_linkage_matrix(method=linkage)

    # Train model
    analyzer.train_model(n_clusters=n_clusters, linkage=linkage)

    # Evaluate model
    eval_results = analyzer.evaluate_model()

    # Analyze cluster characteristics
    cluster_stats = analyzer.analyze_cluster_characteristics()

    # Enhanced analyses
    stability_results = None
    if enhanced:
        # Analyze cluster stability
        stability_results = analyzer.analyze_cluster_stability(n_iterations=10, subsample_ratio=0.8)

    # Visualize results
    analyzer.visualize_results(save_figures=save_results, stability_results=stability_results)

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
            'CAH'
        )

    print("\n" + "="*60)
    print("✓ CAH Analysis Complete!")
    print("="*60)

    results = {
        'model': analyzer.model,
        'labels': analyzer.labels,
        'linkage_matrix': analyzer.linkage_matrix,
        'metrics': eval_results,
        'cluster_stats': cluster_stats,
        'comparison': comparison_results,
        'stability_results': stability_results
    }

    return analyzer, results


if __name__ == "__main__":
    # Run complete CAH analysis
    analyzer, results = run_cah_analysis(
        source='sklearn',
        n_clusters=3,
        linkage='ward',
        compare_methods=True,
        save_results=True
    )
    
    print("\n✓ CAH analysis completed successfully!")
    print(f"  Number of clusters: {len(np.unique(results['labels']))}")
    print(f"  Silhouette Score: {results['metrics']['clustering_metrics']['silhouette_score']:.4f}")