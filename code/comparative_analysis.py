"""
Comparative Analysis Module
Compare KNN, K-Means, and CAH performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from data_preprocessing import load_wine_data
from knn_analysis import KNNAnalyzer
from kmeans_analysis import KMeansAnalyzer
from cah_analysis import CAHAnalyzer
from utils import prepare_data, format_execution_time, get_project_root, get_figure_path
from visualizations import plot_comparative_metrics


class ComparativeAnalyzer:
    """
    Comparative Analysis for KNN, K-Means, and CAH
    """
    
    def __init__(self, X, y, feature_names, target_names):
        """
        Initialize Comparative Analyzer.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Feature names
        target_names : list
            Target class names
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        self.results = {}
        self.timing_results = {}
    
    def run_all_methods(self, save_results=False):
        """
        Run KNN, K-Means, and CAH analyses.
        
        Parameters:
        -----------
        save_results : bool
            Whether to save individual results
        
        Returns:
        --------
        results : dict
            Results from all methods
        """
        print("\n" + "🍷"*30)
        print("COMPARATIVE ANALYSIS: KNN vs K-Means vs CAH")
        print("🍷"*30)
        
        # 1. KNN Analysis
        print("\n[1/3] Running KNN Analysis...")
        start_time = time.time()
        
        knn_analyzer = KNNAnalyzer(self.X, self.y, self.feature_names, self.target_names)
        knn_analyzer.prepare_data()
        best_k, opt_results = knn_analyzer.find_optimal_k(k_range=range(1, 21))
        knn_analyzer.train_model(k=best_k)
        knn_eval = knn_analyzer.evaluate_model()
        
        # VISUALIZE KNN RESULTS
        if save_results:
            print("\n[Generating KNN Visualizations...]")
            knn_analyzer.visualize_results(
                knn_eval['y_test_pred'],
                opt_results,
                save_figures=True
            )
        
        knn_time = time.time() - start_time
        
        self.results['KNN'] = {
            'analyzer': knn_analyzer,
            'metrics': knn_eval['test_metrics'],
            'best_k': best_k
        }
        self.timing_results['KNN'] = knn_time
        
        print(f"✓ KNN completed in {format_execution_time(knn_time)}")
        
        # 2. K-Means Analysis
        print("\n[2/3] Running K-Means Analysis...")
        start_time = time.time()
        
        kmeans_analyzer = KMeansAnalyzer(self.X, self.y, self.feature_names, self.target_names)
        kmeans_analyzer.prepare_data()
        elbow_results = kmeans_analyzer.find_optimal_k_elbow(k_range=range(2, 11))
        kmeans_analyzer.train_model(n_clusters=elbow_results['optimal_k'])
        kmeans_eval = kmeans_analyzer.evaluate_model()
        
        # VISUALIZE K-MEANS RESULTS
        if save_results:
            print("\n[Generating K-Means Visualizations...]")
            kmeans_analyzer.visualize_results(elbow_results, save_figures=True)
        
        kmeans_time = time.time() - start_time
        
        self.results['K-Means'] = {
            'analyzer': kmeans_analyzer,
            'metrics': {**kmeans_eval['clustering_metrics'], **kmeans_eval['classification_metrics']},
            'optimal_k': elbow_results['optimal_k']
        }
        self.timing_results['K-Means'] = kmeans_time
        
        print(f"✓ K-Means completed in {format_execution_time(kmeans_time)}")
        
        # 3. CAH Analysis
        print("\n[3/3] Running CAH Analysis...")
        start_time = time.time()
        
        cah_analyzer = CAHAnalyzer(self.X, self.y, self.feature_names, self.target_names)
        cah_analyzer.prepare_data()
        cah_analyzer.compute_linkage_matrix(method='ward')
        
        # Use the same optimal K as K-Means for fair comparison
        optimal_clusters = elbow_results['optimal_k']
        print(f"  Using K={optimal_clusters} clusters (same as K-Means for comparison)")
        
        cah_analyzer.train_model(n_clusters=optimal_clusters, linkage='ward')
        cah_eval = cah_analyzer.evaluate_model()
        
        # VISUALIZE CAH RESULTS
        if save_results:
            print("\n[Generating CAH Visualizations...]")
            cah_analyzer.visualize_results(save_figures=True)
        
        cah_time = time.time() - start_time
        
        self.results['CAH'] = {
            'analyzer': cah_analyzer,
            'metrics': {**cah_eval['clustering_metrics'], **cah_eval['classification_metrics']},
        }
        self.timing_results['CAH'] = cah_time
        
        print(f"✓ CAH completed in {format_execution_time(cah_time)}")
        
        return self.results
    
    def generate_comparison_report(self):
        """
        Generate comprehensive comparison report.
        
        Returns:
        --------
        report_df : DataFrame
            Comparison report
        """
        print("\n" + "="*60)
        print("COMPARATIVE PERFORMANCE REPORT")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for method_name, method_data in self.results.items():
            metrics = method_data['metrics']
            
            row = {
                'Method': method_name,
                'Accuracy': metrics.get('accuracy', None),
                'Precision': metrics.get('precision', None),
                'Recall': metrics.get('recall', None),
                'F1-Score': metrics.get('f1_score', None),
                'Silhouette': metrics.get('silhouette_score', None),
                'Davies-Bouldin': metrics.get('davies_bouldin_score', None),
                'Purity': metrics.get('purity', None),
                'Execution Time (s)': self.timing_results.get(method_name, None)
            }
            
            comparison_data.append(row)
        
        report_df = pd.DataFrame(comparison_data)
        
        # Print report
        print("\n[Classification Metrics]")
        class_cols = ['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        print(report_df[class_cols].to_string(index=False))
        
        print("\n[Clustering Metrics]")
        cluster_cols = ['Method', 'Silhouette', 'Davies-Bouldin', 'Purity']
        available_cluster_cols = [col for col in cluster_cols if col in report_df.columns]
        if available_cluster_cols:
            cluster_df = report_df[available_cluster_cols].copy()
            # Only show clustering methods
            cluster_df = cluster_df[cluster_df['Method'].isin(['K-Means', 'CAH'])]
            print(cluster_df.to_string(index=False))
        
        print("\n[Execution Time Comparison]")
        time_df = report_df[['Method', 'Execution Time (s)']].copy()
        print(time_df.to_string(index=False))
        
        # Identify best performers
        print("\n" + "="*60)
        print("BEST PERFORMERS")
        print("="*60)
        
        best_accuracy = report_df.loc[report_df['Accuracy'].idxmax()]
        print(f"Best Accuracy: {best_accuracy['Method']} ({best_accuracy['Accuracy']:.4f})")
        
        best_f1 = report_df.loc[report_df['F1-Score'].idxmax()]
        print(f"Best F1-Score: {best_f1['Method']} ({best_f1['F1-Score']:.4f})")
        
        # For clustering methods only
        clustering_df = report_df[report_df['Silhouette'].notna()]
        if not clustering_df.empty:
            best_silhouette = clustering_df.loc[clustering_df['Silhouette'].idxmax()]
            print(f"Best Silhouette Score: {best_silhouette['Method']} ({best_silhouette['Silhouette']:.4f})")
        
        fastest = report_df.loc[report_df['Execution Time (s)'].idxmin()]
        print(f"Fastest Method: {fastest['Method']} ({format_execution_time(fastest['Execution Time (s)'])})")
        
        return report_df
    
    def generate_insights(self):
        """
        Generate insights and recommendations.
        """
        print("\n" + "="*60)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Get metrics
        knn_acc = self.results['KNN']['metrics']['accuracy']
        kmeans_sil = self.results['K-Means']['metrics'].get('silhouette_score', 0)
        cah_sil = self.results['CAH']['metrics'].get('silhouette_score', 0)
        
        print("\n📊 Performance Analysis:")
        print(f"  • KNN achieved {knn_acc:.2%} accuracy in classification")
        print(f"  • K-Means: Silhouette score of {kmeans_sil:.4f}")
        print(f"  • CAH: Silhouette score of {cah_sil:.4f}")
        
        print("\n🎯 Method Comparison:")
        print("  • KNN (Supervised):")
        print("    - Best for classification with labeled data")
        print("    - Interpretable decision boundaries")
        print("    - Requires optimal K selection")
        
        print("  • K-Means (Unsupervised):")
        print("    - Fast and scalable")
        print("    - Works well with spherical clusters")
        print("    - Sensitive to initialization")
        
        print("  • CAH (Unsupervised):")
        print("    - Provides hierarchical structure")
        print("    - No need to specify K in advance (dendrogram)")
        print("    - More computationally expensive")
        
        print("\n💡 Use Case Recommendations:")
        print("  • Use KNN when: You have labeled data and need predictions")
        print("  • Use K-Means when: Fast clustering on large datasets")
        print("  • Use CAH when: Understanding hierarchical relationships is important")
        
        print("\n🍷 Wine Dataset Specific Insights:")
        print("  • The dataset has natural clusters corresponding to wine cultivars")
        print("  • All three methods successfully identify wine types")
        print("  • Feature scaling is crucial for distance-based methods")
    
    def plot_comparative_visualizations(self, save_figures=True):
        """
        Generate comparative visualizations.
        
        Parameters:
        -----------
        save_figures : bool
            Whether to save figures
        """
        print("\n" + "="*60)
        print("Generating Comparative Visualizations")
        print("="*60)
        
        # 1. Metrics Comparison Bar Chart
        report_df = self.generate_comparison_report()

        save_path = get_figure_path('comparative/metrics_comparison.png') if save_figures else None
        plot_comparative_metrics(report_df, save_path=save_path)

        # 2. Execution Time Comparison
        save_path = get_figure_path('comparative/execution_time.png') if save_figures else None
        self._plot_execution_time_comparison(save_path)

        # 3. Side-by-side cluster visualizations
        save_path = get_figure_path('comparative/methods_comparison_pca.png') if save_figures else None
        self._plot_methods_comparison(save_path)
        
        print("\n✓ All comparative visualizations generated!")
    
    def _plot_execution_time_comparison(self, save_path=None):
        """
        Plot execution time comparison.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(self.timing_results.keys())
        times = list(self.timing_results.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Execution time comparison saved: {save_path}")
        
        plt.show()
    
    def _plot_methods_comparison(self, save_path=None):
        """
        Plot side-by-side comparison of all methods.
        """
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Prepare data
        knn_analyzer = self.results['KNN']['analyzer']
        kmeans_analyzer = self.results['K-Means']['analyzer']
        cah_analyzer = self.results['CAH']['analyzer']
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(knn_analyzer.X_test)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # KNN
        ax = axes[0]
        y_pred = self.results['KNN']['analyzer'].model.predict(knn_analyzer.X_test)
        for i in np.unique(y_pred):
            mask = y_pred == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=colors[i % len(colors)], label=f'Class {i}',
                      alpha=0.6, edgecolors='black', linewidth=0.5, s=100)
        ax.set_title('KNN Predictions', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # K-Means
        ax = axes[1]
        X_full_pca = pca.transform(kmeans_analyzer.X_scaled)
        for i in np.unique(kmeans_analyzer.labels):
            mask = kmeans_analyzer.labels == i
            ax.scatter(X_full_pca[mask, 0], X_full_pca[mask, 1],
                      c=colors[i % len(colors)], label=f'Cluster {i}',
                      alpha=0.6, edgecolors='black', linewidth=0.5, s=100)
        ax.set_title('K-Means Clustering', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # CAH
        ax = axes[2]
        for i in np.unique(cah_analyzer.labels):
            mask = cah_analyzer.labels == i
            ax.scatter(X_full_pca[mask, 0], X_full_pca[mask, 1],
                      c=colors[i % len(colors)], label=f'Cluster {i}',
                      alpha=0.6, edgecolors='black', linewidth=0.5, s=100)
        ax.set_title('CAH Clustering', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Comparison: KNN vs K-Means vs CAH (PCA Projection)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Methods comparison saved: {save_path}")
        
        plt.show()


def run_comparative_analysis(source='sklearn', wine_type='red', save_results=True):
    """
    Run complete comparative analysis.
    
    Parameters:
    -----------
    source : str
        Dataset source
    wine_type : str
        Wine type for Kaggle
    save_results : bool
        Whether to save results
    
    Returns:
    --------
    analyzer : ComparativeAnalyzer
        Comparative analyzer object
    report_df : DataFrame
        Comparison report
    """
    # Load data
    X, y, feature_names, target_names = load_wine_data(source=source, wine_type=wine_type)
    
    # Initialize comparative analyzer
    analyzer = ComparativeAnalyzer(X, y, feature_names, target_names)
    
    # Run all methods
    analyzer.run_all_methods(save_results=True)
    
    # Generate report
    report_df = analyzer.generate_comparison_report()
    
    # Generate insights
    analyzer.generate_insights()
    
    # Generate visualizations
    if save_results:
        analyzer.plot_comparative_visualizations(save_figures=True)
    
    # Save report
    if save_results:
        project_root = get_project_root()
        report_path = project_root / 'results' / 'metrics' / 'comparative_report.csv'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False)
        print(f"\n✓ Comparative report saved: {report_path}")
    
    return analyzer, report_df


if __name__ == "__main__":
    print("\n" + "🍷"*30)
    print("COMPARATIVE ANALYSIS MODULE")
    print("🍷"*30)
    
    analyzer, report = run_comparative_analysis(source='sklearn', save_results=True)
    
    print("\n✓ Comparative analysis completed successfully!")