"""
Visualization Module for Wine Dataset Analysis
Provides consistent, publication-quality plots for all analyses
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.cluster.hierarchy import dendrogram
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_confusion_matrix(y_true, y_pred, target_names, method_name, save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list
        Class names
    method_name : str
        Name of the method
    save_path : str, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    
    plt.title(f'Confusion Matrix - {method_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    
    plt.show()


def plot_data_distribution_2d(X, y, target_names, method_name='Dataset', 
                               feature_idx=(0, 1), feature_names=None, save_path=None):
    """
    Plot 2D scatter plot of data distribution.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Labels
    target_names : list
        Class names
    method_name : str
        Title for the plot
    feature_idx : tuple
        Indices of features to plot
    feature_names : list, optional
        Feature names
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, target_name in enumerate(target_names):
        mask = y == i
        ax.scatter(
            X[mask, feature_idx[0]], 
            X[mask, feature_idx[1]],
            c=colors[i % len(colors)],
            label=target_name,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=100
        )
    
    if feature_names:
        ax.set_xlabel(feature_names[feature_idx[0]], fontsize=12)
        ax.set_ylabel(feature_names[feature_idx[1]], fontsize=12)
    else:
        ax.set_xlabel(f'Feature {feature_idx[0]}', fontsize=12)
        ax.set_ylabel(f'Feature {feature_idx[1]}', fontsize=12)
    
    ax.set_title(f'Data Distribution - {method_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plot saved: {save_path}")
    
    plt.show()


def plot_pca_2d(X, y, target_names, method_name='PCA Visualization', save_path=None):
    """
    Plot 2D PCA projection.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Labels
    target_names : list
        Class names
    method_name : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, target_name in enumerate(target_names):
        mask = y == i
        ax.scatter(
            X_pca[mask, 0], 
            X_pca[mask, 1],
            c=colors[i % len(colors)],
            label=target_name,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=100
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title(method_name, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PCA plot saved: {save_path}")
    
    plt.show()
    
    return pca.explained_variance_ratio_


def plot_elbow_curve(K_range, inertias, optimal_k=None, save_path=None):
    """
    Plot elbow curve for K-Means.
    
    Parameters:
    -----------
    K_range : array-like
        Range of K values
    inertias : array-like
        Inertia values for each K
    optimal_k : int, optional
        Optimal K value to highlight
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
    
    if optimal_k:
        optimal_idx = list(K_range).index(optimal_k)
        ax.plot(optimal_k, inertias[optimal_idx], 'r*', markersize=20, 
                label=f'Optimal K = {optimal_k}')
        ax.legend()
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Elbow curve saved: {save_path}")
    
    plt.show()


def plot_silhouette_scores(K_range, silhouette_scores, optimal_k=None, save_path=None):
    """
    Plot silhouette scores for different K values.
    
    Parameters:
    -----------
    K_range : array-like
        Range of K values
    silhouette_scores : array-like
        Silhouette scores for each K
    optimal_k : int, optional
        Optimal K value to highlight
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(K_range, silhouette_scores, marker='s', linewidth=2, 
            markersize=8, color='#FF6B6B')
    
    if optimal_k:
        optimal_idx = list(K_range).index(optimal_k)
        ax.plot(optimal_k, silhouette_scores[optimal_idx], 'g*', 
                markersize=20, label=f'Optimal K = {optimal_k}')
        ax.legend()
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Analysis for Optimal K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Silhouette scores saved: {save_path}")
    
    plt.show()


def plot_clusters_2d(X, labels, centroids=None, method_name='Clustering Results', save_path=None):
    """
    Plot clustering results in 2D (using first 2 features or PCA).
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster assignments
    centroids : array-like, optional
        Cluster centroids
    method_name : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    # Use PCA if more than 2 features
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_plot = pca.fit_transform(X)
        centroids_plot = None
        if centroids is not None:
            centroids_plot = pca.transform(centroids)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
    else:
        X_plot = X
        centroids_plot = centroids
        xlabel = 'Feature 1'
        ylabel = 'Feature 2'
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # Plot clusters
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X_plot[mask, 0], 
            X_plot[mask, 1],
            c=colors[i % len(colors)],
            label=f'Cluster {label}',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=100
        )
    
    # Plot centroids if provided
    if centroids_plot is not None:
        ax.scatter(
            centroids_plot[:, 0], 
            centroids_plot[:, 1],
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(method_name, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster plot saved: {save_path}")
    
    plt.show()


def plot_dendrogram(linkage_matrix, method_name='Hierarchical Clustering', 
                    n_clusters=None, save_path=None):
    """
    Plot dendrogram for hierarchical clustering.
    
    Parameters:
    -----------
    linkage_matrix : array-like
        Linkage matrix from hierarchical clustering
    method_name : str
        Title for the plot
    n_clusters : int, optional
        Number of clusters to highlight
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot dendrogram
    dendro = dendrogram(
        linkage_matrix,
        ax=ax,
        color_threshold=0 if n_clusters is None else None,
        above_threshold_color='gray'
    )
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title(method_name, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dendrogram saved: {save_path}")
    
    plt.show()


def plot_knn_decision_boundary(X, y, model, feature_idx=(0, 1), 
                                target_names=None, save_path=None):
    """
    Plot KNN decision boundary (works best with 2 features).
    
    Parameters:
    -----------
    X : array-like
        Feature matrix (2D)
    y : array-like
        True labels
    model : sklearn model
        Trained KNN model
    feature_idx : tuple
        Indices of features to plot
    target_names : list, optional
        Class names
    save_path : str, optional
        Path to save the figure
    """
    # Use only 2 features for visualization
    X_2d = X[:, feature_idx]
    
    # Create mesh
    h = 0.02  # step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot data points
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    unique_classes = np.unique(y)
    
    for i in unique_classes:
        mask = y == i
        label = target_names[i] if target_names else f'Class {i}'
        ax.scatter(
            X_2d[mask, 0], 
            X_2d[mask, 1],
            c=colors[i % len(colors)],
            label=label,
            edgecolors='black',
            linewidth=0.5,
            s=100,
            alpha=0.8
        )
    
    ax.set_xlabel(f'Feature {feature_idx[0]}', fontsize=12)
    ax.set_ylabel(f'Feature {feature_idx[1]}', fontsize=12)
    ax.set_title('KNN Decision Boundary', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Decision boundary saved: {save_path}")
    
    plt.show()


def plot_k_vs_accuracy(k_values, train_scores, test_scores, optimal_k=None, save_path=None):
    """
    Plot KNN accuracy vs K value.
    
    Parameters:
    -----------
    k_values : array-like
        K values tested
    train_scores : array-like
        Training accuracies
    test_scores : array-like
        Test accuracies
    optimal_k : int, optional
        Optimal K to highlight
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, train_scores, marker='o', linewidth=2, 
            label='Training Accuracy', color='#4ECDC4')
    ax.plot(k_values, test_scores, marker='s', linewidth=2, 
            label='Test Accuracy', color='#FF6B6B')
    
    if optimal_k:
        optimal_idx = list(k_values).index(optimal_k)
        ax.axvline(x=optimal_k, color='green', linestyle='--', 
                   label=f'Optimal K = {optimal_k}', linewidth=2)
        ax.plot(optimal_k, test_scores[optimal_idx], 'g*', markersize=20)
    
    ax.set_xlabel('K (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('KNN: Accuracy vs K Value', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ K vs Accuracy plot saved: {save_path}")
    
    plt.show()


def plot_comparative_metrics(metrics_df, save_path=None):
    """
    Plot comparative bar chart of metrics across methods.
    
    Parameters:
    -----------
    metrics_df : DataFrame
        DataFrame with metrics for different methods
    save_path : str, optional
        Path to save the figure
    """
    # Select key metrics
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    available_metrics = [col for col in metric_cols if col in metrics_df.columns]
    
    if not available_metrics:
        print("⚠️  No classification metrics found for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        methods = metrics_df['Method'].values
        values = metrics_df[metric].values
        
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(methods, rotation=15, ha='right')
    
    # Hide extra subplots if less than 4 metrics
    for idx in range(len(available_metrics), 4):
        axes[idx].axis('off')
    
    plt.suptitle('Comparative Performance Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparative metrics saved: {save_path}")
    
    plt.show()


def plot_roc_curves(roc_results, save_path=None):
    """
    Plot ROC curves for multi-class classification (one-vs-rest).

    Parameters:
    -----------
    roc_results : dict
        Dictionary containing ROC data for each class
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    # Plot ROC curve for each class
    for i, color in enumerate(colors):
        if i not in roc_results:
            continue

        fpr = roc_results[i]['fpr']
        tpr = roc_results[i]['tpr']
        roc_auc = roc_results[i]['auc']
        class_name = roc_results[i]['class_name']

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - One-vs-Rest Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Add macro-average AUC if available
    if 'macro_auc' in roc_results:
        textstr = f'Macro-Average AUC: {roc_results["macro_auc"]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.6, 0.2, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved: {save_path}")

    plt.show()


def plot_learning_curves(learning_curve_results, save_path=None):
    """
    Plot learning curves showing performance vs training set size.

    Parameters:
    -----------
    learning_curve_results : dict
        Dictionary containing learning curve data
    save_path : str, optional
        Path to save the figure
    """
    train_sizes = learning_curve_results['train_sizes']
    train_mean = learning_curve_results['train_scores_mean']
    train_std = learning_curve_results['train_scores_std']
    test_mean = learning_curve_results['test_scores_mean']
    test_std = learning_curve_results['test_scores_std']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training scores
    ax.plot(train_sizes, train_mean, 'o-', color='#4ECDC4', linewidth=2,
            label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='#4ECDC4')

    # Plot validation scores
    ax.plot(train_sizes, test_mean, 'o-', color='#FF6B6B', linewidth=2,
            label='Cross-Validation Score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.2, color='#FF6B6B')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_title('Learning Curves - Model Performance vs Training Size', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Learning curves saved: {save_path}")

    plt.show()


def plot_feature_importance(importance_results, top_n=None, save_path=None):
    """
    Plot feature importance from permutation importance.

    Parameters:
    -----------
    importance_results : dict
        Dictionary containing feature importance data
    top_n : int, optional
        Show only top N features (default: all)
    save_path : str, optional
        Path to save the figure
    """
    importances_mean = importance_results['importances_mean']
    importances_std = importance_results['importances_std']
    feature_names = importance_results['feature_names']

    # Sort by importance
    sorted_idx = np.argsort(importances_mean)[::-1]

    # Limit to top N if specified
    if top_n is not None:
        sorted_idx = sorted_idx[:top_n]

    sorted_importances = importances_mean[sorted_idx]
    sorted_stds = importances_std[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_idx) * 0.4)))

    # Create horizontal bar chart
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_importances, xerr=sorted_stds,
            color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Permutation Importance (Decrease in Accuracy)', fontsize=12)
    ax.set_title('Feature Importance for KNN Classification', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance saved: {save_path}")

    plt.show()


def plot_silhouette_histogram(X, labels, save_path=None):
    """
    Plot silhouette histogram showing distribution of silhouette scores per sample.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster assignments
    save_path : str, optional
        Path to save the figure
    """
    from sklearn.metrics import silhouette_samples

    # Calculate silhouette scores for each sample
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = np.mean(silhouette_vals)

    n_clusters = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=(10, 7))

    y_lower = 10
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    for i in range(n_clusters):
        # Get silhouette scores for samples in this cluster
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i % len(colors)]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=12, fontweight='bold')

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster Label', fontsize=12)
    ax.set_title('Silhouette Analysis - Distribution per Cluster', fontsize=14, fontweight='bold')

    # The vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
               label=f'Average Score: {silhouette_avg:.3f}')

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xlim([-0.1, 1])
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Silhouette histogram saved: {save_path}")

    plt.show()


def plot_cluster_stability(ari_scores, save_path=None):
    """
    Plot cluster stability results from resampling analysis.

    Parameters:
    -----------
    ari_scores : list
        List of ARI scores from stability analysis
    save_path : str, optional
        Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: ARI scores over iterations
    ax1.plot(range(1, len(ari_scores) + 1), ari_scores,
             marker='o', linewidth=2, markersize=8, color='#4ECDC4')
    ax1.axhline(y=np.mean(ari_scores), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(ari_scores):.3f}')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Adjusted Rand Index', fontsize=12)
    ax1.set_title('Cluster Stability Across Resampling', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution of ARI scores
    ax2.hist(ari_scores, bins=15, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(ari_scores), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(ari_scores):.3f}')
    ax2.axvline(x=np.median(ari_scores), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(ari_scores):.3f}')
    ax2.set_xlabel('Adjusted Rand Index', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Stability Scores', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Cluster Stability Analysis (Mean ARI: {np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster stability plot saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Visualization Module for Wine Dataset Analysis")
    print("\nAvailable visualization functions:")
    print("  - plot_confusion_matrix()")
    print("  - plot_data_distribution_2d()")
    print("  - plot_pca_2d()")
    print("  - plot_elbow_curve()")
    print("  - plot_silhouette_scores()")
    print("  - plot_clusters_2d()")
    print("  - plot_dendrogram()")
    print("  - plot_knn_decision_boundary()")
    print("  - plot_k_vs_accuracy()")
    print("  - plot_comparative_metrics()")
    print("  - plot_roc_curves()")
    print("  - plot_learning_curves()")
    print("  - plot_feature_importance()")
    print("  - plot_silhouette_histogram()")
    print("  - plot_cluster_stability()")
    print("\nAll functions support saving with save_path parameter")