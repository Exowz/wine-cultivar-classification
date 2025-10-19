"""
Utility Functions for Wine Dataset Analysis
KNN, K-Means, and CAH Implementation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
import warnings
warnings.filterwarnings('ignore')


def create_directory_structure():
    """
    Create the necessary directory structure for the project.
    """
    # Get the parent directory of 'code' folder
    code_dir = Path(__file__).parent
    project_root = code_dir.parent

    directories = [
        project_root / 'data',
        project_root / 'results',
        project_root / 'results' / 'figures',
        project_root / 'results' / 'figures' / 'knn',
        project_root / 'results' / 'figures' / 'kmeans',
        project_root / 'results' / 'figures' / 'cah',
        project_root / 'results' / 'figures' / 'comparative',
        project_root / 'results' / 'metrics'
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"✓ Directory structure created successfully at: {project_root}")


def prepare_data(X, y, test_size=0.3, random_state=42, scale=True):
    """
    Prepare data: split into train/test and optionally scale.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility
    scale : bool
        Whether to scale features
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split and optionally scaled data
    scaler : StandardScaler or None
        Fitted scaler object if scale=True
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    print(f"✓ Data prepared: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler


def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate comprehensive classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str
        Averaging method for multi-class metrics
    
    Returns:
    --------
    metrics : dict
        Dictionary of calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def calculate_clustering_metrics(X, labels, true_labels=None):
    """
    Calculate clustering quality metrics.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster assignments
    true_labels : array-like, optional
        True labels for comparison
    
    Returns:
    --------
    metrics : dict
        Dictionary of clustering metrics
    """
    metrics = {}
    
    # Internal metrics (don't need true labels)
    try:
        metrics['silhouette_score'] = silhouette_score(X, labels)
    except:
        metrics['silhouette_score'] = None
    
    try:
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    except:
        metrics['davies_bouldin_score'] = None
    
    try:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    except:
        metrics['calinski_harabasz_score'] = None
    
    # External metrics (need true labels)
    if true_labels is not None:
        metrics['purity'] = calculate_purity(true_labels, labels)
        metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)

    return metrics


def calculate_purity(true_labels, cluster_labels):
    """
    Calculate cluster purity score.
    
    Parameters:
    -----------
    true_labels : array-like
        True class labels
    cluster_labels : array-like
        Cluster assignments
    
    Returns:
    --------
    purity : float
        Purity score
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, cluster_labels)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    
    return purity


def get_project_root():
    """
    Get the project root directory (parent of 'code' folder).

    Returns:
    --------
    project_root : Path
        Path to project root directory
    """
    code_dir = Path(__file__).parent
    return code_dir.parent


def get_figure_path(relative_path):
    """
    Get absolute path for saving figures in the results directory.

    Parameters:
    -----------
    relative_path : str
        Relative path like 'knn/confusion_matrix.png'

    Returns:
    --------
    absolute_path : str
        Absolute path to save the figure
    """
    project_root = get_project_root()
    full_path = project_root / 'results' / 'figures' / relative_path

    # Ensure directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return str(full_path)


def save_metrics_to_csv(metrics_dict, filename, method_name):
    """
    Save metrics to CSV file.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics
    filename : str
        Output filename
    method_name : str
        Name of the method (KNN, K-Means, CAH)
    """
    df = pd.DataFrame([metrics_dict])
    df.insert(0, 'Method', method_name)

    project_root = get_project_root()
    filepath = project_root / 'results' / 'metrics' / filename

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing file or create new
    if filepath.exists():
        existing_df = pd.read_csv(filepath)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(filepath, index=False)
    print(f"✓ Metrics saved to {filepath}")


def print_metrics(metrics_dict, title="Metrics"):
    """
    Pretty print metrics dictionary.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics
    title : str
        Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in metrics_dict.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{key:.<40} {value:.4f}")
            else:
                print(f"{key:.<40} {value}")
        else:
            print(f"{key:.<40} N/A")
    
    print(f"{'='*60}\n")


def get_feature_names(dataset_type='sklearn'):
    """
    Get feature names based on dataset type.
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset ('sklearn' or 'kaggle')
    
    Returns:
    --------
    feature_names : list
        List of feature names
    """
    if dataset_type == 'sklearn':
        return [
            'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
            'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue',
            'od280/od315_of_diluted_wines', 'proline'
        ]
    elif dataset_type == 'kaggle':
        return [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
    else:
        return None


def format_execution_time(seconds):
    """
    Format execution time in human-readable format.

    Parameters:
    -----------
    seconds : float
        Time in seconds

    Returns:
    --------
    formatted_time : str
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} min {secs:.2f} sec"


def calculate_cluster_stability(X, clustering_model, n_iterations=10, subsample_ratio=0.8):
    """
    Calculate cluster stability using resampling.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    clustering_model : sklearn clustering model
        Clustering model to evaluate
    n_iterations : int
        Number of resampling iterations
    subsample_ratio : float
        Ratio of data to sample in each iteration

    Returns:
    --------
    stability_score : float
        Mean ARI across iterations
    ari_scores : list
        List of ARI scores from each iteration
    """
    from sklearn.utils import resample

    n_samples = int(len(X) * subsample_ratio)
    ari_scores = []

    # Get baseline clustering on full data
    baseline_labels = clustering_model.fit_predict(X)

    for i in range(n_iterations):
        # Resample data
        indices = resample(range(len(X)), n_samples=n_samples, replace=False, random_state=i)
        X_resampled = X[indices]

        # Cluster resampled data
        resampled_labels = clustering_model.fit_predict(X_resampled)

        # Compare with baseline (only for resampled indices)
        baseline_resampled = baseline_labels[indices]
        ari = adjusted_rand_score(baseline_resampled, resampled_labels)
        ari_scores.append(ari)

    stability_score = np.mean(ari_scores)

    return stability_score, ari_scores


if __name__ == "__main__":
    print("Utility functions module for Wine Dataset Analysis")
    print("This module provides helper functions for:")
    print("  - Directory structure creation")
    print("  - Data preparation and splitting")
    print("  - Metrics calculation (classification & clustering)")
    print("  - Results saving and formatting")
    create_directory_structure()