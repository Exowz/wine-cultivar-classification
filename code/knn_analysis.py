"""
K-Nearest Neighbors (KNN) Analysis Module
Implementation and evaluation of KNN classifier on Wine dataset
"""

import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, learning_curve
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

# Import custom modules
from data_preprocessing import load_wine_data
from utils import (
    prepare_data, calculate_classification_metrics,
    save_metrics_to_csv, print_metrics, format_execution_time,
    get_figure_path
)
from visualizations import (
    plot_confusion_matrix, plot_k_vs_accuracy,
    plot_pca_2d, plot_knn_decision_boundary,
    plot_roc_curves, plot_learning_curves, plot_feature_importance
)


class KNNAnalyzer:
    """
    KNN Analysis Class for Wine Dataset
    """
    
    def __init__(self, X, y, feature_names, target_names):
        """
        Initialize KNN Analyzer.
        
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
        self.model = None
        self.best_k = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
    
    def prepare_data(self, test_size=0.3, random_state=42):
        """
        Prepare data: split and scale.
        
        Parameters:
        -----------
        test_size : float
            Test set proportion
        random_state : int
            Random seed
        """
        print("\n" + "="*60)
        print("Data Preparation for KNN")
        print("="*60)
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = prepare_data(
            self.X, self.y, test_size=test_size, random_state=random_state, scale=True
        )
    
    def find_optimal_k(self, k_range=range(1, 31), cv=5):
        """
        Find optimal K using cross-validation.
        
        Parameters:
        -----------
        k_range : range or list
            Range of K values to test
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        best_k : int
            Optimal K value
        results : dict
            Dictionary with K values and scores
        """
        print("\n" + "="*60)
        print("Finding Optimal K for KNN")
        print("="*60)
        
        train_scores = []
        test_scores = []
        cv_scores = []
        
        start_time = time.time()
        
        for k in k_range:
            # Train model
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            
            # Calculate scores
            train_score = knn.score(self.X_train, self.y_train)
            test_score = knn.score(self.X_test, self.y_test)
            cv_score = cross_val_score(knn, self.X_train, self.y_train, cv=cv).mean()
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            cv_scores.append(cv_score)
            
            print(f"K={k:2d} | Train: {train_score:.4f} | Test: {test_score:.4f} | CV: {cv_score:.4f}")
        
        # Find best K based on CV score
        best_idx = np.argmax(cv_scores)
        self.best_k = list(k_range)[best_idx]
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Optimal K found: {self.best_k}")
        print(f"  Best CV Score: {cv_scores[best_idx]:.4f}")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")
        
        results = {
            'k_values': list(k_range),
            'train_scores': train_scores,
            'test_scores': test_scores,
            'cv_scores': cv_scores,
            'best_k': self.best_k
        }
        
        return self.best_k, results
    
    def train_model(self, k=None, n_neighbors=None, weights='uniform', metric='minkowski', p=2):
        """
        Train KNN model with specified parameters.
        
        Parameters:
        -----------
        k : int, optional
            Number of neighbors (alternative to n_neighbors)
        n_neighbors : int, optional
            Number of neighbors
        weights : str
            Weight function ('uniform' or 'distance')
        metric : str
            Distance metric
        p : int
            Power parameter for Minkowski metric
        
        Returns:
        --------
        model : KNeighborsClassifier
            Trained model
        """
        # Use k if provided, otherwise n_neighbors, otherwise best_k
        if k is not None:
            n_neighbors = k
        elif n_neighbors is None:
            n_neighbors = self.best_k if self.best_k else 5
        
        print("\n" + "="*60)
        print("Training KNN Model")
        print("="*60)
        print(f"Parameters:")
        print(f"  n_neighbors: {n_neighbors}")
        print(f"  weights: {weights}")
        print(f"  metric: {metric}")
        print(f"  p: {p}")
        
        start_time = time.time()
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Model trained successfully!")
        print(f"  Training time: {format_execution_time(elapsed_time)}")
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate trained model on train and test sets.
        
        Returns:
        --------
        results : dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("\n" + "="*60)
        print("KNN Model Evaluation")
        print("="*60)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_metrics = calculate_classification_metrics(self.y_train, y_train_pred)
        test_metrics = calculate_classification_metrics(self.y_test, y_test_pred)
        
        # Print results
        print("\n[Training Set Performance]")
        print_metrics(train_metrics, "Training Metrics")
        
        print("\n[Test Set Performance]")
        print_metrics(test_metrics, "Test Metrics")
        
        # Detailed classification report
        print("\n[Detailed Classification Report - Test Set]")
        print(classification_report(
            self.y_test, y_test_pred,
            target_names=self.target_names,
            digits=4
        ))
        
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        return results
    
    def grid_search_hyperparameters(self):
        """
        Perform grid search for optimal hyperparameters.

        Returns:
        --------
        best_params : dict
            Best parameters found
        """
        print("\n" + "="*60)
        print("Grid Search for Optimal Hyperparameters")
        print("="*60)

        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }

        start_time = time.time()

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        elapsed_time = time.time() - start_time

        print(f"\n✓ Grid search completed!")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")
        print(f"\nBest Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV Score: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        self.best_k = grid_search.best_params_['n_neighbors']

        return grid_search.best_params_

    def perform_cross_validation(self, cv=10):
        """
        Perform comprehensive cross-validation analysis.

        Parameters:
        -----------
        cv : int
            Number of cross-validation folds (default: 10)

        Returns:
        --------
        cv_results : dict
            Cross-validation results including mean and std
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("\n" + "="*60)
        print(f"Performing {cv}-Fold Cross-Validation")
        print("="*60)

        start_time = time.time()

        # Perform cross-validation with multiple metrics
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = cross_validate(
            self.model,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )

        elapsed_time = time.time() - start_time

        # Calculate statistics
        results = {}
        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            results[metric] = {
                'test_mean': np.mean(cv_results[test_key]),
                'test_std': np.std(cv_results[test_key]),
                'train_mean': np.mean(cv_results[train_key]),
                'train_std': np.std(cv_results[train_key])
            }

        print(f"\n{cv}-Fold Cross-Validation Results:")
        print(f"{'Metric':<20} {'Test Mean':<15} {'Test Std':<15} {'Train Mean':<15} {'Train Std':<15}")
        print("-" * 80)
        for metric, values in results.items():
            print(f"{metric:<20} {values['test_mean']:.4f} ± {values['test_std']:<10.4f} "
                  f"{values['train_mean']:.4f} ± {values['train_std']:<10.4f}")

        print(f"\n✓ Cross-validation completed!")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")

        return results

    def compute_roc_curves(self):
        """
        Compute ROC curves and AUC for each class (one-vs-rest).

        Returns:
        --------
        roc_results : dict
            ROC curve data and AUC scores for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("\n" + "="*60)
        print("Computing ROC Curves and AUC")
        print("="*60)

        # Get predicted probabilities
        y_proba = self.model.predict_proba(self.X_test)

        # Binarize labels for one-vs-rest
        n_classes = len(self.target_names)
        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))

        # Compute ROC curve and AUC for each class
        roc_results = {}

        for i in range(n_classes):
            fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            roc_results[i] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc,
                'class_name': self.target_names[i]
            }

            print(f"  {self.target_names[i]:<15} AUC: {roc_auc:.4f}")

        # Compute macro-average AUC
        macro_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        print(f"\n  Macro-Average AUC: {macro_auc:.4f}")

        roc_results['macro_auc'] = macro_auc

        print("\n✓ ROC curves computed!")

        return roc_results

    def compute_learning_curves(self, train_sizes=np.linspace(0.1, 1.0, 10), cv=5):
        """
        Compute learning curves to show model performance vs training set size.

        Parameters:
        -----------
        train_sizes : array-like
            Relative or absolute numbers of training examples
        cv : int
            Number of cross-validation folds

        Returns:
        --------
        learning_curve_results : dict
            Learning curve data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("\n" + "="*60)
        print("Computing Learning Curves")
        print("="*60)

        start_time = time.time()

        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.model,
            self.X_train,
            self.y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        elapsed_time = time.time() - start_time

        # Calculate means and stds
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        print(f"\nLearning Curve Results:")
        print(f"{'Train Size':<15} {'Train Acc':<20} {'Val Acc':<20}")
        print("-" * 60)
        for size, train_mean, train_std, test_mean, test_std in zip(
            train_sizes_abs, train_scores_mean, train_scores_std,
            test_scores_mean, test_scores_std
        ):
            print(f"{size:<15} {train_mean:.4f} ± {train_std:<10.4f} "
                  f"{test_mean:.4f} ± {test_std:<10.4f}")

        print(f"\n✓ Learning curves computed!")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")

        results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores_mean,
            'train_scores_std': train_scores_std,
            'test_scores_mean': test_scores_mean,
            'test_scores_std': test_scores_std
        }

        return results

    def compute_feature_importance(self, n_repeats=10):
        """
        Compute feature importance using permutation importance.

        Parameters:
        -----------
        n_repeats : int
            Number of times to permute each feature

        Returns:
        --------
        importance_results : dict
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("\n" + "="*60)
        print("Computing Feature Importance (Permutation)")
        print("="*60)

        start_time = time.time()

        # Compute permutation importance
        perm_importance = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        elapsed_time = time.time() - start_time

        # Sort features by importance
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]

        print(f"\nFeature Importance Ranking:")
        print(f"{'Rank':<6} {'Feature':<35} {'Importance':<15} {'Std':<10}")
        print("-" * 70)

        for rank, idx in enumerate(sorted_idx, 1):
            feature_name = self.feature_names[idx]
            importance_mean = perm_importance.importances_mean[idx]
            importance_std = perm_importance.importances_std[idx]
            print(f"{rank:<6} {feature_name:<35} {importance_mean:.4f} ± {importance_std:<10.4f}")

        print(f"\n✓ Feature importance computed!")
        print(f"  Time elapsed: {format_execution_time(elapsed_time)}")

        results = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'sorted_indices': sorted_idx,
            'feature_names': self.feature_names
        }

        return results
    
    def visualize_results(self, y_test_pred, optimization_results=None, save_figures=True,
                         roc_results=None, learning_curve_results=None, feature_importance_results=None):
        """
        Generate visualizations for KNN results.

        Parameters:
        -----------
        y_test_pred : array-like
            Test set predictions
        optimization_results : dict, optional
            Results from K optimization
        save_figures : bool
            Whether to save figures
        roc_results : dict, optional
            ROC curve results
        learning_curve_results : dict, optional
            Learning curve results
        feature_importance_results : dict, optional
            Feature importance results
        """
        print("\n" + "="*60)
        print("Generating KNN Visualizations")
        print("="*60)

        # 1. K vs Accuracy (if optimization results available)
        if optimization_results:
            save_path = get_figure_path('knn/k_vs_accuracy.png') if save_figures else None
            plot_k_vs_accuracy(
                optimization_results['k_values'],
                optimization_results['train_scores'],
                optimization_results['test_scores'],
                optimal_k=optimization_results['best_k'],
                save_path=save_path
            )

        # 2. Confusion Matrix - Shows prediction accuracy
        save_path = get_figure_path('knn/confusion_matrix.png') if save_figures else None
        plot_confusion_matrix(
            self.y_test, y_test_pred,
            self.target_names,
            'KNN Classifier',
            save_path=save_path
        )

        # 3. PCA Visualization with predictions
        save_path = get_figure_path('knn/pca_predictions.png') if save_figures else None
        plot_pca_2d(
            self.X_test, y_test_pred,
            self.target_names,
            'KNN Predictions - PCA Visualization',
            save_path=save_path
        )

        # 4. True labels visualization (for comparison)
        save_path = get_figure_path('knn/pca_true_labels.png') if save_figures else None
        plot_pca_2d(
            self.X_test, self.y_test,
            self.target_names,
            'True Labels - PCA Visualization (Ground Truth)',
            save_path=save_path
        )

        # 5. Per-class performance analysis
        save_path = get_figure_path('knn/per_class_performance.png') if save_figures else None
        self._plot_per_class_performance(y_test_pred, save_path)

        # 6. Prediction confidence analysis
        save_path = get_figure_path('knn/prediction_confidence.png') if save_figures else None
        self._plot_prediction_confidence(save_path)

        # 7. ROC Curves (if available)
        if roc_results:
            save_path = get_figure_path('knn/roc_curves.png') if save_figures else None
            plot_roc_curves(roc_results, save_path=save_path)

        # 8. Learning Curves (if available)
        if learning_curve_results:
            save_path = get_figure_path('knn/learning_curves.png') if save_figures else None
            plot_learning_curves(learning_curve_results, save_path=save_path)

        # 9. Feature Importance (if available)
        if feature_importance_results:
            save_path = get_figure_path('knn/feature_importance.png') if save_figures else None
            plot_feature_importance(feature_importance_results, top_n=10, save_path=save_path)

        print("\n✓ All visualizations generated!")
    
    def _plot_per_class_performance(self, y_test_pred, save_path=None):
        """
        Plot detailed per-class performance metrics.
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        n_classes = len(self.target_names)
        
        # Calculate per-class metrics
        precisions = precision_score(self.y_test, y_test_pred, average=None, zero_division=0)
        recalls = recall_score(self.y_test, y_test_pred, average=None, zero_division=0)
        f1_scores = f1_score(self.y_test, y_test_pred, average=None, zero_division=0)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_classes)
        width = 0.25
        
        bars1 = ax.bar(x - width, precisions, width, label='Precision', 
                      color='#FF6B6B', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, recalls, width, label='Recall', 
                      color='#4ECDC4', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', 
                      color='#45B7D1', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Wine Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics\n(How well KNN performs on each wine type)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.target_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Per-class performance saved: {save_path}")
        
        plt.show()
    
    def _plot_prediction_confidence(self, save_path=None):
        """
        Plot prediction confidence (probability distributions).
        """
        import matplotlib.pyplot as plt
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(self.X_test)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for class_idx in range(len(self.target_names)):
                ax = axes[class_idx]
                
                # Get probabilities for this class
                class_probas = y_proba[:, class_idx]
                
                # Separate by correct vs incorrect predictions
                y_pred = self.model.predict(self.X_test)
                correct_mask = (y_pred == self.y_test) & (self.y_test == class_idx)
                incorrect_mask = (y_pred != self.y_test) & (self.y_test == class_idx)
                
                # Plot histograms
                if np.sum(correct_mask) > 0:
                    ax.hist(class_probas[correct_mask], bins=20, alpha=0.6, 
                           label='Correct Predictions', color='green', edgecolor='black')
                
                if np.sum(incorrect_mask) > 0:
                    ax.hist(class_probas[incorrect_mask], bins=20, alpha=0.6, 
                           label='Incorrect Predictions', color='red', edgecolor='black')
                
                ax.set_xlabel('Prediction Probability', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{self.target_names[class_idx]}\nPrediction Confidence', 
                           fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xlim([0, 1])
            
            plt.suptitle('KNN Prediction Confidence Analysis\n(Higher probability = More confident prediction)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Prediction confidence saved: {save_path}")
            
            plt.show()
        else:
            print("  (Prediction confidence visualization requires predict_proba)")


def run_knn_analysis(source='sklearn', wine_type='red', save_results=True, enhanced=True):
    """
    Complete KNN analysis pipeline.

    Parameters:
    -----------
    source : str
        Dataset source ('sklearn' or 'kaggle')
    wine_type : str
        Wine type for Kaggle dataset ('red' or 'white')
    save_results : bool
        Whether to save results
    enhanced : bool
        Whether to run enhanced analyses (CV, ROC, learning curves, feature importance)

    Returns:
    --------
    analyzer : KNNAnalyzer
        Trained KNN analyzer object
    results : dict
        Analysis results
    """
    print("\n" + "🍷"*30)
    print("K-NEAREST NEIGHBORS (KNN) ANALYSIS")
    print("🍷"*30)

    # Load data
    X, y, feature_names, target_names = load_wine_data(source=source, wine_type=wine_type)

    # Initialize analyzer
    analyzer = KNNAnalyzer(X, y, feature_names, target_names)

    # Prepare data
    analyzer.prepare_data()

    # Find optimal K
    best_k, optimization_results = analyzer.find_optimal_k(k_range=range(1, 31))

    # Train model with optimal K
    analyzer.train_model(k=best_k)

    # Evaluate model
    eval_results = analyzer.evaluate_model()

    # Enhanced analyses
    cv_results = None
    roc_results = None
    learning_curve_results = None
    feature_importance_results = None

    if enhanced:
        # Perform 10-fold cross-validation
        cv_results = analyzer.perform_cross_validation(cv=10)

        # Compute ROC curves and AUC
        roc_results = analyzer.compute_roc_curves()

        # Compute learning curves
        learning_curve_results = analyzer.compute_learning_curves()

        # Compute feature importance
        feature_importance_results = analyzer.compute_feature_importance()

    # Visualize results
    analyzer.visualize_results(
        eval_results['y_test_pred'],
        optimization_results,
        save_figures=save_results,
        roc_results=roc_results,
        learning_curve_results=learning_curve_results,
        feature_importance_results=feature_importance_results
    )

    # Save metrics
    if save_results:
        save_metrics_to_csv(
            eval_results['test_metrics'],
            'results_summary.csv',
            'KNN'
        )

    print("\n" + "="*60)
    print("✓ KNN Analysis Complete!")
    print("="*60)

    results = {
        'model': analyzer.model,
        'best_k': best_k,
        'metrics': eval_results['test_metrics'],
        'optimization_results': optimization_results,
        'cv_results': cv_results,
        'roc_results': roc_results,
        'learning_curve_results': learning_curve_results,
        'feature_importance_results': feature_importance_results
    }

    return analyzer, results


if __name__ == "__main__":
    # Run complete KNN analysis
    analyzer, results = run_knn_analysis(source='sklearn', save_results=True)
    
    print("\n✓ KNN analysis completed successfully!")
    print(f"  Best K: {results['best_k']}")
    print(f"  Test Accuracy: {results['metrics']['accuracy']:.4f}")