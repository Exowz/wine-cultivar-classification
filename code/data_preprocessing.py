"""
Data Loading and Preprocessing Module
Supports both sklearn Wine Dataset and Kaggle Wine Quality Dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')


class WineDataLoader:
    """
    Flexible Wine Dataset Loader
    Supports: sklearn Wine Recognition, Kaggle Wine Quality (Red/White)
    """
    
    def __init__(self, source='sklearn', wine_type='red', data_path='data/'):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        source : str
            'sklearn' or 'kaggle'
        wine_type : str
            'red' or 'white' (only for Kaggle dataset)
        data_path : str
            Path to data directory
        """
        self.source = source
        self.wine_type = wine_type
        self.data_path = data_path
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.data = None
        
    def load_data(self):
        """
        Load data based on specified source.
        
        Returns:
        --------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Feature names
        target_names : list
            Target class names
        """
        if self.source == 'sklearn':
            return self._load_sklearn_wine()
        elif self.source == 'kaggle':
            return self._load_kaggle_wine()
        else:
            raise ValueError(f"Unknown source: {self.source}. Use 'sklearn' or 'kaggle'")
    
    def _load_sklearn_wine(self):
        """
        Load sklearn Wine Recognition dataset.
        
        Dataset Info:
        - 178 samples
        - 13 features
        - 3 classes (wine cultivars)
        """
        print("\n" + "="*60)
        print("Loading sklearn Wine Recognition Dataset")
        print("="*60)
        
        wine_data = load_wine()
        self.X = wine_data.data
        self.y = wine_data.target
        self.feature_names = wine_data.feature_names
        self.target_names = wine_data.target_names
        
        # Create DataFrame for easier manipulation
        self.data = pd.DataFrame(self.X, columns=self.feature_names)
        self.data['target'] = self.y
        
        self._print_dataset_info()
        
        return self.X, self.y, self.feature_names, self.target_names
    
    def _load_kaggle_wine(self):
        """
        Load Kaggle Wine Quality dataset.
        
        Dataset Info:
        - Red Wine: ~1,599 samples
        - White Wine: ~4,898 samples
        - 11-12 features
        - Quality ratings: 3-9 (converted to 3 classes)
        """
        print("\n" + "="*60)
        print(f"Loading Kaggle Wine Quality Dataset ({self.wine_type.capitalize()} Wine)")
        print("="*60)
        
        # Define file paths
        if self.wine_type == 'red':
            filepath = os.path.join(self.data_path, 'winequality-red.csv')
        elif self.wine_type == 'white':
            filepath = os.path.join(self.data_path, 'winequality-white.csv')
        else:
            raise ValueError("wine_type must be 'red' or 'white'")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"\n⚠️  WARNING: Kaggle dataset not found at {filepath}")
            print("\nTo use Kaggle dataset:")
            print("1. Download from: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009")
            print(f"2. Place '{os.path.basename(filepath)}' in the '{self.data_path}' directory")
            print("\n📌 Falling back to sklearn Wine dataset...\n")
            return self._load_sklearn_wine()
        
        # Load CSV
        try:
            self.data = pd.read_csv(filepath, sep=';')
        except:
            self.data = pd.read_csv(filepath)
        
        # Separate features and target
        self.feature_names = [col for col in self.data.columns if col != 'quality']
        self.X = self.data[self.feature_names].values
        
        # Convert quality ratings to 3 classes
        # Low: 3-5, Medium: 6, High: 7-9
        quality = self.data['quality'].values
        self.y = np.digitize(quality, bins=[5.5, 6.5]) - 1  # Results in 0, 1, 2
        
        self.target_names = ['Low Quality (3-5)', 'Medium Quality (6)', 'High Quality (7-9)']
        
        self._print_dataset_info()
        
        return self.X, self.y, self.feature_names, self.target_names
    
    def _print_dataset_info(self):
        """Print dataset information."""
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Source: {self.source}")
        print(f"  Samples: {self.X.shape[0]}")
        print(f"  Features: {self.X.shape[1]}")
        print(f"  Classes: {len(np.unique(self.y))}")
        print(f"\nFeature Names:")
        for i, name in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {name}")
        print(f"\nTarget Classes:")
        for i, name in enumerate(self.target_names):
            count = np.sum(self.y == i)
            print(f"  Class {i}: {name:30s} ({count} samples)")
        print("="*60 + "\n")
    
    def get_dataset_summary(self):
        """
        Get statistical summary of the dataset.
        
        Returns:
        --------
        summary : DataFrame
            Statistical summary
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = self.data.describe()
        return summary
    
    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        
        Returns:
        --------
        missing_info : DataFrame
            Missing value information
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing = self.data.isnull().sum()
        missing_percent = 100 * missing / len(self.data)
        
        missing_info = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_percent
        })
        
        return missing_info[missing_info['Missing_Count'] > 0]
    
    def get_class_distribution(self):
        """
        Get class distribution.
        
        Returns:
        --------
        distribution : dict
            Class distribution
        """
        if self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        unique, counts = np.unique(self.y, return_counts=True)
        distribution = {
            self.target_names[i]: counts[i] 
            for i in range(len(unique))
        }
        
        return distribution


def load_wine_data(source='sklearn', wine_type='red', data_path='data/'):
    """
    Convenience function to load wine data.
    
    Parameters:
    -----------
    source : str
        'sklearn' or 'kaggle'
    wine_type : str
        'red' or 'white' (for Kaggle)
    data_path : str
        Path to data directory
    
    Returns:
    --------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        Feature names
    target_names : list
        Target class names
    """
    loader = WineDataLoader(source=source, wine_type=wine_type, data_path=data_path)
    return loader.load_data()


def main():
    """
    Test the data loading functionality.
    """
    print("\n" + "🍷"*30)
    print("Wine Dataset Loader - Test Mode")
    print("🍷"*30 + "\n")
    
    # Test sklearn dataset
    print("\n[TEST 1] Loading sklearn Wine Dataset...")
    loader_sklearn = WineDataLoader(source='sklearn')
    X_sk, y_sk, features_sk, targets_sk = loader_sklearn.load_data()
    
    print("\nDataset Summary (first 5 rows):")
    print(loader_sklearn.get_dataset_summary().iloc[:, :5])
    
    print("\nClass Distribution:")
    for class_name, count in loader_sklearn.get_class_distribution().items():
        print(f"  {class_name}: {count}")
    
    # Test Kaggle dataset (will fall back to sklearn if not available)
    print("\n[TEST 2] Attempting to load Kaggle Wine Dataset...")
    loader_kaggle = WineDataLoader(source='kaggle', wine_type='red')
    X_kg, y_kg, features_kg, targets_kg = loader_kaggle.load_data()
    
    print("\n✓ Data loading tests completed!")


if __name__ == "__main__":
    main()