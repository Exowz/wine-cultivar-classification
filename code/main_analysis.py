"""
Main Analysis Script
KNN, K-Means, and CAH Implementation on Wine Dataset
Complete Pipeline Orchestrator

Run this script to execute all analyses:
    python main_analysis.py
"""

import sys
import time
import argparse
from pathlib import Path

# Import analysis modules
from utils import create_directory_structure, format_execution_time
from data_preprocessing import load_wine_data
from knn_analysis import run_knn_analysis
from kmeans_analysis import run_kmeans_analysis
from cah_analysis import run_cah_analysis
from comparative_analysis import run_comparative_analysis


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          🍷 WINE DATASET ANALYSIS PROJECT 🍷                     ║
    ║                                                                  ║
    ║    KNN, K-Means, and CAH - Research & Implementation            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print analysis menu."""
    menu = """
    ┌────────────────────────────────────────────────────────────┐
    │                    ANALYSIS OPTIONS                         │
    ├────────────────────────────────────────────────────────────┤
    │  1. Run KNN Analysis Only                                   │
    │  2. Run K-Means Analysis Only                               │
    │  3. Run CAH Analysis Only                                   │
    │  4. Run All Methods + Comparative Analysis                  │
    │  5. Quick Demo (All methods, minimal output)                │
    │  0. Exit                                                    │
    └────────────────────────────────────────────────────────────┘
    """
    print(menu)


def run_single_knn(source='sklearn', wine_type='red'):
    """Run KNN analysis only."""
    print("\n" + "="*70)
    print(" RUNNING: K-NEAREST NEIGHBORS (KNN) ANALYSIS")
    print("="*70)
    
    start_time = time.time()
    analyzer, results = run_knn_analysis(source=source, wine_type=wine_type, save_results=True)
    elapsed = time.time() - start_time
    
    print(f"\n✅ KNN Analysis completed in {format_execution_time(elapsed)}")
    print(f"   Best K: {results['best_k']}")
    print(f"   Test Accuracy: {results['metrics']['accuracy']:.4f}")


def run_single_kmeans(source='sklearn', wine_type='red'):
    """Run K-Means analysis only."""
    print("\n" + "="*70)
    print(" RUNNING: K-MEANS CLUSTERING ANALYSIS")
    print("="*70)
    
    start_time = time.time()
    analyzer, results = run_kmeans_analysis(source=source, wine_type=wine_type, save_results=True)
    elapsed = time.time() - start_time
    
    print(f"\n✅ K-Means Analysis completed in {format_execution_time(elapsed)}")
    print(f"   Optimal K: {results['optimal_k']}")
    print(f"   Silhouette Score: {results['metrics']['clustering_metrics']['silhouette_score']:.4f}")


def run_single_cah(source='sklearn', wine_type='red'):
    """Run CAH analysis only."""
    print("\n" + "="*70)
    print(" RUNNING: HIERARCHICAL CLUSTERING (CAH) ANALYSIS")
    print("="*70)
    
    start_time = time.time()
    analyzer, results = run_cah_analysis(
        source=source, 
        wine_type=wine_type, 
        n_clusters=3,
        linkage='ward',
        compare_methods=True,
        save_results=True
    )
    elapsed = time.time() - start_time
    
    print(f"\n✅ CAH Analysis completed in {format_execution_time(elapsed)}")
    print(f"   Silhouette Score: {results['metrics']['clustering_metrics']['silhouette_score']:.4f}")


def run_complete_analysis(source='sklearn', wine_type='red'):
    """Run all analyses and comparative analysis."""
    print("\n" + "="*70)
    print(" RUNNING: COMPLETE ANALYSIS (ALL METHODS + COMPARISON)")
    print("="*70)
    
    start_time = time.time()
    
    # Run comparative analysis (which runs all methods)
    analyzer, report = run_comparative_analysis(source=source, wine_type=wine_type, save_results=True)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Complete Analysis finished in {format_execution_time(elapsed)}")
    print("\n📊 Summary:")
    print(report[['Method', 'Accuracy', 'F1-Score', 'Silhouette']].to_string(index=False))


def run_quick_demo(source='sklearn', wine_type='red'):
    """Run quick demo with all methods."""
    print("\n" + "="*70)
    print(" RUNNING: QUICK DEMO MODE")
    print("="*70)
    print("\nThis will run all three methods with minimal output...")
    
    start_time = time.time()
    
    # Load data once
    X, y, feature_names, target_names = load_wine_data(source=source, wine_type=wine_type)
    
    print("\n[1/3] KNN...", end=' ')
    sys.stdout.flush()
    analyzer_knn, _ = run_knn_analysis(source=source, wine_type=wine_type, save_results=False)
    print("✓")
    
    print("[2/3] K-Means...", end=' ')
    sys.stdout.flush()
    analyzer_kmeans, _ = run_kmeans_analysis(source=source, wine_type=wine_type, save_results=False)
    print("✓")
    
    print("[3/3] CAH...", end=' ')
    sys.stdout.flush()
    analyzer_cah, _ = run_cah_analysis(
        source=source, 
        wine_type=wine_type,
        n_clusters=3,
        compare_methods=False,
        save_results=False
    )
    print("✓")
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Quick demo completed in {format_execution_time(elapsed)}")


def interactive_mode():
    """Run in interactive menu mode."""
    print_banner()
    create_directory_structure()
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    
    while True:
        print_menu()
        
        try:
            choice = input("    Enter your choice (0-5): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye! Thank you for using the Wine Analysis tool.")
                break
            elif choice == '1':
                run_single_knn()
            elif choice == '2':
                run_single_kmeans()
            elif choice == '3':
                run_single_cah()
            elif choice == '4':
                run_complete_analysis()
            elif choice == '5':
                run_quick_demo()
            else:
                print("\n❌ Invalid choice. Please select 0-5.")
                continue
            
            input("\n\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Analysis interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again.")
            input("\nPress Enter to continue...")


def command_line_mode(args):
    """Run in command line mode with arguments."""
    print_banner()
    create_directory_structure()
    
    # Determine source and wine type
    source = args.source if hasattr(args, 'source') else 'sklearn'
    wine_type = args.wine_type if hasattr(args, 'wine_type') else 'red'
    
    if args.method == 'knn':
        run_single_knn(source=source, wine_type=wine_type)
    elif args.method == 'kmeans':
        run_single_kmeans(source=source, wine_type=wine_type)
    elif args.method == 'cah':
        run_single_cah(source=source, wine_type=wine_type)
    elif args.method == 'all':
        run_complete_analysis(source=source, wine_type=wine_type)
    elif args.method == 'demo':
        run_quick_demo(source=source, wine_type=wine_type)


def main():
    """
    Main entry point for the analysis.
    """
    parser = argparse.ArgumentParser(
        description='Wine Dataset Analysis: KNN, K-Means, and CAH Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (menu-driven)
  python main_analysis.py

  # Run all methods with comparative analysis
  python main_analysis.py --method all

  # Run specific method
  python main_analysis.py --method knn
  python main_analysis.py --method kmeans
  python main_analysis.py --method cah

  # Quick demo
  python main_analysis.py --method demo

  # Use Kaggle dataset (if available)
  python main_analysis.py --method all --source kaggle --wine-type red

For more information, see README.md
        """
    )
    
    parser.add_argument(
        '--method',
        choices=['knn', 'kmeans', 'cah', 'all', 'demo'],
        help='Analysis method to run'
    )
    
    parser.add_argument(
        '--source',
        choices=['sklearn', 'kaggle'],
        default='sklearn',
        help='Dataset source (default: sklearn)'
    )
    
    parser.add_argument(
        '--wine-type',
        choices=['red', 'white'],
        default='red',
        help='Wine type for Kaggle dataset (default: red)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.method:
            # Command line mode
            command_line_mode(args)
        else:
            # Interactive mode
            interactive_mode()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()