# 🐍 Conda Quick Start Guide

## For Conda Users

This guide is specifically for setting up and running the Wine Analysis project with Conda.

---

## 📦 Step-by-Step Setup

### 1. Create and Activate Environment

```bash
# Navigate to the project root directory
cd DELIVERY-1-PROJECT-KMEANS-CAH-KNN-Research&Implementation

# Create the conda environment
conda env create -f environment.yml

# This creates an environment named 'wine-analysis' with all dependencies
```

### 2. Activate the Environment

```bash
# Activate the environment
conda activate wine-analysis

# You should see (wine-analysis) in your terminal prompt
```

### 3. Navigate to Code Directory

```bash
# Move to the code directory
cd code
```

---

## 🚀 Running the Analysis

### Option 1: Interactive Menu (Easiest)

```bash
python main_analysis.py
```

This will show you a menu:
```
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
```

**Recommended**: Choose **option 4** to run everything!

### Option 2: Command Line

```bash
# Run complete analysis (KNN + K-Means + CAH + Comparison)
python main_analysis.py --method all

# Run individual methods
python main_analysis.py --method knn
python main_analysis.py --method kmeans
python main_analysis.py --method cah

# Quick demo (faster, less output)
python main_analysis.py --method demo
```

### Option 3: Run Individual Scripts

```bash
# Run each analysis separately
python knn_analysis.py
python kmeans_analysis.py
python cah_analysis.py
python comparative_analysis.py
```

---

## 📊 What Happens When You Run

The scripts will:

1. ✅ Create necessary directories (`results/`, `data/`, etc.)
2. ✅ Load the Wine dataset (sklearn built-in)
3. ✅ Run the selected analysis/analyses
4. ✅ Generate visualizations and save to `results/figures/`
5. ✅ Save metrics to `results/metrics/`
6. ✅ Display results in the terminal

---

## 📁 Output Files

After running, you'll find:

```
results/
├── figures/
│   ├── knn/
│   │   ├── confusion_matrix.png
│   │   ├── pca_visualization.png
│   │   └── k_vs_accuracy.png
│   ├── kmeans/
│   │   ├── elbow_curve.png
│   │   ├── silhouette_scores.png
│   │   └── clusters_pca.png
│   ├── cah/
│   │   ├── dendrogram.png
│   │   └── clusters_pca.png
│   └── comparative/
│       ├── metrics_comparison.png
│       ├── execution_time.png
│       └── methods_comparison_pca.png
└── metrics/
    ├── results_summary.csv
    └── comparative_report.csv
```

---

## 🎯 Recommended Workflow for Your Project

### For Quick Testing:
```bash
conda activate wine-analysis
cd code
python main_analysis.py --method demo
```
⏱️ Takes ~30 seconds, gives you a quick overview

### For Complete Analysis (Recommended):
```bash
conda activate wine-analysis
cd code
python main_analysis.py --method all
```
⏱️ Takes ~2-3 minutes, generates all visualizations and reports

### For Presentation Preparation:
```bash
conda activate wine-analysis
cd code
python main_analysis.py  # Interactive mode
# Then choose option 4
```
⏱️ Takes ~2-3 minutes, with detailed output you can reference

---

## 🔧 Conda-Specific Commands

### Check Your Environment

```bash
# List all conda environments
conda env list

# Check installed packages in current environment
conda list

# Verify you're in the right environment
conda info --envs
```

### Update Environment

```bash
# If you need to add more packages
conda install -n wine-analysis package_name

# Or update a specific package
conda update -n wine-analysis scikit-learn
```

### Export Your Environment

```bash
# Export for sharing with others
conda env export > environment_export.yml

# Or export with only manually installed packages
conda env export --from-history > environment_minimal.yml
```

### Remove Environment (if needed)

```bash
# Deactivate first
conda deactivate

# Remove the environment
conda env remove -n wine-analysis
```

---

## 💡 Pro Tips for Conda Users

### 1. Always Activate Before Running
```bash
# Always make sure you see (wine-analysis) in your prompt
conda activate wine-analysis
```

### 2. Jupyter Notebook Support (Optional)
If you want to run the code in Jupyter:

```bash
conda activate wine-analysis
# Install ipykernel (already in environment.yml)
python -m ipykernel install --user --name=wine-analysis

# Launch Jupyter
jupyter notebook
# Then select 'wine-analysis' kernel
```

### 3. VS Code Integration
If using VS Code:
1. Install Python extension
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "Python: Select Interpreter"
4. Choose the `wine-analysis` conda environment

### 4. Quick Verification
```bash
conda activate wine-analysis
python -c "import sklearn, pandas, matplotlib; print('✅ All packages imported successfully!')"
```

---

## 🐛 Common Conda Issues

### Issue: "CondaError: Run 'conda init' first"
```bash
# Initialize conda for your shell
conda init bash  # or zsh, fish, etc.
# Then restart your terminal
```

### Issue: "Environment already exists"
```bash
# Remove and recreate
conda env remove -n wine-analysis
conda env create -f environment.yml
```

### Issue: "Solving environment" takes too long
```bash
# Use mamba (faster conda alternative)
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
```

### Issue: Packages conflict
```bash
# Try creating with specific Python version
conda create -n wine-analysis python=3.10
conda activate wine-analysis
pip install -r requirements.txt
```

---

## 📚 Additional Conda Resources

- **Conda Cheat Sheet**: https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
- **Managing Environments**: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- **Conda vs Pip**: Both are included! Use conda for major packages, pip for specific ones

---

## ✅ Pre-Run Checklist

Before running the analysis, make sure:

- [ ] Conda is installed (`conda --version`)
- [ ] Environment is created (`conda env list` shows wine-analysis)
- [ ] Environment is activated (`conda activate wine-analysis`)
- [ ] You're in the code directory (`cd code/`)
- [ ] All files are present (check with `ls` or `dir`)

---

## 🎓 For Your Presentation

When demonstrating the project:

1. **Show environment setup**: 
   ```bash
   conda activate wine-analysis
   ```

2. **Run complete analysis**:
   ```bash
   python main_analysis.py --method all
   ```

3. **Show the results folder structure**:
   ```bash
   tree results/  # or ls -R results/
   ```

4. **Open generated figures** from `results/figures/comparative/`

5. **Show the metrics CSV** from `results/metrics/comparative_report.csv`

---

## 🚀 Ready to Run!

You're all set! Here's the simplest way to get started:

```bash
# 1. Setup (one time only)
conda env create -f environment.yml

# 2. Every time you work on the project
conda activate wine-analysis
cd code

# 3. Run the analysis
python main_analysis.py --method all

# 4. Check your results
ls results/figures/comparative/
```

**That's it!** Your analysis will run and generate all the results you need for your project. 🍷✨

---

**Need help?** All scripts have detailed output and error messages to guide you through any issues!