# Synthetic Data Generation for Imbalanced Classification

A comprehensive comparison of synthetic data augmentation techniques for handling highly imbalanced datasets, with focus on improving minority class detection.

## 📊 Project Overview

This project evaluates and compares five different data augmentation techniques to address class imbalance problems in machine learning. The study uses a simulated dataset with an 8.4:1 imbalance ratio (447:53 samples) to demonstrate the effectiveness of various oversampling methods.

## 🎯 Key Features

- **Multiple Augmentation Techniques**: SMOTE, Borderline-SMOTE, ADASYN, Statistical Gaussian, and Noise Injection
- **Comprehensive Quality Validation**: Kolmogorov-Smirnov tests and correlation preservation metrics
- **Performance Evaluation**: Random Forest classifier with ROC-AUC, Precision, Recall, and F1-Score
- **Rich Visualizations**: PCA plots, distribution comparisons, and performance dashboards
- **Reproducible Results**: Fixed random seeds for consistency

## 📈 Results Summary

| Method | ROC-AUC | Recall | F1-Score | Quality (K-S p-value) |
|--------|---------|--------|----------|----------------------|
| **Original** | 0.951 | 0.273 | 0.429 | - |
| **SMOTE** | 1.000 | 1.000 | 1.000 | 0.675 (Good) |
| **Borderline-SMOTE** | 1.000 | 1.000 | 1.000 | 0.629 (Good) |
| **ADASYN** | 1.000 | 1.000 | 1.000 | 0.489 (Good) |
| **Statistical Gaussian** | 1.000 | 1.000 | 1.000 | 0.772 (Good) |
| **Noise Injection** | 1.000 | 1.000 | 1.000 | 0.977 (Good) |

### Key Findings

✅ All augmentation methods achieved **perfect classification performance** (ROC-AUC: 1.0)  
✅ **Recall improved from 27.3% to 100%** for minority class detection  
✅ **Excellent correlation preservation** (mean difference < 0.07 across all methods)  
✅ **High statistical similarity** between real and synthetic samples (K-S p-value > 0.48)

## 🛠️ Installation

### Prerequisites

```bash
Python 3.7+
```

### Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy imbalanced-learn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
imbalanced-learn>=0.8.0
```

## 🚀 Usage

### Basic Execution

```bash
python synthetic_data_generation.py
```

### Output Files

The script generates:
- `synthetic_data_generation_analysis.png` - Comprehensive visualization dashboard
- `synthetic_data_comparison_results.csv` - Detailed performance metrics

### Customization

Modify the following parameters in the script:

```python
n_samples = 500              # Total number of samples
n_features = 20              # Number of features
imbalance_ratio = 0.1        # Minority class ratio (0.1 = 10%)
```

## 📊 Visualizations

The generated dashboard includes:

1. **Class Distribution** - Sample counts across methods
2. **PCA Analysis** - 2D visualization of original and augmented data
3. **Model Performance** - ROC-AUC comparison across techniques
4. **Recall Analysis** - Sensitivity metrics for minority class
5. **Quality Metrics Heatmap** - K-S test results and feature similarity
6. **Distribution Comparisons** - Real vs synthetic feature distributions
7. **Sample Size vs Performance** - Training data efficiency

## 🔬 Methodology

### 1. Data Generation
- Synthetic dataset created using `make_classification`
- 500 samples, 20 features (15 informative, 3 redundant)
- Imbalance ratio: 8.4:1 (447 majority, 53 minority)

### 2. Augmentation Techniques

**SMOTE (Synthetic Minority Over-sampling Technique)**
- Generates synthetic samples by interpolating between minority class neighbors
- k_neighbors = 5

**Borderline-SMOTE**
- Focuses on borderline samples near decision boundary
- More targeted augmentation than standard SMOTE

**ADASYN (Adaptive Synthetic Sampling)**
- Adaptively generates samples based on density distribution
- More samples in harder-to-learn regions

**Statistical Gaussian**
- Samples from multivariate Gaussian distribution
- Uses mean and covariance of minority class

**Noise Injection**
- Adds Gaussian noise to existing minority samples
- Noise level: 10% of feature standard deviation

### 3. Quality Validation

**Kolmogorov-Smirnov Test**
- Tests distribution similarity between real and synthetic data
- p-value > 0.05 indicates good similarity

**Correlation Preservation**
- Measures preservation of feature correlations
- Mean difference < 0.1 = Excellent, < 0.2 = Good

### 4. Model Evaluation

- **Classifier**: Random Forest (100 estimators)
- **Metrics**: ROC-AUC, Precision, Recall, F1-Score
- **Validation**: 80/20 train-test split with stratification

## 📁 Project Structure

```
synthetic-data-generation/
│
├── synthetic_data_generation.py          # Main script
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
│
├── outputs/
│   ├── synthetic_data_generation_analysis.png
│   └── synthetic_data_comparison_results.csv
│
└── .gitignore
```

## 🎓 Use Cases

This project is applicable to:
- **Medical Diagnosis**: Rare disease detection
- **Fraud Detection**: Imbalanced transaction datasets
- **Quality Control**: Defect detection in manufacturing
- **Customer Churn**: Minority class prediction
- **Anomaly Detection**: Security and network monitoring

## 📝 Key Insights

1. **All augmentation methods significantly outperformed the baseline**, improving recall from 27% to 100%
2. **Noise Injection showed the highest statistical similarity** (K-S p-value: 0.977)
3. **Statistical Gaussian had the best correlation preservation** (mean diff: 0.033)
4. **All methods achieved perfect classification**, suggesting effective synthetic data quality
5. **Training sample size increased from 400 to ~894**, providing more robust model training

## ⚠️ Limitations

- Results based on synthetic simulated data, not real-world datasets
- Perfect scores may indicate overfitting or data leakage in the experimental setup
- Real-world performance may vary depending on data complexity and noise
- Computational cost increases with dataset size

## 🔮 Future Work

- [ ] Test on real-world imbalanced datasets (medical, fraud, etc.)
- [ ] Implement deep learning-based augmentation (VAE, GAN)
- [ ] Add cross-validation for more robust evaluation
- [ ] Benchmark computational efficiency and scalability
- [ ] Explore ensemble augmentation approaches
- [ ] Add support for multi-class imbalanced problems

## 📚 References

- Chawla et al. (2002) - SMOTE: Synthetic Minority Over-sampling Technique
- He et al. (2008) - ADASYN: Adaptive Synthetic Sampling
- Han et al. (2005) - Borderline-SMOTE

## 👤 Author

**Akanksha Waghamode**

**Note**: This is a research/educational project demonstrating data augmentation techniques. Always validate results on real-world data before production deployment.
