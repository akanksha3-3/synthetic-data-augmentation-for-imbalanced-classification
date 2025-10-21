#!/usr/bin/env python
# coding: utf-8


"""
Synthetic Data Generation for Rare Cancer Subtypes
======================================================
Comparing SMOTE, ADASYN, and Statistical Methods For Data Augmentation

Author: Akanksha Waghamode
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_fscore_support
from scipy import stats 
from scipy.stats import ks_2samp
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import warnings
warnings.filterwarnings('ignore')

# set random seed for reproducibility
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


# =========================================================================================
# DATA GENERATION
# =========================================================================================

n_samples = 500
n_features = 20
n_informative = 15
n_redundant = 3
imbalance_ratio = 0.1

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_clusters_per_class=2,
    weights=[1-imbalance_ratio, imbalance_ratio],
    flip_y=0.01,
    random_state=42
)

feature_names = [f'Gene_Expression_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['Cancer_Status'] = y

print(f"\nDataset created successfully!")
print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Class Distribution:\n{df['Cancer_Status'].value_counts()}")
print(f"Imbalance Ratio: {(y==0).sum()}:{(y==1).sum()}")


# =========================================================================================
# SYNTHETIC DATA GENERATION
# =========================================================================================

X_train = df.drop('Cancer_Status', axis=1).values
y_train = df['Cancer_Status'].values
minority_indices = y_train == 1
X_minority_original = X_train[minority_indices]

synthetic_datasets = {'Original': (X_train, y_train)}

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
synthetic_datasets['SMOTE'] = (X_smote, y_smote)

# Borderline-SMOTE
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
X_borderline, y_borderline = borderline_smote.fit_resample(X_train, y_train)
synthetic_datasets['Borderline_SMOTE'] = (X_borderline, y_borderline)

# ADASYN
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
synthetic_datasets['ADASYN'] = (X_adasyn, y_adasyn)

# Statistical Gaussian
mean_vector = np.mean(X_minority_original, axis=0)
cov_matrix = np.cov(X_minority_original.T)
n_synthetic = len(y_train[y_train==0]) - len(y_train[y_train==1])
X_synthetic_gaussian = np.random.multivariate_normal(mean_vector, cov_matrix, n_synthetic)
X_gaussian = np.vstack([X_train, X_synthetic_gaussian])
y_gaussian = np.hstack([y_train, np.ones(n_synthetic)])
synthetic_datasets['Statistical_Gaussian'] = (X_gaussian, y_gaussian)

# Noise Injection
noise_level = 0.1
X_minority_noisy = X_minority_original.copy()
for _ in range(n_synthetic):
    idx = np.random.randint(0, len(X_minority_original))
    sample = X_minority_original[idx]
    noise = np.random.normal(0, noise_level * np.std(X_minority_original, axis=0), n_features)
    noisy_sample = sample + noise
    X_minority_noisy = np.vstack([X_minority_noisy, noisy_sample])
X_noise = np.vstack([X_train, X_minority_noisy[len(X_minority_original):]])
y_noise = np.hstack([y_train, np.ones(len(X_minority_noisy) - len(X_minority_original))])
synthetic_datasets['Noise_Injection'] = (X_noise, y_noise)

print(f"\nGenerated {len(synthetic_datasets)-1} augmented datasets")


# =========================================================================================
# QUALITY VALIDATION
# =========================================================================================

print("\n" + "="*80)
print("QUALITY VALIDATION (Kolmogorov-Smirnov Test)")
print("="*80)

ks_results = []
for method_name, (X_synthetic, y_synthetic) in synthetic_datasets.items():
    if method_name == "Original":
        continue
    
    synthetic_minority_mask = (y_synthetic == 1) & (np.arange(len(y_synthetic)) >= len(y_train))
    X_synthetic_minority = X_synthetic[synthetic_minority_mask]

    if len(X_synthetic_minority) == 0:
        continue

    ks_scores = []
    for feature_idx in range(n_features):
        real_feature = X_minority_original[:, feature_idx]
        synthetic_feature = X_synthetic_minority[:, feature_idx]
        statistic, pvalue = ks_2samp(real_feature, synthetic_feature)
        ks_scores.append(pvalue)

    avg_pvalue = np.mean(ks_scores)
    ks_results.append({
        'Method': method_name,
        'Avg_P_Value': avg_pvalue,
        'Features_Similar': sum(np.array(ks_scores) > 0.05),
        'Quality': 'Good' if avg_pvalue > 0.05 else 'Review'
    })
    
    print(f"{method_name:20s} | Avg p-value: {avg_pvalue:.4f} | Similar features: {sum(np.array(ks_scores) > 0.05)}/{n_features} | {ks_results[-1]['Quality']}")

# Correlation Preservation
print("\n" + "="*80)
print("CORRELATION PRESERVATION")
print("="*80)

original_corr = np.corrcoef(X_minority_original.T)
correlation_preservation = []

for method_name, (X_synthetic, y_synthetic) in synthetic_datasets.items():
    if method_name == 'Original':
        continue

    synthetic_minority_mask = (y_synthetic == 1) & (np.arange(len(y_synthetic)) >= len(y_train))
    X_synthetic_minority = X_synthetic[synthetic_minority_mask]
    
    if len(X_synthetic_minority) == 0:
        continue
    
    synthetic_corr = np.corrcoef(X_synthetic_minority.T)
    corr_diff = np.abs(original_corr - synthetic_corr)
    mean_diff = np.mean(corr_diff[np.triu_indices_from(corr_diff, k=1)])
    
    correlation_preservation.append({
        'Method': method_name,
        'Mean_Correlation_Difference': mean_diff,
        'Preservation': 'Excellent' if mean_diff < 0.1 else 'Good' if mean_diff < 0.2 else 'Fair'
    })
    
    print(f"{method_name:20s} | Mean difference: {mean_diff:.4f} | {correlation_preservation[-1]['Preservation']}")


# =========================================================================================
# MODEL PERFORMANCE
# =========================================================================================

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

X_orig, y_orig = synthetic_datasets['Original']
X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

model_results = []
for method_name, (X_aug, y_aug) in synthetic_datasets.items():
    X_use = X_train_orig if method_name == 'Original' else X_aug
    y_use = y_train_orig if method_name == 'Original' else y_aug
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_use, y_use)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    model_results.append({
        'Method': method_name,
        'ROC_AUC': roc_auc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Training_Samples': len(y_use)
    })

results_df = pd.DataFrame(model_results)
print(results_df.to_string(index=False))

best_idx = results_df['ROC_AUC'].idxmax()
best_method = results_df.loc[best_idx]
print(f"\nBest Method: {best_method['Method']}")
print(f"ROC-AUC: {best_method['ROC_AUC']:.4f} | Recall: {best_method['Recall']:.4f} | F1: {best_method['F1_Score']:.4f}")


# =========================================================================================
# VISUALIZATION
# =========================================================================================

fig = plt.figure(figsize=(18,12))

# Class Distribution
ax1 = plt.subplot(3, 3, 1)
methods = list(synthetic_datasets.keys())
class_0_counts = [sum(synthetic_datasets[m][1] == 0) for m in methods]
class_1_counts = [sum(synthetic_datasets[m][1] == 1) for m in methods]
x_pos = np.arange(len(methods))
width = 0.35
ax1.bar(x_pos - width/2, class_0_counts, width, label='Normal', alpha=0.8)
ax1.bar(x_pos + width/2, class_1_counts, width, label='Cancer', alpha=0.8)
ax1.set_ylabel('Samples')
ax1.set_title('Class Distribution', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# PCA - Original
ax2 = plt.subplot(3, 3, 2)
pca = PCA(n_components=2)
X_orig_pca = pca.fit_transform(X_train_orig)
colors_orig = ['blue' if y == 0 else 'red' for y in y_train_orig]
ax2.scatter(X_orig_pca[:, 0], X_orig_pca[:, 1], c=colors_orig, alpha=0.6, s=30)
ax2.set_title('PCA: Original Data', fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC1 ({pca.explained_variance_ratio_[1]:.1%})')

# PCA - SMOTE
ax3 = plt.subplot(3, 3, 3)
X_smote_data, y_smote_data = synthetic_datasets['SMOTE']
X_smote_pca = pca.transform(X_smote_data)
is_synthetic = np.arange(len(y_smote_data)) >= len(y_train)
colors_smote = ['blue' if y == 0 else 'orange' if syn else 'red'
               for y, syn in zip(y_smote_data, is_synthetic)]
ax3.scatter(X_smote_pca[:, 0], X_smote_pca[:, 1], c=colors_smote, alpha=0.6, s=30)
ax3.set_title('PCA: SMOTE Agumented', fontweight='bold')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax3.set_ylabel(f'PC1 ({pca.explained_variance_ratio_[1]:.1%})')

# ROC-AUC Performance 
ax4 = plt.subplot(3, 3, 4)
methods_plot = results_df['Method'].tolist()
roc_aucs = results_df['ROC_AUC'].tolist()
colors = ['crimson' if m == best_method['Method'] else 'steelblue' for m in methods_plot]
bars = ax4.barh(methods_plot, roc_aucs, color=colors, alpha=0.8)
ax4.set_xlabel('ROC-AUC')
ax4.set_title('Model Performance', fontweight='bold')
ax4.set_xlim([0, 1.05])
for bar, val in zip(bars, roc_aucs):
    ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax4.grid(axis='x', alpha=0.3)

# Recall Performance
ax5 = plt.subplot(3, 3, 5)
recalls = results_df['Recall'].tolist()
bars = ax5.barh(methods_plot, recalls, color=colors, alpha=0.8)
ax5.set_xlabel('Recall')
ax5.set_title('Recall (Sensitivity)', fontweight='bold')
ax5.set_xlim([0, 1.05])
for bar, val in zip(bars, recalls):
    ax5.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax5.grid(axis='x', alpha=0.3)

# Quality Metrics Heatmap
ax6 = plt.subplot(3, 3, 6)
quality_df = pd.DataFrame(ks_results)
if len(quality_df) > 0:
    quality_matrix = quality_df.set_index('Method')[['Avg_P_Value', 'Features_Similar']]
    quality_matrix['Features_Similar'] = quality_matrix['Features_Similar'] / n_features
    sns.heatmap(quality_matrix.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax6,
               cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    ax6.set_title('Quality Metrics', fontweight='bold')

# Feature Distribution 1
ax7 = plt.subplot(3, 3, 7)
feature_idx = 0
X_smote_synthetic = X_smote_data[(y_smote_data == 1) & (np.arange(len(y_smote_data)) >= len(y_train))]
ax7.hist(X_minority_original[:, feature_idx], bins=20, alpha=0.6, label='Real', color='red', edgecolor='black')
ax7.hist(X_smote_synthetic[:, feature_idx], bins=20, alpha=0.6, label='Synthetic', color='orange', edgecolor='black')
ax7.set_xlabel(feature_names[feature_idx])
ax7.set_ylabel('Frequency')
ax7.set_title('Distribution Comparision (Feature 1)', fontweight='bold')
ax7.legend()

# Feature Distribution 2
ax8 = plt.subplot(3, 3, 8)
feature_idx = 4
ax8.hist(X_minority_original[:, feature_idx], bins=20, alpha=0.6, label='Real', color='red', edgecolor='black')
ax8.hist(X_smote_synthetic[:, feature_idx], bins=20, alpha=0.6, label='Synthetic', color='orange', edgecolor='black')
ax8.set_xlabel(feature_names[feature_idx])
ax8.set_ylabel('Frequency')
ax8.set_title('Distribution Comparision (Feature 5)', fontweight='bold')
ax8.legend()

# Training Size vs Performance
ax9 = plt.subplot(3, 3, 9)
training_samples = results_df['Training_Samples'].tolist()
ax9.scatter(training_samples, roc_aucs, s=100, c=colors, alpha=0.7, edgecolor='black')
for i, method in enumerate(methods_plot):
    ax9.annotate(method, (training_samples[i], roc_aucs[i]), fontsize=8, ha='right', va='bottom')
ax9.set_xlabel('Training Samples')
ax9.set_ylabel('ROC-AUC')
ax9.set_title('Sample Size vs Performance', fontweight='bold')
ax9.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('synthetic_data_generation_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved: synthetic_data_generation_analysis.png")


# =========================================================================================
# SAVE RESULTS
# =========================================================================================

results_df.to_csv('synthetic_data_comparison_results.csv', index=False)
print("Results saved: synthetic_data_comparison_results.csv")

print("\n" + "="*80)


# In[ ]:




