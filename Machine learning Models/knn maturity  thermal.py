import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import cv2
from skimage.feature import hog
import joblib
import json
from datetime import datetime

# Force UTF-8 encoding for Windows
os.environ['PYTHONUTF8'] = '1'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# ============================================
# CONFIGURATION - THERMAL MATURITY
# ============================================
TRAIN_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split  thermal\train'
VAL_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split  thermal\val'
TEST_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split  thermal\test'

IMG_SIZE = (128, 128)
OUTPUT_DIR = 'Thermal_Maturity_KNN_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("K-NEAREST NEIGHBORS - THERMAL IMAGE MATURITY CLASSIFICATION")
print("="*80)
print(f"Task: Classify guava maturity (immature, mature, semi_mature)")
print(f"Image Type: Thermal")
print(f"Model: K-Nearest Neighbors (KNN)")
print("="*80)

# ============================================
# ENHANCED FEATURE EXTRACTION
# ============================================
def extract_enhanced_features(image_path):
    """
    Extract HOG + Color Histogram + Texture features
    Optimized for thermal images
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 1. HOG features (shape and gradient information)
    hog_features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    # 2. Color histogram features (thermal intensity distribution)
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img_resized], [i], None, [64], [0, 256])
        hist_features.extend(hist.flatten())
    
    # 3. Texture features (surface characteristics)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    texture_features = [
        np.mean(laplacian),
        np.std(laplacian),
        np.max(laplacian),
        np.min(laplacian)
    ]
    
    # 4. Statistical features (thermal image specific)
    statistical_features = [
        np.mean(img_gray),
        np.std(img_gray),
        np.median(img_gray),
        np.percentile(img_gray, 25),
        np.percentile(img_gray, 75)
    ]
    
    # Combine all features
    combined_features = np.concatenate([
        hog_features, 
        hist_features, 
        texture_features,
        statistical_features
    ])
    
    return combined_features.astype(np.float32)

def load_dataset(directory, dataset_name):
    """Load images and extract features"""
    X, y = [], []
    class_names = sorted([d for d in os.listdir(directory) 
                         if os.path.isdir(os.path.join(directory, d))])
    
    print(f"\nLoading {dataset_name} data:")
    for label_idx, label in enumerate(class_names):
        label_dir = os.path.join(directory, label)
        image_files = [f for f in os.listdir(label_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {label:15s}: {len(image_files):4d} images", end='')
        
        loaded_count = 0
        for filename in image_files:
            image_path = os.path.join(label_dir, filename)
            features = extract_enhanced_features(image_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)
                loaded_count += 1
        
        print(f" -> {loaded_count:4d} loaded")
    
    return np.array(X, dtype=np.float32), np.array(y), class_names

# ============================================
# LOAD DATASETS
# ============================================
print("\n" + "-"*80)
print("LOADING THERMAL MATURITY DATASET")
print("-"*80)

X_train, y_train, class_names = load_dataset(TRAIN_DIR, "Training")
X_val, y_val, _ = load_dataset(VAL_DIR, "Validation")
X_test, y_test, _ = load_dataset(TEST_DIR, "Test")

# Combine train and validation for final training
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

print(f"\n{'='*80}")
print(f"DATASET SUMMARY")
print(f"{'='*80}")
print(f"Classes detected: {class_names}")
print(f"Number of classes: {len(class_names)}")
print(f"Training samples: {len(X_train_full)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature dimension: {X_train_full.shape[1]}")
print(f"{'='*80}")

# ============================================
# FEATURE SCALING (CRITICAL FOR KNN)
# ============================================
print("\nScaling features (CRITICAL for KNN distance calculations)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)
print("[OK] Features scaled using StandardScaler")

# ============================================
# FIND OPTIMAL K VALUE
# ============================================
print("\n" + "="*80)
print("FINDING OPTIMAL K VALUE")
print("="*80)
print("\nTesting K values: [3, 5, 7, 9, 11, 13, 15]")
print("Using 3-fold cross-validation on training set...")
print("-"*80)

k_values = [3, 5, 7, 9, 11, 13, 15]
cv_scores = []
cv_std = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    scores = cross_val_score(knn_temp, X_train_scaled, y_train_full, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_std.append(scores.std())
    print(f"K={k:2d}: CV Accuracy = {scores.mean()*100:5.2f}% (+/- {scores.std()*100:4.2f}%)")

optimal_k = k_values[np.argmax(cv_scores)]
print(f"\n[INFO] Optimal K value: {optimal_k}")
print(f"[INFO] Best CV Accuracy: {max(cv_scores)*100:.2f}%")

# ============================================
# TRAIN KNN MODEL
# ============================================
print("\n" + "="*80)
print("TRAINING KNN MODEL")
print("="*80)
print(f"\nModel Configuration:")
print(f"  - K (number of neighbors): {optimal_k}")
print(f"  - Distance weighting: Enabled (closer neighbors have more influence)")
print(f"  - Distance metric: Euclidean")
print(f"  - Algorithm: Auto (selects best for dataset)")
print("\nStarting training...")

knn_model = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='distance',      # Weight by distance
    metric='euclidean',      # Euclidean distance
    algorithm='auto',        # Auto-select best algorithm
    n_jobs=-1
)

knn_model.fit(X_train_scaled, y_train_full)
print("\n[SUCCESS] Model trained successfully!")

# ============================================
# COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Training metrics
train_pred = knn_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train_full, train_pred)
train_precision = precision_score(y_train_full, train_pred, average='weighted')
train_recall = recall_score(y_train_full, train_pred, average='weighted')
train_f1 = f1_score(y_train_full, train_pred, average='weighted')

# Test metrics
test_pred = knn_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, average='weighted')
test_recall = recall_score(y_test, test_pred, average='weighted')
test_f1 = f1_score(y_test, test_pred, average='weighted')

# Overfitting gap
overfitting_gap = (train_accuracy - test_accuracy) * 100

print(f"\nTRAINING METRICS:")
print(f"  Accuracy:  {train_accuracy*100:.2f}%")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

print(f"\nTEST METRICS:")
print(f"  Accuracy:  {test_accuracy*100:.2f}%")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

print(f"\nOVERFITTING ANALYSIS:")
print(f"  Gap: {overfitting_gap:.2f}%", end=' ')
if overfitting_gap < 5:
    print("(Excellent - No overfitting)")
elif overfitting_gap < 10:
    print("(Good - Minimal overfitting)")
elif overfitting_gap < 15:
    print("(Moderate overfitting)")
else:
    print("(Severe overfitting - Model memorizing data)")

# Per-class evaluation
print("\n" + "-"*80)
print("DETAILED CLASSIFICATION REPORT")
print("-"*80)
report = classification_report(y_test, test_pred, target_names=class_names, digits=4)
print(report)

cm = confusion_matrix(y_test, test_pred)

# ============================================
# SAVE RESULTS PERMANENTLY
# ============================================
print("\n" + "-"*80)
print("SAVING RESULTS")
print("-"*80)

joblib.dump(knn_model, os.path.join(OUTPUT_DIR, 'knn_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

results = {
    'model': 'K-Nearest Neighbors (KNN)',
    'image_type': 'Thermal',
    'task': 'Maturity Classification',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'classes': class_names,
    'parameters': {
        'n_neighbors': int(optimal_k),
        'weights': 'distance',
        'metric': 'euclidean',
        'image_size': IMG_SIZE
    },
    'k_value_analysis': {
        'tested_k_values': k_values,
        'cv_accuracies': [float(score) for score in cv_scores],
        'cv_std_deviations': [float(std) for std in cv_std],
        'optimal_k': int(optimal_k),
        'best_cv_accuracy': float(max(cv_scores))
    },
    'model_info': {
        'feature_dimension': int(X_train_full.shape[1]),
        'training_samples': int(len(X_train_full))
    },
    'training_metrics': {
        'accuracy': float(train_accuracy),
        'precision': float(train_precision),
        'recall': float(train_recall),
        'f1_score': float(train_f1)
    },
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1)
    },
    'overfitting_gap_percent': float(overfitting_gap),
    'classification_report': report
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("KNN THERMAL MATURITY CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Classes: {', '.join(class_names)}\n")
    f.write(f"Optimal K value: {optimal_k}\n")
    f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.2f}%\n\n")
    f.write(report)

print("[OK] Model saved: knn_model.pkl")
print("[OK] Scaler saved: scaler.pkl")
print("[OK] Results saved: results.json, classification_report.txt")

# ============================================
# VISUALIZATIONS
# ============================================
print("\nGenerating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
plt.title('KNN - Thermal Maturity Classification\nConfusion Matrix', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: confusion_matrix.png")

# 2. Performance Metrics Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy*100, train_precision*100, train_recall*100, train_f1*100]
test_scores = [test_accuracy*100, test_precision*100, test_recall*100, test_f1*100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, train_scores, width, label='Training', 
                color='#9b59b6', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, test_scores, width, label='Test', 
                color='#8e44ad', alpha=0.8, edgecolor='black')

ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 105)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Per-class accuracy
per_class_acc = []
for i in range(len(class_names)):
    class_correct = cm[i, i]
    class_total = np.sum(cm[i, :])
    class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
    per_class_acc.append(class_acc)

colors_per_class = ['#e74c3c', '#f39c12', '#27ae60']
bars3 = ax2.bar(class_names, per_class_acc, color=colors_per_class, 
                alpha=0.8, edgecolor='black')
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars3, per_class_acc):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: performance_metrics.png")

# 3. Overfitting Analysis
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Training\nAccuracy', 'Test\nAccuracy', 'Overfitting\nGap']
values = [train_accuracy*100, test_accuracy*100, overfitting_gap]
colors_over = ['#9b59b6', '#8e44ad', '#e74c3c' if overfitting_gap > 10 else '#f39c12']

bars = ax.bar(categories, values, color=colors_over, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('KNN - Thermal Maturity: Overfitting Analysis', 
             fontsize=16, fontweight='bold')
ax.set_ylim(0, max(values) + 15)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: overfitting_analysis.png")

# 4. K-Value Analysis (Cross-Validation)
fig, ax = plt.subplots(figsize=(12, 6))
cv_accuracies_percent = [score*100 for score in cv_scores]
cv_std_percent = [std*100 for std in cv_std]

ax.errorbar(k_values, cv_accuracies_percent, yerr=cv_std_percent, 
            fmt='o-', linewidth=2.5, markersize=10, color='purple',
            ecolor='lightgray', elinewidth=2, capsize=5, capthick=2)
ax.plot(optimal_k, cv_accuracies_percent[k_values.index(optimal_k)], 
        'r*', markersize=20, label=f'Optimal K={optimal_k}', zorder=5)
ax.set_xlabel('K Value', fontsize=12)
ax.set_ylabel('Cross-Validation Accuracy (%)', fontsize=12)
ax.set_title('KNN - K Value vs Cross-Validation Accuracy', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(k_values)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'k_value_cv_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: k_value_cv_analysis.png")

# 5. K-Value Comparison on Test Set
print("\nTesting all K values on test set for comparison...")
k_test_accuracies = []
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    knn_temp.fit(X_train_scaled, y_train_full)
    pred = knn_temp.predict(X_test_scaled)
    k_test_accuracies.append(accuracy_score(y_test, pred) * 100)

plt.figure(figsize=(12, 6))
plt.plot(k_values, cv_accuracies_percent, 'o-', linewidth=2.5, markersize=8, 
         color='purple', label='CV Accuracy')
plt.plot(k_values, k_test_accuracies, 's-', linewidth=2.5, markersize=8, 
         color='orange', label='Test Accuracy')
plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
            label=f'Optimal K={optimal_k}')
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('KNN - Cross-Validation vs Test Accuracy for Different K Values', 
          fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.xticks(k_values)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'k_value_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: k_value_comparison.png")

# 6. Class Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Training set distribution
train_class_counts = [np.sum(y_train_full == i) for i in range(len(class_names))]
ax1.bar(class_names, train_class_counts, color=['#e74c3c', '#f39c12', '#27ae60'], 
        alpha=0.8, edgecolor='black')
ax1.set_ylabel('Number of Samples', fontsize=12)
ax1.set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for i, (name, count) in enumerate(zip(class_names, train_class_counts)):
    ax1.text(i, count, f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Test set distribution
test_class_counts = [np.sum(y_test == i) for i in range(len(class_names))]
ax2.bar(class_names, test_class_counts, color=['#e74c3c', '#f39c12', '#27ae60'], 
        alpha=0.8, edgecolor='black')
ax2.set_ylabel('Number of Samples', fontsize=12)
ax2.set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for i, (name, count) in enumerate(zip(class_names, test_class_counts)):
    ax2.text(i, count, f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: class_distribution.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*80)
print("RESULTS SAVED SUCCESSFULLY!")
print("="*80)
print(f"\nAll results saved in '{OUTPUT_DIR}/' directory:")
print("  [OK] knn_model.pkl (trained model)")
print("  [OK] scaler.pkl (feature scaler)")
print("  [OK] results.json (all metrics)")
print("  [OK] classification_report.txt")
print("  [OK] confusion_matrix.png")
print("  [OK] performance_metrics.png")
print("  [OK] overfitting_analysis.png")
print("  [OK] k_value_cv_analysis.png")
print("  [OK] k_value_comparison.png")
print("  [OK] class_distribution.png")
print("\nYou can view these files anytime without re-running the code!")

print("\n" + "="*80)
print("KEY RESULTS SUMMARY")
print("="*80)
print(f"Model: K-Nearest Neighbors (KNN)")
print(f"Image Type: Thermal")
print(f"Task: Maturity Classification")
print(f"Classes: {', '.join(class_names)}")
print(f"\nOptimal K Value: {optimal_k}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Overfitting Gap: {overfitting_gap:.2f}%")
print(f"Best CV Accuracy: {max(cv_scores)*100:.2f}%")
print("="*80)

# Memory cleanup
del X_train, X_val, X_train_full, X_train_scaled
import gc
gc.collect()
print("\n[OK] Memory cleaned up")
print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80 + "\n")
