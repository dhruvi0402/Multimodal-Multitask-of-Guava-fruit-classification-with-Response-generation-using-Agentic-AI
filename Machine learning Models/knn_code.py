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
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================
# CONFIGURATION
# ============================================
TRAIN_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Diseases\train'
VAL_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Diseases\val'
TEST_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Diseases\test'

IMG_SIZE = (128, 128)
OUTPUT_DIR = 'KNN_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("K-NEAREST NEIGHBORS IMAGE CLASSIFICATION - GUAVA DISEASE DETECTION")
print("="*70)

# ============================================
# ENHANCED FEATURE EXTRACTION
# ============================================
def extract_enhanced_features(image_path):
    """Extract HOG + Color Histogram + Texture features"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 1. HOG features
    hog_features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    # 2. Color histogram features
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img_resized], [i], None, [64], [0, 256])
        hist_features.extend(hist.flatten())
    
    # 3. Texture features
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    texture_features = [
        np.mean(laplacian),
        np.std(laplacian),
        np.max(laplacian),
        np.min(laplacian)
    ]
    
    combined_features = np.concatenate([hog_features, hist_features, texture_features])
    return combined_features.astype(np.float32)

def load_dataset(directory, dataset_name):
    """Load images and extract features"""
    X, y = [], []
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    print(f"\nLoading {dataset_name} data:")
    for label_idx, label in enumerate(class_names):
        label_dir = os.path.join(directory, label)
        image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {label}: {len(image_files)} images", end='')
        
        for filename in image_files:
            image_path = os.path.join(label_dir, filename)
            features = extract_enhanced_features(image_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)
        
        print(f" → {len([yi for yi in y if yi == label_idx])} loaded")
    
    return np.array(X, dtype=np.float32), np.array(y), class_names

# Load datasets
X_train, y_train, class_names = load_dataset(TRAIN_DIR, "Training")
X_val, y_val, _ = load_dataset(VAL_DIR, "Validation")
X_test, y_test, _ = load_dataset(TEST_DIR, "Test")

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

print(f"\n{'='*70}")
print(f"Dataset Summary:")
print(f"  Classes: {class_names}")
print(f"  Training samples: {len(X_train_full)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Feature dimension: {X_train_full.shape[1]}")
print(f"{'='*70}")

# ============================================
# FEATURE SCALING (CRITICAL FOR KNN)
# ============================================
print("\nScaling features (critical for KNN distance calculations)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# ============================================
# FIND OPTIMAL K VALUE
# ============================================
print("\nFinding optimal K value using cross-validation...")
k_values = [3, 5, 7, 9, 11, 13, 15]
cv_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    scores = cross_val_score(knn_temp, X_train_scaled, y_train_full, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"  K={k:2d}: CV Accuracy = {scores.mean()*100:.2f}% (±{scores.std()*100:.2f}%)")

optimal_k = k_values[np.argmax(cv_scores)]
print(f"\n✓ Optimal K value: {optimal_k}")

# ============================================
# TRAIN KNN MODEL
# ============================================
print(f"\nTraining KNN model with K={optimal_k}...")

knn_model = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='distance',      # Weight by distance (closer neighbors have more influence)
    metric='euclidean',      # Euclidean distance
    n_jobs=-1
)

knn_model.fit(X_train_scaled, y_train_full)
print("✓ Model trained successfully!")

# ============================================
# COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

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
    print("(✓ Excellent - No overfitting)")
elif overfitting_gap < 10:
    print("(✓ Good - Minimal overfitting)")
elif overfitting_gap < 15:
    print("(⚠ Moderate overfitting)")
else:
    print("(❌ Severe overfitting)")

print("\n" + "-"*70)
print("DETAILED CLASSIFICATION REPORT")
print("-"*70)
report = classification_report(y_test, test_pred, target_names=class_names, digits=4)
print(report)

cm = confusion_matrix(y_test, test_pred)

# ============================================
# SAVE RESULTS PERMANENTLY
# ============================================
print("\nSaving results...")

joblib.dump(knn_model, os.path.join(OUTPUT_DIR, 'knn_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

results = {
    'model': 'K-Nearest Neighbors (KNN)',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'parameters': {
        'n_neighbors': int(optimal_k),
        'weights': 'distance',
        'metric': 'euclidean'
    },
    'k_value_analysis': {
        'tested_k_values': k_values,
        'cv_accuracies': [float(score) for score in cv_scores],
        'optimal_k': int(optimal_k)
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
    f.write("KNN CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Optimal K value: {optimal_k}\n")
    f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.2f}%\n\n")
    f.write(report)

# ============================================
# VISUALIZATIONS
# ============================================

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
plt.title('KNN - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance Metrics & K-value Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy*100, train_precision*100, train_recall*100, train_f1*100]
test_scores = [test_accuracy*100, test_precision*100, test_recall*100, test_f1*100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, train_scores, width, label='Training', color='#9b59b6', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_scores, width, label='Test', color='#8e44ad', alpha=0.8)

ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_title('KNN - Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 105)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# K-value analysis
k_accuracies = [score*100 for score in cv_scores]
ax2.plot(k_values, k_accuracies, 'o-', linewidth=2.5, markersize=10, color='purple')
ax2.plot(optimal_k, k_accuracies[k_values.index(optimal_k)], 'ro', markersize=15, 
         label=f'Optimal K={optimal_k}', zorder=5)
ax2.set_xlabel('K Value', fontsize=12)
ax2.set_ylabel('Cross-Validation Accuracy (%)', fontsize=12)
ax2.set_title('K Value vs Accuracy', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xticks(k_values)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Overfitting Analysis
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Training\nAccuracy', 'Test\nAccuracy', 'Overfitting\nGap']
values = [train_accuracy*100, test_accuracy*100, overfitting_gap]
colors = ['#9b59b6', '#8e44ad', '#e74c3c' if overfitting_gap > 10 else '#f39c12']

bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('KNN - Overfitting Analysis', fontsize=16, fontweight='bold')
ax.set_ylim(0, max(values) + 15)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Detailed K-value comparison
print("\nTesting all K values on test set for comparison...")
k_test_accuracies = []
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    knn_temp.fit(X_train_scaled, y_train_full)
    pred = knn_temp.predict(X_test_scaled)
    k_test_accuracies.append(accuracy_score(y_test, pred) * 100)

plt.figure(figsize=(12, 6))
plt.plot(k_values, k_accuracies, 'o-', linewidth=2.5, markersize=8, color='purple', label='CV Accuracy')
plt.plot(k_values, k_test_accuracies, 's-', linewidth=2.5, markersize=8, color='orange', label='Test Accuracy')
plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('KNN - Cross-Validation vs Test Accuracy for Different K Values', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'k_value_detailed_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("RESULTS SAVED SUCCESSFULLY!")
print("="*70)
print(f"\nAll results saved in '{OUTPUT_DIR}/' directory:")
print("  ✓ knn_model.pkl (trained model)")
print("  ✓ scaler.pkl (feature scaler)")
print("  ✓ results.json (all metrics)")
print("  ✓ classification_report.txt")
print("  ✓ confusion_matrix.png")
print("  ✓ performance_analysis.png")
print("  ✓ overfitting_analysis.png")
print("  ✓ k_value_detailed_analysis.png")
print("\nYou can view these files anytime without re-running the code!")

# Memory cleanup
del X_train, X_val, X_train_full, X_train_scaled
import gc
gc.collect()
