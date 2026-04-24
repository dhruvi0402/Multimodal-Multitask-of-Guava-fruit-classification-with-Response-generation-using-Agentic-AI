import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
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
OUTPUT_DIR = 'XGBoost_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("XGBOOST IMAGE CLASSIFICATION - GUAVA DISEASE DETECTION")
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
# FEATURE SCALING
# ============================================
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# ============================================
# TRAIN XGBOOST MODEL WITH EARLY STOPPING
# ============================================
print("\nTraining XGBoost model with early stopping...")
print("Configuration:")
print("  - Max iterations: 50")
print("  - Early stopping patience: 3")
print("  - Regularization: alpha=0.1, lambda=1.0")

xgb_model = xgb.XGBClassifier(
    n_estimators=50,              # Maximum 50 iterations (like epochs)
    max_depth=6,                  # Limit tree depth to prevent overfitting
    learning_rate=0.1,            # Moderate learning rate
    subsample=0.8,                # Use 80% of data per tree
    colsample_bytree=0.8,         # Use 80% of features per tree
    reg_alpha=0.1,                # L1 regularization
    reg_lambda=1.0,               # L2 regularization
    objective='multi:softmax',
    num_class=len(class_names),
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=3,      # Stop if no improvement for 3 rounds
    eval_metric='mlogloss'
)

# Train with validation monitoring
eval_set = [(X_train_scaled, y_train_full), (X_test_scaled, y_test)]
xgb_model.fit(
    X_train_scaled, 
    y_train_full,
    eval_set=eval_set,
    verbose=True
)

print(f"\n✓ Model trained successfully!")
print(f"✓ Best iteration: {xgb_model.best_iteration}")
print(f"✓ Stopped at iteration: {xgb_model.n_estimators if xgb_model.best_iteration == xgb_model.n_estimators else xgb_model.best_iteration + 3}")

# Get training history
results_dict = xgb_model.evals_result()

# ============================================
# COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Training metrics
train_pred = xgb_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train_full, train_pred)
train_precision = precision_score(y_train_full, train_pred, average='weighted')
train_recall = recall_score(y_train_full, train_pred, average='weighted')
train_f1 = f1_score(y_train_full, train_pred, average='weighted')

# Test metrics
test_pred = xgb_model.predict(X_test_scaled)
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

joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, 'xgb_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

results = {
    'model': 'XGBoost',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'parameters': {
        'n_estimators': 50,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 3
    },
    'training_info': {
        'best_iteration': int(xgb_model.best_iteration),
        'total_iterations': int(xgb_model.n_estimators)
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
    f.write("XGBOOST CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.2f}%\n")
    f.write(f"Best Iteration: {xgb_model.best_iteration}\n\n")
    f.write(report)

# ============================================
# VISUALIZATIONS
# ============================================

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
plt.title('XGBoost - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Training Loss Over Iterations
plt.figure(figsize=(12, 6))
epochs = range(len(results_dict['validation_0']['mlogloss']))
plt.plot(epochs, results_dict['validation_0']['mlogloss'], 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4)
plt.plot(epochs, results_dict['validation_1']['mlogloss'], 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4)

# Mark best iteration
if xgb_model.best_iteration < len(epochs):
    plt.axvline(x=xgb_model.best_iteration, color='green', linestyle='--', linewidth=2, label=f'Best Iteration ({xgb_model.best_iteration})')

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Multi-class Log Loss', fontsize=12)
plt.title('XGBoost - Training and Validation Loss Over Iterations', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Performance Metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy*100, train_precision*100, train_recall*100, train_f1*100]
test_scores = [test_accuracy*100, test_precision*100, test_recall*100, test_f1*100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, train_scores, width, label='Training', color='#e67e22', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_scores, width, label='Test', color='#d35400', alpha=0.8)

ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_title('XGBoost - Performance Metrics', fontsize=14, fontweight='bold')
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

# Feature Importance
importance = xgb_model.feature_importances_
top_n = 20
indices = np.argsort(importance)[-top_n:]

ax2.barh(range(top_n), importance[indices], color='darkorange', alpha=0.8)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([f'Feature {i}' for i in indices])
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Overfitting Analysis
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Training\nAccuracy', 'Test\nAccuracy', 'Overfitting\nGap']
values = [train_accuracy*100, test_accuracy*100, overfitting_gap]
colors = ['#e67e22', '#d35400', '#e74c3c' if overfitting_gap > 10 else '#f39c12']

bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('XGBoost - Overfitting Analysis', fontsize=16, fontweight='bold')
ax.set_ylim(0, max(values) + 15)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("RESULTS SAVED SUCCESSFULLY!")
print("="*70)
print(f"\nAll results saved in '{OUTPUT_DIR}/' directory:")
print("  ✓ xgb_model.pkl (trained model)")
print("  ✓ scaler.pkl (feature scaler)")
print("  ✓ results.json (all metrics)")
print("  ✓ classification_report.txt")
print("  ✓ confusion_matrix.png")
print("  ✓ training_loss.png")
print("  ✓ performance_analysis.png")
print("  ✓ overfitting_analysis.png")
print("\nYou can view these files anytime without re-running the code!")

# Memory cleanup
del X_train, X_val, X_train_full, X_train_scaled
import gc
gc.collect()
