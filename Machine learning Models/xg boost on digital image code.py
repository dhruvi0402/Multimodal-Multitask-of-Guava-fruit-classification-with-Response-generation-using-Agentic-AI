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
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# ============================================
# CONFIGURATION - DIGITAL MATURITY
# ============================================
TRAIN_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split for digital\train'
VAL_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split for digital\val'
TEST_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split for digital\test'

IMG_SIZE = (128, 128)
OUTPUT_DIR = 'Digital_Maturity_XGBoost_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("XGBOOST - DIGITAL IMAGE MATURITY CLASSIFICATION")
print("="*80)
print(f"Task: Classify guava maturity (immature, mature, semi_mature)")
print(f"Image Type: Digital")
print(f"Model: XGBoost with Early Stopping")
print("="*80)

# ============================================
# ENHANCED FEATURE EXTRACTION
# ============================================
def extract_enhanced_features(image_path):
    """
    Extract HOG + Color Histogram + Texture features
    Optimized for digital images
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
    
    # 2. Color histogram features (RGB color distribution)
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
    
    # 4. Statistical features (color/intensity specific)
    statistical_features = [
        np.mean(img_gray),
        np.std(img_gray),
        np.median(img_gray),
        np.percentile(img_gray, 25),
        np.percentile(img_gray, 75)
    ]
    
    # 5. Color moments (for digital images)
    color_moments = []
    for i in range(3):
        channel = img_resized[:,:,i]
        color_moments.extend([
            np.mean(channel),
            np.std(channel),
            np.mean(np.abs(channel - np.mean(channel))**3)**(1/3)  # Skewness approximation
        ])
    
    # Combine all features
    combined_features = np.concatenate([
        hog_features, 
        hist_features, 
        texture_features,
        statistical_features,
        color_moments
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
print("LOADING DIGITAL MATURITY DATASET")
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
# FEATURE SCALING (IMPORTANT FOR XGBOOST)
# ============================================
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)
print("[OK] Features scaled using StandardScaler")

# ============================================
# TRAIN XGBOOST MODEL WITH EARLY STOPPING
# ============================================
print("\n" + "="*80)
print("TRAINING XGBOOST MODEL")
print("="*80)
print("\nModel Configuration:")
print("  - Max iterations (n_estimators): 50")
print("  - Early stopping patience: 3")
print("  - Max tree depth: 6 (prevents overfitting)")
print("  - Learning rate: 0.1")
print("  - Subsample ratio: 0.8")
print("  - L1 regularization (alpha): 0.1")
print("  - L2 regularization (lambda): 1.0")
print("\nStarting training...\n")

xgb_model = xgb.XGBClassifier(
    n_estimators=50,              # Maximum 50 iterations
    max_depth=6,                  # Limit tree depth
    learning_rate=0.1,            # Moderate learning rate
    subsample=0.8,                # Use 80% of data per tree
    colsample_bytree=0.8,         # Use 80% of features per tree
    reg_alpha=0.1,                # L1 regularization
    reg_lambda=1.0,               # L2 regularization
    objective='multi:softmax',    # Multi-class classification
    num_class=len(class_names),   # Number of classes
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

print(f"\n[SUCCESS] Model trained successfully!")
print(f"[INFO] Best iteration: {xgb_model.best_iteration}")
total_iterations = len(xgb_model.evals_result()['validation_0']['mlogloss'])
print(f"[INFO] Total iterations completed: {total_iterations}")

if xgb_model.best_iteration < total_iterations - 1:
    print(f"[INFO] Early stopping triggered at iteration {xgb_model.best_iteration + 3}")
else:
    print(f"[INFO] Training completed all {total_iterations} iterations")

# Get training history
results_dict = xgb_model.evals_result()

# ============================================
# COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

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

joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, 'xgb_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

results = {
    'model': 'XGBoost',
    'image_type': 'Digital',
    'task': 'Maturity Classification',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'classes': class_names,
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
        'total_iterations': total_iterations,
        'feature_dimension': int(X_train_full.shape[1])
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
    f.write("XGBOOST DIGITAL MATURITY CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Classes: {', '.join(class_names)}\n")
    f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.2f}%\n")
    f.write(f"Best Iteration: {xgb_model.best_iteration}\n\n")
    f.write(report)

print("[OK] Models saved: xgb_model.pkl, scaler.pkl")
print("[OK] Results saved: results.json, classification_report.txt")

# ============================================
# VISUALIZATIONS
# ============================================
print("\nGenerating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
plt.title('XGBoost - Digital Maturity Classification\nConfusion Matrix', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: confusion_matrix.png")

# 2. Training Loss Over Iterations
plt.figure(figsize=(12, 6))
epochs = range(len(results_dict['validation_0']['mlogloss']))
plt.plot(epochs, results_dict['validation_0']['mlogloss'], 
         'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4)
plt.plot(epochs, results_dict['validation_1']['mlogloss'], 
         'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4)

# Mark best iteration
if xgb_model.best_iteration < len(epochs):
    plt.axvline(x=xgb_model.best_iteration, color='green', linestyle='--', 
                linewidth=2.5, label=f'Best Iteration ({xgb_model.best_iteration})')

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Multi-class Log Loss', fontsize=12)
plt.title('XGBoost - Digital Maturity: Training and Validation Loss', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: training_loss.png")

# 3. Performance Metrics Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy*100, train_precision*100, train_recall*100, train_f1*100]
test_scores = [test_accuracy*100, test_precision*100, test_recall*100, test_f1*100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, train_scores, width, label='Training', 
                color='#e67e22', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, test_scores, width, label='Test', 
                color='#d35400', alpha=0.8, edgecolor='black')

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

# 4. Feature Importance (Top 30)
importance = xgb_model.feature_importances_
top_n = 30
indices = np.argsort(importance)[-top_n:]

plt.figure(figsize=(12, 10))
plt.barh(range(top_n), importance[indices], color='darkorange', alpha=0.8, edgecolor='black')
plt.yticks(range(top_n), [f'Feature {i}' for i in indices])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title(f'XGBoost - Digital Maturity: Top {top_n} Feature Importances', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: feature_importance.png")

# 5. Overfitting Analysis
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Training\nAccuracy', 'Test\nAccuracy', 'Overfitting\nGap']
values = [train_accuracy*100, test_accuracy*100, overfitting_gap]
colors_over = ['#e67e22', '#d35400', '#e74c3c' if overfitting_gap > 10 else '#f39c12']

bars = ax.bar(categories, values, color=colors_over, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('XGBoost - Digital Maturity: Overfitting Analysis', 
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
print("  [OK] xgb_model.pkl (trained model)")
print("  [OK] scaler.pkl (feature scaler)")
print("  [OK] results.json (all metrics)")
print("  [OK] classification_report.txt")
print("  [OK] confusion_matrix.png")
print("  [OK] training_loss.png")
print("  [OK] performance_metrics.png")
print("  [OK] feature_importance.png")
print("  [OK] overfitting_analysis.png")
print("  [OK] class_distribution.png")
print("\nYou can view these files anytime without re-running the code!")

print("\n" + "="*80)
print("KEY RESULTS SUMMARY")
print("="*80)
print(f"Model: XGBoost")
print(f"Image Type: Digital")
print(f"Task: Maturity Classification")
print(f"Classes: {', '.join(class_names)}")
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Overfitting Gap: {overfitting_gap:.2f}%")
print(f"Best Iteration: {xgb_model.best_iteration}/{total_iterations}")
print("="*80)

# Memory cleanup
del X_train, X_val, X_train_full, X_train_scaled
import gc
gc.collect()
print("\n[OK] Memory cleaned up")
print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80 + "\n")
