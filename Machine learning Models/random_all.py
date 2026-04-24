import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
OUTPUT_DIR = 'SVM_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("SVM IMAGE CLASSIFICATION - GUAVA DISEASE DETECTION")
print("="*60)

# ============================================
# FEATURE EXTRACTION
# ============================================
def extract_features(image_path):
    """Extract HOG + Color Histogram features"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize and convert
    img_resized = cv2.resize(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # HOG features
    hog_features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    # Color histogram features
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
        hist_features.extend(hist.flatten())
    
    # Combine features
    combined_features = np.concatenate([hog_features, hist_features])
    return combined_features

def load_dataset(directory, dataset_name):
    """Load images and extract features"""
    X, y = [], []
    class_names = sorted(os.listdir(directory))
    
    print(f"\nLoading {dataset_name} data...")
    for label_idx, label in enumerate(class_names):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
            
        image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {label}: {len(image_files)} images")
        
        for filename in image_files:
            image_path = os.path.join(label_dir, filename)
            features = extract_features(image_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)
    
    return np.array(X), np.array(y), class_names

# Load datasets
X_train, y_train, class_names = load_dataset(TRAIN_DIR, "Training")
X_val, y_val, _ = load_dataset(VAL_DIR, "Validation")
X_test, y_test, _ = load_dataset(TEST_DIR, "Test")

# Combine train and validation for better model
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

print(f"\nDataset Summary:")
print(f"  Training samples: {len(X_train_full)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Classes: {class_names}")
print(f"  Feature dimension: {X_train_full.shape[1]}")

# ============================================
# FEATURE SCALING
# ============================================
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# ============================================
# HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================
print("\nPerforming Grid Search for best hyperparameters...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly']
}

svm_base = SVC(random_state=42)
grid_search = GridSearchCV(
    svm_base,
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train_full)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# ============================================
# TRAIN FINAL MODEL
# ============================================
print("\nTraining final SVM model with best parameters...")
best_svm = grid_search.best_estimator_

# ============================================
# EVALUATE MODEL
# ============================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Training accuracy
train_pred = best_svm.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train_full, train_pred)

# Test accuracy
test_pred = best_svm.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Classification report
print("\n" + "-"*60)
print("CLASSIFICATION REPORT")
print("-"*60)
report = classification_report(y_test, test_pred, target_names=class_names)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)

# ============================================
# SAVE RESULTS PERMANENTLY
# ============================================
print("\nSaving results...")

# Save model and scaler
joblib.dump(best_svm, os.path.join(OUTPUT_DIR, 'svm_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

# Save metrics as JSON
results = {
    'model': 'Support Vector Machine (SVM)',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'best_parameters': grid_search.best_params_,
    'training_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'classes': class_names,
    'feature_dimension': int(X_train_full.shape[1]),
    'training_samples': int(len(X_train_full)),
    'test_samples': int(len(X_test)),
    'classification_report': report
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Save classification report as text
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("SVM CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n\n")
    f.write(report)

# ============================================
# PLOT AND SAVE CONFUSION MATRIX
# ============================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('SVM - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# PLOT AND SAVE ACCURACY COMPARISON
# ============================================
plt.figure(figsize=(10, 6))
accuracies = [train_accuracy*100, test_accuracy*100]
labels = ['Training', 'Test']
colors = ['#2ecc71', '#3498db']

bars = plt.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylim(0, 105)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('SVM - Training vs Test Accuracy', fontsize=16, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RESULTS SAVED SUCCESSFULLY!")
print("="*60)
print(f"\nAll results saved in '{OUTPUT_DIR}/' directory:")
print("  - svm_model.pkl (trained model)")
print("  - scaler.pkl (feature scaler)")
print("  - results.json (all metrics)")
print("  - classification_report.txt")
print("  - confusion_matrix.png")
print("  - accuracy_comparison.png")
print("\nYou can view these files anytime without re-running the code!")
