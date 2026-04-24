import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# CONFIGURATION - THERMAL MATURITY
# ============================================
TRAIN_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Diseases\train'
VAL_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Diseases\val'
TEST_DIR = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Diseases\test'

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 3

OUTPUT_DIR = 'Thermal_Maturity_CNN_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("CNN - THERMAL IMAGE MATURITY CLASSIFICATION")
print("="*80)
print(f"Task: Classify guava maturity (immature, mature, semi_mature)")
print(f"Image Type: Thermal")
print(f"Model: Convolutional Neural Network (CNN)")
print(f"Framework: TensorFlow/Keras")
print("="*80)

# ============================================
# DATA GENERATORS WITH AUGMENTATION
# ============================================
print("\n" + "-"*80)
print("SETTING UP DATA GENERATORS")
print("-"*80)

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Validation and test data (no augmentation, only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

# Load validation data
val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load test data
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"\n[INFO] Classes detected: {class_names}")
print(f"[INFO] Number of classes: {num_classes}")
print(f"[INFO] Training samples: {train_generator.samples}")
print(f"[INFO] Validation samples: {val_generator.samples}")
print(f"[INFO] Test samples: {test_generator.samples}")

# ============================================
# BUILD CNN MODEL
# ============================================
print("\n" + "="*80)
print("BUILDING CNN MODEL")
print("="*80)

model = models.Sequential([
    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Convolutional Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[INFO] Model Architecture:")
model.summary()

# ============================================
# CALLBACKS
# ============================================
print("\n" + "-"*80)
print("SETTING UP CALLBACKS")
print("-"*80)

# Early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint - FIXED: Changed to .keras format
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(OUTPUT_DIR, 'best_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Reduce learning rate on plateau
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

print(f"[INFO] Early Stopping: patience={EARLY_STOPPING_PATIENCE}")
print(f"[INFO] Learning Rate Reduction: factor=0.5, patience=3")

# ============================================
# TRAIN MODEL
# ============================================
print("\n" + "="*80)
print("TRAINING CNN MODEL")
print("="*80)
print(f"\nConfiguration:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Image size: {IMG_SIZE}")
print(f"  - Optimizer: Adam (lr=0.001)")
print(f"  - Data augmentation: Enabled")
print("\nStarting training...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

print("\n[SUCCESS] Model trained successfully!")

# ============================================
# EVALUATE MODEL
# ============================================
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

# Get predictions
print("\n[INFO] Generating predictions...")
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Calculate metrics
from sklearn.metrics import precision_score, recall_score, f1_score

test_precision = precision_score(y_true, y_pred, average='weighted')
test_recall = recall_score(y_true, y_pred, average='weighted')
test_f1 = f1_score(y_true, y_pred, average='weighted')

# Training accuracy (from final epoch)
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

# Overfitting gap
overfitting_gap = (train_accuracy - test_accuracy) * 100

print(f"\n{'='*80}")
print(f"FINAL RESULTS")
print(f"{'='*80}")
print(f"\nTRAINING METRICS (Final Epoch):")
print(f"  Accuracy: {train_accuracy*100:.2f}%")

print(f"\nVALIDATION METRICS (Final Epoch):")
print(f"  Accuracy: {val_accuracy*100:.2f}%")

print(f"\nTEST METRICS:")
print(f"  Accuracy:  {test_accuracy*100:.2f}%")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  Loss:      {test_loss:.4f}")

print(f"\nOVERFITTING ANALYSIS:")
print(f"  Gap: {overfitting_gap:.2f}%", end=' ')
if overfitting_gap < 5:
    print("(Excellent - No overfitting)")
elif overfitting_gap < 10:
    print("(Good - Minimal overfitting)")
elif overfitting_gap < 15:
    print("(Moderate overfitting)")
else:
    print("(Severe overfitting)")

# Classification report
print("\n" + "-"*80)
print("DETAILED CLASSIFICATION REPORT")
print("-"*80)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

cm = confusion_matrix(y_true, y_pred)

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "-"*80)
print("SAVING RESULTS")
print("-"*80)

# Save final model - FIXED: Changed to .keras format
model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))

# Save training history
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    json.dump(history_dict, f, indent=4)

# Save results
results = {
    'model': 'Convolutional Neural Network (CNN)',
    'image_type': 'Thermal',
    'task': 'Maturity Classification',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'classes': class_names,
    'architecture': {
        'total_params': int(model.count_params()),
        'trainable_params': int(sum([tf.size(w).numpy() for w in model.trainable_weights])),
        'layers': len(model.layers),
        'epochs_trained': len(history.history['loss']),
        'batch_size': BATCH_SIZE,
        'image_size': IMG_SIZE
    },
    'training_metrics': {
        'final_accuracy': float(train_accuracy),
        'final_loss': float(history.history['loss'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss']))
    },
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'loss': float(test_loss)
    },
    'overfitting_gap_percent': float(overfitting_gap),
    'classification_report': report
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("CNN THERMAL MATURITY CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Classes: {', '.join(class_names)}\n")
    f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
    f.write(f"Validation Accuracy: {val_accuracy*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.2f}%\n")
    f.write(f"Total Epochs: {len(history.history['loss'])}\n\n")
    f.write(report)

print("[OK] Models saved: best_model.keras, final_model.keras")
print("[OK] Results saved: results.json, classification_report.txt")
print("[OK] Training history saved: training_history.json")

# ============================================
# VISUALIZATIONS
# ============================================
print("\nGenerating visualizations...")

# 1. Training History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
epochs_range = range(1, len(history.history['accuracy']) + 1)
ax1.plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('CNN - Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Loss
ax2.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('CNN - Training and Validation Loss', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: training_history.png")

# 2. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
plt.title('CNN - Thermal Maturity Classification\nConfusion Matrix', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: confusion_matrix.png")

# 3. Performance Metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
test_scores = [test_accuracy*100, test_precision*100, test_recall*100, test_f1*100]

bars = ax1.bar(metrics, test_scores, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], 
               alpha=0.8, edgecolor='black')
ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_title('CNN - Test Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, test_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Per-class accuracy
per_class_acc = []
for i in range(len(class_names)):
    class_correct = cm[i, i]
    class_total = np.sum(cm[i, :])
    class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
    per_class_acc.append(class_acc)

colors_per_class = ['#e74c3c', '#f39c12', '#27ae60']
bars2 = ax2.bar(class_names, per_class_acc, color=colors_per_class, 
                alpha=0.8, edgecolor='black')
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('CNN - Per-Class Accuracy', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars2, per_class_acc):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: performance_metrics.png")

# 4. Overfitting Analysis
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Training\nAccuracy', 'Validation\nAccuracy', 'Test\nAccuracy', 'Overfitting\nGap']
values = [train_accuracy*100, val_accuracy*100, test_accuracy*100, overfitting_gap]
colors_over = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c' if overfitting_gap > 10 else '#f39c12']

bars = ax.bar(categories, values, color=colors_over, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('CNN - Thermal Maturity: Overfitting Analysis', 
             fontsize=16, fontweight='bold')
ax.set_ylim(0, max(values) + 15)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: overfitting_analysis.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*80)
print("RESULTS SAVED SUCCESSFULLY!")
print("="*80)
print(f"\nAll results saved in '{OUTPUT_DIR}/' directory:")
print("  [OK] best_model.keras (best model during training)")
print("  [OK] final_model.keras (final trained model)")
print("  [OK] results.json (all metrics)")
print("  [OK] training_history.json (epoch-by-epoch history)")
print("  [OK] classification_report.txt")
print("  [OK] training_history.png")
print("  [OK] confusion_matrix.png")
print("  [OK] performance_metrics.png")
print("  [OK] overfitting_analysis.png")

print("\n" + "="*80)
print("KEY RESULTS SUMMARY")
print("="*80)
print(f"Model: Convolutional Neural Network (CNN)")
print(f"Image Type: Thermal")
print(f"Task: Maturity Classification")
print(f"Classes: {', '.join(class_names)}")
print(f"\nTotal Parameters: {model.count_params():,}")
print(f"Epochs Trained: {len(history.history['loss'])}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Overfitting Gap: {overfitting_gap:.2f}%")
print("="*80)

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80 + "\n")
