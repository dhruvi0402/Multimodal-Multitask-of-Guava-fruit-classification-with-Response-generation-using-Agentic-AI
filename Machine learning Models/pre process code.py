import os
import shutil
from collections import defaultdict
import random
import sys

# Force UTF-8 encoding for Windows
os.environ['PYTHONUTF8'] = '1'

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================
# Source directories (your original unsplit data)
DIGITAL_SOURCE = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\digital_organized'
THERMAL_SOURCE = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\thermal_organized'

# Output directories (will be created automatically)
DIGITAL_OUTPUT = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split for digital'
THERMAL_OUTPUT = r'C:\Users\SATYABRATA MISHRA\Downloads\Final preprocessed data\Maturity\train test val split  thermal'

# Split ratios (70% train, 15% validation, 15% test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

print("="*80)
print("DATASET SPLITTING FOR GUAVA MATURITY CLASSIFICATION")
print("="*80)
print(f"\nSplit Ratios: Train={TRAIN_RATIO*100:.0f}% | Val={VAL_RATIO*100:.0f}% | Test={TEST_RATIO*100:.0f}%")
print(f"Random Seed: {RANDOM_SEED}")

# ============================================
# SPLIT FUNCTION
# ============================================
def split_dataset(source_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into train/val/test while maintaining class distribution
    
    Args:
        source_dir: Path to source directory with class folders
        output_dir: Path to output directory
        train_ratio: Proportion for training set (default 0.70)
        val_ratio: Proportion for validation set (default 0.15)
        test_ratio: Proportion for test set (default 0.15)
        random_seed: Random seed for reproducibility
    
    Returns:
        class_stats: Per-class statistics
        total_stats: Overall statistics
    """
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        print(f"\n[ERROR] Source directory not found: {source_dir}")
        return None, None
    
    # Get class folders (e.g., immature, mature, semi_mature)
    classes = sorted([d for d in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('.')])
    
    if not classes:
        print(f"\n[ERROR] No class folders found in {source_dir}")
        return None, None
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(source_dir).upper()} IMAGES")
    print(f"{'='*80}")
    print(f"Classes found: {classes}")
    
    # Validate split ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.001:
        print(f"\n[WARNING] Split ratios don't sum to 1.0")
        print(f"Adjusting ratios to sum to 1.0")
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            output_path = os.path.join(output_dir, split, class_name)
            os.makedirs(output_path, exist_ok=True)
    
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    class_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
    
    # Process each class
    print(f"\nSplitting images...")
    print("-"*80)
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        
        # Get all image files
        all_files = os.listdir(class_path)
        images = [f for f in all_files 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if not images:
            print(f"[WARNING] {class_name:15s}: No images found - SKIPPED")
            continue
        
        # Shuffle images with fixed seed for reproducibility
        random.seed(random_seed)
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Assign remaining to test
        
        # Split images into three sets
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy files to respective directories
        print(f"\nClass: {class_name}")
        print(f"  Total: {n_total} images")
        
        # Copy training images
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, 'train', class_name, img)
            shutil.copy2(src, dst)
        print(f"  [OK] Train: {len(train_images):4d} images ({len(train_images)/n_total*100:5.1f}%)")
        
        # Copy validation images
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, 'val', class_name, img)
            shutil.copy2(src, dst)
        print(f"  [OK] Val:   {len(val_images):4d} images ({len(val_images)/n_total*100:5.1f}%)")
        
        # Copy test images
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, 'test', class_name, img)
            shutil.copy2(src, dst)
        print(f"  [OK] Test:  {len(test_images):4d} images ({len(test_images)/n_total*100:5.1f}%)")
        
        # Update statistics
        class_stats[class_name]['train'] = len(train_images)
        class_stats[class_name]['val'] = len(val_images)
        class_stats[class_name]['test'] = len(test_images)
        class_stats[class_name]['total'] = n_total
        
        total_stats['train'] += len(train_images)
        total_stats['val'] += len(val_images)
        total_stats['test'] += len(test_images)
    
    # Print summary
    print(f"\n{'-'*80}")
    print(f"[SUCCESS] Split completed for {os.path.basename(source_dir).upper()}")
    print(f"\nSUMMARY:")
    print(f"  Train: {total_stats['train']:5d} images ({total_stats['train']/(total_stats['train']+total_stats['val']+total_stats['test'])*100:.1f}%)")
    print(f"  Val:   {total_stats['val']:5d} images ({total_stats['val']/(total_stats['train']+total_stats['val']+total_stats['test'])*100:.1f}%)")
    print(f"  Test:  {total_stats['test']:5d} images ({total_stats['test']/(total_stats['train']+total_stats['val']+total_stats['test'])*100:.1f}%)")
    print(f"  Total: {sum(total_stats.values()):5d} images")
    
    return class_stats, total_stats

# ============================================
# SPLIT DIGITAL DATASET
# ============================================
print(f"\n{'#'*80}")
print("STEP 1: SPLITTING DIGITAL IMAGES")
print(f"{'#'*80}")

digital_class_stats, digital_total_stats = split_dataset(
    source_dir=DIGITAL_SOURCE,
    output_dir=DIGITAL_OUTPUT,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED
)

# ============================================
# SPLIT THERMAL DATASET
# ============================================
print(f"\n{'#'*80}")
print("STEP 2: SPLITTING THERMAL IMAGES")
print(f"{'#'*80}")

thermal_class_stats, thermal_total_stats = split_dataset(
    source_dir=THERMAL_SOURCE,
    output_dir=THERMAL_OUTPUT,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED
)

# ============================================
# FINAL COMPREHENSIVE SUMMARY
# ============================================
print(f"\n{'='*80}")
print("FINAL SUMMARY - ALL DATASETS")
print(f"{'='*80}")

if digital_total_stats and thermal_total_stats:
    print("\nDIGITAL IMAGES:")
    print(f"  Train: {digital_total_stats['train']:5d} images")
    print(f"  Val:   {digital_total_stats['val']:5d} images")
    print(f"  Test:  {digital_total_stats['test']:5d} images")
    print(f"  Total: {sum(digital_total_stats.values()):5d} images")
    
    print("\nTHERMAL IMAGES:")
    print(f"  Train: {thermal_total_stats['train']:5d} images")
    print(f"  Val:   {thermal_total_stats['val']:5d} images")
    print(f"  Test:  {thermal_total_stats['test']:5d} images")
    print(f"  Total: {sum(thermal_total_stats.values()):5d} images")
    
    print(f"\n{'='*80}")
    print("[SUCCESS] DATASET SPLITTING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    
    print(f"\nOutput Locations:")
    print(f"  Digital: {DIGITAL_OUTPUT}")
    print(f"  Thermal: {THERMAL_OUTPUT}")
    
    print(f"\nDirectory Structure Created:")
    print(f"  Maturity_Split/")
    print(f"  ├── Digital/")
    print(f"  │   ├── train/")
    print(f"  │   │   ├── immature/")
    print(f"  │   │   ├── mature/")
    print(f"  │   │   └── semi_mature/")
    print(f"  │   ├── val/")
    print(f"  │   │   ├── immature/")
    print(f"  │   │   ├── mature/")
    print(f"  │   │   └── semi_mature/")
    print(f"  │   └── test/")
    print(f"  │       ├── immature/")
    print(f"  │       ├── mature/")
    print(f"  │       └── semi_mature/")
    print(f"  └── Thermal/")
    print(f"      └── (same structure)")
    
    # Per-class breakdown
    if digital_class_stats:
        print(f"\n{'='*80}")
        print("PER-CLASS BREAKDOWN - DIGITAL")
        print(f"{'='*80}")
        print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
        print("-"*80)
        for class_name in sorted(digital_class_stats.keys()):
            stats = digital_class_stats[class_name]
            print(f"{class_name:<15} {stats['train']:8d} {stats['val']:8d} {stats['test']:8d} {stats['total']:8d}")
    
    if thermal_class_stats:
        print(f"\n{'='*80}")
        print("PER-CLASS BREAKDOWN - THERMAL")
        print(f"{'='*80}")
        print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
        print("-"*80)
        for class_name in sorted(thermal_class_stats.keys()):
            stats = thermal_class_stats[class_name]
            print(f"{class_name:<15} {stats['train']:8d} {stats['val']:8d} {stats['test']:8d} {stats['total']:8d}")
    
    print(f"\n{'='*80}")
    print("[SUCCESS] Ready for Model Training!")
    print(f"{'='*80}")
    print("\nNext Steps:")
    print("  1. Verify the split directories contain correct images")
    print("  2. Run SVM, Random Forest, XGBoost, and KNN models")
    print("  3. Compare Digital vs Thermal classification performance")
    
elif digital_total_stats is None and thermal_total_stats is None:
    print("\n[ERROR] Both dataset splits failed. Please check:")
    print("  1. Source directory paths are correct")
    print("  2. Class folders exist in source directories")
    print("  3. Images are present in class folders")
    print("  4. You have write permissions for output directories")

elif digital_total_stats is None:
    print("\n[ERROR] Digital dataset split failed")
    print("[SUCCESS] Thermal dataset split completed")
    
elif thermal_total_stats is None:
    print("\n[SUCCESS] Digital dataset split completed")
    print("[ERROR] Thermal dataset split failed")

print(f"\n{'='*80}")
print("SCRIPT COMPLETED")
print(f"{'='*80}\n")
