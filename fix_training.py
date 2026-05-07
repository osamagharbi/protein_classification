#!/usr/bin/env python3
"""
Suggested fixes for the protein classification training issues.

This script provides solutions for:
1. Handling extreme class imbalance (1027:1 ratio)
2. Dealing with classes that have very few samples (some only 2 samples)
3. Working with classes that appear only in train (9 classes have no test samples)
4. Proper train/val split when stratification is difficult
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter

def compute_class_weights(label_dict, num_classes, device, method='inverse_sqrt'):
    """
    Compute class weights to handle extreme imbalance.

    Args:
        label_dict: Dictionary mapping sample_id -> class_id
        num_classes: Total number of classes
        device: torch device
        method: 'inverse' (1/count) or 'inverse_sqrt' (1/sqrt(count)) or 'effective_samples'

    Returns:
        torch.Tensor of shape (num_classes,) with weights
    """
    # Count samples per class
    class_counts = Counter(label_dict.values())

    # Create count array (handle missing classes with 0)
    counts = np.array([class_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)

    # Compute weights based on method
    if method == 'inverse':
        # Simple inverse frequency: w_i = 1 / count_i
        weights = 1.0 / (counts + 1e-6)  # Add epsilon to avoid division by zero
    elif method == 'inverse_sqrt':
        # Smoother inverse: w_i = 1 / sqrt(count_i)
        weights = 1.0 / (np.sqrt(counts) + 1e-6)
    elif method == 'effective_samples':
        # Effective number of samples (from Class-Balanced Loss paper)
        # More robust for extreme imbalance
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to num_classes (keeps loss scale similar)
    weights = weights * num_classes / weights.sum()

    # Handle classes with 0 samples (set weight to 0 so they don't contribute if seen)
    weights[counts == 0] = 0.0

    print(f"\nClass weights computed using '{method}' method:")
    print(f"  Min weight: {weights[counts > 0].min():.4f} (class with most samples)")
    print(f"  Max weight: {weights[counts > 0].max():.4f} (class with fewest samples)")
    print(f"  Mean weight: {weights[counts > 0].mean():.4f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_robust_train_val_split(dataset, label_dict, val_split=0.2, min_samples_for_stratify=10, seed=42):
    """
    Create train/val split that handles classes with very few samples.

    Strategy:
    - Classes with >= min_samples_for_stratify: Use stratified split
    - Classes with < min_samples_for_stratify: Put all in training set

    This ensures:
    - Rare classes are fully used for training (not wasted in tiny val sets)
    - Common classes are properly validated
    - No sklearn errors from impossible stratification

    Args:
        dataset: The full training dataset
        label_dict: Dictionary mapping sample_id -> class_id
        val_split: Fraction for validation (default 0.2)
        min_samples_for_stratify: Minimum samples needed to stratify a class
        seed: Random seed

    Returns:
        train_indices, val_indices (lists)
    """
    from sklearn.model_selection import train_test_split

    # Count samples per class
    class_counts = Counter(label_dict.values())

    # Get all sample IDs and their labels
    all_ids = list(label_dict.keys())
    all_labels = [label_dict[sid] for sid in all_ids]

    # Separate into stratifiable and non-stratifiable
    stratifiable_indices = []
    non_stratifiable_indices = []

    for idx, (sample_id, label) in enumerate(zip(all_ids, all_labels)):
        if class_counts[label] >= min_samples_for_stratify:
            stratifiable_indices.append(idx)
        else:
            non_stratifiable_indices.append(idx)

    print(f"\nTrain/Val Split Strategy:")
    print(f"  Stratifiable samples: {len(stratifiable_indices)} ({len(stratifiable_indices)/len(all_ids)*100:.1f}%)")
    print(f"  Non-stratifiable samples: {len(non_stratifiable_indices)} ({len(non_stratifiable_indices)/len(all_ids)*100:.1f}%)")
    print(f"  Rare classes (< {min_samples_for_stratify} samples) will be TRAIN-ONLY")

    # Split stratifiable samples
    if len(stratifiable_indices) > 0:
        strat_labels = [all_labels[i] for i in stratifiable_indices]
        strat_train_idx, strat_val_idx = train_test_split(
            stratifiable_indices,
            test_size=val_split,
            stratify=strat_labels,
            random_state=seed
        )
    else:
        strat_train_idx, strat_val_idx = [], []

    # All non-stratifiable go to training
    train_indices = strat_train_idx + non_stratifiable_indices
    val_indices = strat_val_idx

    print(f"\nFinal split:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples ({len(val_indices)/(len(train_indices)+len(val_indices))*100:.1f}%)")

    # Check class coverage
    train_classes = set([all_labels[i] for i in train_indices])
    val_classes = set([all_labels[i] for i in val_indices])
    print(f"  Train classes: {len(train_classes)}")
    print(f"  Val classes: {len(val_classes)}")
    print(f"  Classes only in train: {len(train_classes - val_classes)}")

    return train_indices, val_indices


def focal_loss(outputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    """
    Focal Loss for handling class imbalance.

    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard examples and down-weights easy examples.
    Particularly effective for extreme class imbalance.

    Args:
        outputs: Model logits, shape (batch_size, num_classes)
        targets: Ground truth labels, shape (batch_size,)
        alpha: Per-class weights, shape (num_classes,) or None
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar tensor (if reduction != 'none')
    """
    ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none', weight=alpha)
    p_t = torch.exp(-ce_loss)  # Probability of correct class
    focal_weight = (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


class FocalLoss(nn.Module):
    """Focal Loss as a nn.Module for easy use."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        return focal_loss(outputs, targets, self.alpha, self.gamma, self.reduction)


def suggest_modifications():
    """Print suggested code modifications for classify_gnn.py"""

    print("="*80)
    print("SUGGESTED CODE MODIFICATIONS FOR classify_gnn.py")
    print("="*80)

    print("\n1. IMPORT THIS MODULE at the top:")
    print("-" * 80)
    print("from fix_training import (")
    print("    compute_class_weights,")
    print("    create_robust_train_val_split,")
    print("    FocalLoss")
    print(")")

    print("\n2. REPLACE train_test_split (around line 357-368) with:")
    print("-" * 80)
    print("""
# Create label dictionary for full train set
full_train_dict = {}
for i in range(len(train_dataset)):
    sample_id = list(train_dataset.label_dict.keys())[i]
    full_train_dict[sample_id] = train_dataset.label_dict[sample_id]

# Robust split that handles rare classes
train_indices, val_indices = create_robust_train_val_split(
    train_dataset,
    full_train_dict,
    val_split=args.val_split,
    min_samples_for_stratify=10,  # Classes with <10 samples go only to train
    seed=args.seed
)

# Create subset datasets
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)
""")

    print("\n3. ADD class weights computation (after creating train_subset, around line 375):")
    print("-" * 80)
    print("""
# Compute class weights for imbalanced dataset
train_label_dict = {list(train_dataset.label_dict.keys())[i]: list(train_dataset.label_dict.values())[i]
                    for i in train_indices}
class_weights = compute_class_weights(
    train_label_dict,
    num_classes,
    device,
    method='inverse_sqrt'  # Options: 'inverse', 'inverse_sqrt', 'effective_samples'
)
""")

    print("\n4. REPLACE loss function (around line 407-408) with:")
    print("-" * 80)
    print("""
# Option A: Weighted Cross-Entropy (simpler, usually sufficient)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option B: Focal Loss (better for extreme imbalance, may need tuning)
# criterion = FocalLoss(alpha=class_weights, gamma=2.0)
""")

    print("\n5. ADD evaluation metrics for imbalanced classes (optional enhancement):")
    print("-" * 80)
    print("""
# In compute_metrics function, add confusion matrix analysis
from sklearn.metrics import confusion_matrix

def analyze_predictions(y_true, y_pred, num_classes):
    \"\"\"Analyze which classes are being predicted vs actual.\"\"\"
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Find classes that are never predicted
    never_predicted = np.where(cm.sum(axis=0) == 0)[0]
    if len(never_predicted) > 0:
        print(f"⚠️  Classes never predicted: {never_predicted.tolist()}")

    # Find classes where model always predicts wrong
    always_wrong = np.where(np.diag(cm) == 0)[0]
    if len(always_wrong) > 0:
        print(f"⚠️  Classes always wrong: {always_wrong.tolist()}")

    return cm
""")

    print("\n6. ADJUST learning rate and regularization:")
    print("-" * 80)
    print("""
# Recommended hyperparameters for imbalanced dataset:
# --lr 0.0005              (Lower learning rate, was 0.001)
# --weight-decay 0.0001    (Keep moderate regularization)
# --dropout 0.4            (Increase dropout, was 0.3)
# --epochs 100             (May need fewer epochs with weights)
# --batch-size 32          (Larger batches for stability, was 16)
""")

    print("\n7. ADD early stopping to prevent overfitting:")
    print("-" * 80)
    print("""
# Add after line 292 in main()
early_stopping_patience = 20
epochs_without_improvement = 0
best_val_f1 = 0.0

# In training loop, after validation:
if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    epochs_without_improvement = 0
    # Save best model...
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= early_stopping_patience:
        print(f"\\nEarly stopping at epoch {epoch+1}")
        print(f"No improvement for {early_stopping_patience} epochs")
        break
""")

    print("\n" + "="*80)
    print("SUMMARY OF CHANGES")
    print("="*80)
    print("""
These modifications will:
✅ Handle extreme class imbalance (1027:1 ratio) with class weights
✅ Deal with rare classes (2-5 samples) by keeping them in training only
✅ Prevent stratification errors
✅ Add early stopping to prevent overfitting
✅ Use focal loss or weighted cross-entropy
✅ Monitor which classes are being predicted

Expected improvements:
- No more "single label" warnings
- Better generalization to test set
- More balanced predictions across classes
- Proper handling of the 9 train-only classes
""")

    print("\n" + "="*80)
    print("RECOMMENDED COMMAND TO RUN")
    print("="*80)
    print("""
python classify_gnn.py \\
    --device 0 \\
    --model gcn \\
    --hidden-dim 256 \\
    --num-layers 3 \\
    --lr 0.0005 \\
    --weight-decay 0.0001 \\
    --dropout 0.4 \\
    --batch-size 32 \\
    --epochs 100 \\
    --val-split 0.15
""")


if __name__ == '__main__':
    suggest_modifications()
