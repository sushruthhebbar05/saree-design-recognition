# Approach Note

## Architecture
RGB textile crops are encoded by an ImageNet-initialized ResNet-18 backbone followed by a 256-dimensional projection head with L2 normalization. The embedding represents design identity independent of color.

## Why Metric Learning?
The task is retrieval with potentially changing gallery identities. Closed-set classification is inappropriate because:
1. New designs may appear in the gallery
2. The goal is ranking by similarity, not assigning fixed labels
3. Metric learning naturally generalizes to unknown designs

## Color Invariance Strategy
Strong color augmentation (ColorJitter, RandomGrayscale, GaussianBlur, affine transforms) creates different color palettes from the same motif. The supervised contrastive loss pulls same-design views (regardless of color) together and pushes different designs apart.

## Loss Function
Supervised Contrastive Loss (SupCon) with temperature τ=0.07:
- Positive pairs: same design across color variants
- Negative pairs: different designs
- Encourages tight clustering of same-design embeddings

## Sampling
Mixed strategy: combines random negatives with same-color-family negatives to force the model to distinguish designs that might have similar palettes.

## Preprocessing
- Resize/crop to 224×224
- ImageNet normalization
- Train: RandomResizedCrop + aggressive augmentation
- Eval: center resize + normalize

## Evaluation
- Top-K retrieval: measure rank of correct design in sorted gallery
- Verification: ROC-AUC, PR-AUC, F1, EER on design pairs
- Color Invariance: embedding stability under color transforms
- Leakage-free splits by design identity
