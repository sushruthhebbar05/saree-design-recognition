# Evaluation Protocol

## Design-level Splits
All splits are based on `design_id`, not individual images. This prevents trivial retrieval where the same design appears in multiple splits.

### Split Strategy
- **Train designs:** 60% of unique design IDs
- **Validation designs:** 20% of unique design IDs
- **Gallery/Query designs:** 20% of unique design IDs
  - Query: one image per design (held-out)
  - Gallery: remaining images from same designs

## Retrieval Evaluation (Protocol A)
- Input: query image from held-out split
- Gallery: reference images from known designs
- Output: ranked list of gallery images
- Metric: design match (any gallery image with correct design_id counts as hit)
- Report: Top-1, Top-5, Top-10 accuracy; MRR

## Verification Evaluation (Protocol B)
- Input: pairs of images
- Task: predict same_design vs different_design
- Threshold: selected on validation pairs only (no test leakage)
- Metrics: ROC-AUC, PR-AUC, F1, EER, accuracy, precision, recall

## Color Invariance Analysis
- Original image → embedding E_0
- Grayscale version → embedding E_gray
- Hue shift → embedding E_hue
- Saturation shift → embedding E_sat
- Brightness change → embedding E_bright
- Score = mean(cosine_sim(E_0, [E_gray, E_hue, E_sat, E_bright]))
- Higher score indicates better color invariance

## Important Rules
1. Never tune thresholds on test set
2. Design IDs must be consistent across splits
3. If colorway labels are unavailable, document approximations
4. Report only metrics that have been computed (never fabricate)
5. Use deterministic splits (fixed random seed)
