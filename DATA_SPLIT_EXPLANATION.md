# Data Split Explanation

## Dataset Partitioning

All datasets in the CoDyRA 5-task benchmark use proper train/val splits with large validation sets, ensuring meaningful evaluation metrics.

### Split Sizes

| Dataset | Train Samples | Val Samples | Split Ratio |
|---------|---------------|-------------|-------------|
| FGVC Aircraft | 6,667 | 3,333 | 67/33 |
| DTD | 3,760 | 1,880 | 67/33 |
| EuroSAT | 22,950 | 4,050 | 85/15 |
| Flowers102 | 2,040 | 6,149 | 25/75 |
| Oxford Pets | 6,282 | 1,108 | 85/15 |

### Key Properties

✅ **Zero overlap** between training and validation sets
✅ **Large validation sets** (1,000–6,000+ samples) for statistically meaningful metrics
✅ **Class-balanced splits** where possible
✅ **Consistent across runs** (deterministic splits from dataset-provided splits)

### Evaluation Method

Zero-shot classification accuracy is used as the primary metric:

1. Encode all class names using prompt templates → class text features
2. Encode each validation image → image features
3. Compute cosine similarity between image and all class features
4. Predict the class with highest similarity
5. Report top-1 accuracy

This is a **zero-shot** evaluation — no training is done on the validation set.

### Baseline Validation

| Dataset | Val Samples | Pretrained CLIP Accuracy | Expected Range |
|---------|-------------|------------------------|----------------|
| FGVC Aircraft | 3,333 | 23.97% | ✅ Expected (fine-grained, 100 classes) |
| DTD | 1,880 | 43.99% | ✅ Expected (textures are ambiguous) |
| EuroSAT | 4,050 | 46.86% | ✅ Expected (satellite is out-of-domain) |
| Flowers102 | 6,149 | 67.88% | ✅ Expected (natural images, CLIP knows flowers) |
| Oxford Pets | 1,108 | 87.27% | ✅ Expected (common animals, CLIP excels) |

All pretrained baselines are within expected ranges for CLIP ViT-B/16, confirming correct data loading and evaluation.
