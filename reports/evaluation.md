# Baseline evaluation

Model: `tf-baseline-399b28d752c536d0`. Training: 4,000 tickets. Held-out validation: 800 tickets. No validation examples were used for fitting, vocabulary construction, or parameter selection. Exact combined-input overlap between train and validation is zero; synthetic template or semantic overlap has not been ruled out.

| Target | Accuracy | Macro F1 |
|---|---:|---:|
| Primary category | 57.50% | 0.6045 |
| Secondary category including none | 94.25% | 0.5412 |
| Urgency | 97.375% | 0.9194 |

All three outputs are correct together on 52.875% of validation tickets. Primary multiclass log loss is 1.5096. Confidence is the logistic-regression probability, not a separately calibrated estimate.

## Primary accuracy by language

| Language | Tickets | Accuracy | Macro F1 |
|---|---:|---:|---:|
| English | 280 | 45.36% | 0.5094 |
| Sinhala | 160 | 47.50% | 0.4699 |
| Tamil | 120 | 55.83% | 0.5267 |
| Singlish | 200 | 83.50% | 0.8318 |
| Tanglish | 20 | 65.00% | 0.5552 |
| Mixed | 20 | 50.00% | 0.3394 |

Language is used only for this breakdown; the API contract does not supply it. Small groups have unstable estimates. Per-group macro F1 is computed over labels present in that group's truth or predictions.

## Interpretation

This is a working baseline, not a production-quality routing model. Primary classification remains the main weakness. Food quality and missing/wrong orders have low recall. The full class-level report and confusion matrix are in `metrics.json`.

Secondary labels are absent in 720 of 800 validation tickets. Always predicting none would achieve 90% accuracy. The model gets 94.25%, but misses 46 of the 80 actual secondary labels; secondary delivery-delay recall is zero. Its aggregate accuracy must not be read as reliable detection of all secondary issues.

Urgency precision is 96.83% and recall is 76.25% (61 of 80 urgent examples detected). Nineteen urgent cases were missed. This baseline should not autonomously handle safety-critical triage. The always-nonurgent accuracy baseline is 90%.

The majority primary category is payment_refund (18% of validation tickets), so the trained model improves substantially over that baseline. All data is synthetic and the dataset notes state Sinhala/Tamil text was not independently reviewed by native speakers; these scores do not establish real-world accuracy.

## Reproducibility and next experiments

The pipeline has 61,893 TF-IDF features and three logistic-regression classifiers with C=4.0, maximum 1,000 iterations and seed 42. Character features preserve Unicode; preprocessing does not strip Sinhala/Tamil combining marks. Model training converged in 60, 35 and 22 iterations for primary, secondary and urgency respectively. Class weighting is applied only to urgency. The model is approximately 6.3 MB on disk.

For subsequent improvement, use internal training folds to compare class weighting, word/character weights and regularization, then calibrate confidence on training-only folds. Evaluate recall for rare secondary categories and urgency, and inspect errors involving quoted history or conflicting issues. Reserve a new untouched test split if this public validation set is used for iterative tuning.
