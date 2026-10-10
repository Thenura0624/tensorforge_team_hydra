# Verification record

The following work is complete:

- Syntax checks on all three self-contained scripts.
- Identical split fingerprints and partition sizes verified across all three scripts.
- Feature construction verified to exclude ticket IDs and language tags.
- Spam and secondary-label consistency rules checked.
- Full SVM parameter search on the real training dataset: 18 primary configurations plus secondary/urgency parameter searches; complete search and refit took approximately 125 seconds on this host.
- Saved SVM candidate evaluated on the common 600-ticket comparison partition: 99.0% primary accuracy, macro F1 0.9892.
- Reloaded SVM exactly reproduced all 600 saved comparison predictions/probabilities. It also loaded through the original API's model adapter, with responses validated against the original prediction schema.
- Tiny random XLM-R and DeBERTa-v2 architectures tested through a real training epoch, backpropagation, validation selection, safetensors saving and offline inference. Both passed.
- A tokenizer compatibility issue found during testing was corrected for Transformers 5.
- Winner comparison and finalization tested end-to-end on a separate synthetic fixture, including fixed-choice enforcement and rejection of repeated finalization into the same directory. Fixture scores are not presented as real model results.

The SVM's 99.0% score comes from a new internal split of the training dataset. It is **not comparable** to the original 57.5% on the supplied validation set. The supplied validation file was not opened by the new real-data training/comparison run.

Not completed: pretrained XLM-R or mDeBERTa accuracy training, GPU execution/memory profiling, a real three-model winner selection, or the winner's final validation. The installed local PyTorch is CPU-only even though the RTX 4050 is detected by NVIDIA's driver. The package includes isolated CUDA installation instructions and fails clearly if neural training is started without CUDA.

The current SVM result folder is an actual trained candidate. It is not labeled as the overall best model. The other two results must be collected before the comparison command selects a winner.

## Follow-up: FP16 gradient-scaler fix

After CUDA-enabled PyTorch was installed, a user run exposed FP16 trainable gradients at `GradScaler.unscale_`. The loader now explicitly requests FP32 model weights and casts the complete network to FP32 before optimizer creation. A guard verifies the trainable parameter dtype. Forward computation still uses FP16 autocast and gradient scaling.

Regression tests now load tiny checkpoints stored in FP16 and train both architectures on CUDA, then verify FP32 saved weights and offline inference. Both passed. A further test used the actual cached pretrained mDeBERTa checkpoint on the RTX 4050: 32 training tickets, batch size 2, accumulation 8, two successful unscale calls with FP32 parameters and gradients. It completed without the reported exception. Peak allocated GPU memory for this short sample was approximately 2.05 GiB; this is not a worst-case memory or full accuracy benchmark.

The local environment now reports PyTorch 2.10.0+cu128 and CUDA available. Earlier CPU-only limitations above describe the original verification run, not this follow-up. Full neural accuracy experiments are still not complete.
