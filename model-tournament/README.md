# Three-computer model comparison

**There is no confirmed overall winner yet.** Run the three candidates, compare their results, then retrain and verify the selected model on one computer. XLM-R and mDeBERTa are strong multilingual candidates; neither is guaranteed to beat the SVM on these tickets.

Your hardware: each computer has a Ryzen 7, 16 GB RAM and RTX 4050 with 6 GB VRAM. The neural defaults are designed to reduce memory use on that hardware, but full-size GPU training and peak VRAM have not been measured here.

There are exactly three standalone training scripts. Each contains the shared comparison, final training and inference commands, so no fourth Python script is needed.

| Computer | Script | Model | Why test it? |
|---|---|---|---|
| 1 | `computer_1_xlmr.py` | `FacebookAI/xlm-roberta-base`, fine-tuned encoder with three heads | Multilingual context and cross-language transfer |
| 2 | `computer_2_mdeberta.py` | `microsoft/mdeberta-v3-base`, fine-tuned encoder with three heads | Alternative multilingual architecture; a credible challenger |
| 3 | `computer_3_linear_svm.py` | Word + character TF-IDF and Linear SVM | Fast lexical baseline, transliteration features and inexpensive CPU inference |

The third computer does not need to use its GPU. A different model family is a more useful comparison than duplicating the same GPU run. The three GPUs work independently; their VRAM is not pooled into an 18 GB GPU.

## 1. Copy and install

Extract the package on all three computers. Keep `data/train.jsonl` and `data/validation.jsonl` unchanged. The input ZIPs do not need to be extracted again; their original JSONL data is included.

In PowerShell, open the extracted `model-tournament` directory. With Python 3.14 installed:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

Python 3.12 or 3.13 can also be used by changing the launcher version; the executed local checks used Python 3.14.6. On this machine the plain `python` command points to a different MSYS installation, so prefer the explicit virtual-environment interpreter shown below.

On **computers 1 and 2**:

```powershell
py -3.14 -m venv .venv
```

The CUDA check must print `True`. The existing installation on the inspected computer is `torch 2.10.0+cpu`, which will not use the RTX 4050. These commands create an isolated environment and do not replace that existing installation. The CUDA 12.8 package command follows [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v2-10-0); a compatible NVIDIA driver is required.

On **computer 3**:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-cpu.txt
```

The first neural run downloads public pretrained weights and tokenizer files. Training data stays local. Allow several GB of download/cache space and approximately 10 GB free disk per neural computer for environment, cache and outputs. Plug laptops into power and run one training job per GPU.

## 2. Run one script on each computer

Computer 1:

```powershell
.\.venv\Scripts\python computer_1_xlmr.py check
.\.venv\Scripts\python computer_1_xlmr.py train --out results_xlmr
```

Computer 2:

```powershell
.\.venv\Scripts\python computer_2_mdeberta.py check
.\.venv\Scripts\python computer_2_mdeberta.py train --out results_mdeberta
```

Computer 3:

```powershell
.\.venv\Scripts\python computer_3_linear_svm.py train --out results_svm_new
```

The package already includes the fully completed `results_svm` run from this computer. You can use it directly, or train `results_svm_new` yourself. Do not overwrite a completed result directory.

Both neural scripts run three learning-rate trials, up to five epochs each, followed by a refit of the best configuration. They print epoch time so you can estimate the remaining duration on your own laptop. No wall-clock training estimate is guaranteed. If time is very short, choose one learning rate **before** comparing:

```powershell
.\.venv\Scripts\python computer_1_xlmr.py train --learning-rates 0.00002 --epochs 4 --out results_xlmr_quick
```

For CUDA out-of-memory, first close other GPU applications. Retry into a new directory with `--batch-size 1 --accumulation 16`; this retains effective batch size 16. If necessary, `--max-length 192` reduces memory further but also discards more ticket content. Do not run `--allow-cpu` for a normal neural experiment unless you deliberately accept much slower training.

Completed reports are written to disk, but interrupted neural runs restart from scratch. Mid-epoch resume is not implemented. `--smoke` is a minimal SVM check only and its result cannot enter the comparison.

## 3. Select the winner on one computer

Copy the **entire** `results_xlmr`, `results_mdeberta`, and `results_svm` folders onto computer 1, alongside the scripts. If you reran SVM, substitute `results_svm_new`.

```powershell
.\.venv\Scripts\python computer_1_xlmr.py compare --runs results_xlmr results_mdeberta results_svm --out comparison
```

This verifies data fingerprints, split IDs and completed-run status, recomputes metrics from saved predictions, and writes:

- `comparison/leaderboard.json`: all three candidates;
- `comparison/winner.json`: chosen family and exact parameters.

**Ranking is primary-category accuracy first, primary macro F1 second, and all-three-target exact-match accuracy third.** This prioritizes your accuracy request. Reports also show secondary classification, urgency precision/recall and language breakdowns. There is no claim that this is the organizer's official aggregate scoring formula.

## 4. Train and check only the winner

Still on computer 1:

```powershell
.\.venv\Scripts\python computer_1_xlmr.py finalize --winner comparison/winner.json --out best_model
```

This automatically handles whichever model wins, including mDeBERTa or SVM. It locks the chosen configuration, trains it on all 4,000 training tickets, then opens the 800-ticket supplied validation set and reports final accuracy. The final directory contains the offline model, configuration, validation predictions, report and artifact checksums.

Do not choose a different winner based on that final validation result and continue calling it a blind test. It was previously evaluated for the original baseline; this tournament excludes it from parameter/model selection, but an independently collected test set would be needed for a genuinely new external test.

## What is trained?

Input: channel + subject + ticket body. Ticket IDs and language tags are excluded from features. Language is used only to report subgroup performance.

| Output | Classes | Training |
|---|---:|---|
| Primary category | 11 | Multiclass classification |
| Secondary category | 12, including `none` | Separate classification head/classifier |
| Urgency | 2 | Separate classification head/classifier |
| Support team | Fixed mapping | Derived from primary category |

After inference, secondary equals primary is converted to none; spam is forced to nonurgent with no secondary. This is identical across all three candidates.

## Parameter breakdown

| Neural parameter | Default/search | Meaning |
|---|---|---|
| Learning rate | `1e-5`, `2e-5`, `3e-5` | Size of training updates; selected using tuning data |
| Maximum epochs | 5 per trial | Passes through the fitting split |
| Early-stopping patience | 2 epochs | Stop a trial after two epochs without improvement |
| Microbatch | 2 | Tickets held on GPU for one training step |
| Gradient accumulation | 8 | Combine eight microbatches before updating weights |
| Effective batch | 16 | `2 × 8` tickets per optimizer update |
| Token limit | 256 | Preserve approximately 75% beginning and 25% end if too long |
| Optimizer | AdamW | Updates trainable weights |
| Weight decay | 0.01, excluding bias/normalization | Regularization |
| Warm-up | 10% of update steps | Gradually increase the learning rate at startup |
| Schedule | Linear decay | Decrease learning rate during training |
| Dropout | 0.1 | Reduces reliance on individual hidden features |
| Gradient clipping | 1.0 | Limits gradient norm |
| Training precision | FP16 autocast + gradient scaling on CUDA | Lower activation memory use |
| Gradient checkpointing | Enabled | Saves activation memory by recomputing during backward pass |
| Token embedding table | Frozen | Saves gradient/optimizer memory for the large multilingual vocabulary |
| Encoder layers + three heads | Trainable | Learns the ticket classification task |
| Loss weights | Primary 1.0; secondary 0.5; urgency 0.5 | Primary task receives the largest direct weight |
| Class weights | Primary unweighted; secondary/urgency inverse square-root frequency | Supports less frequent secondary/urgent outcomes |
| Random seed | 42 | Repeatable initialization/order, subject to GPU numerical differences |

XLM-R and mDeBERTa are each roughly 280-million-parameter models. Freezing the vocabulary embedding table leaves roughly 85–90 million encoder/head parameters trainable; the scripts print the exact counts. This is fine-tuning pretrained encoders, not training a language model from scratch. [XLM-R model card](https://huggingface.co/FacebookAI/xlm-roberta-base), [Microsoft mDeBERTa model card](https://huggingface.co/microsoft/mdeberta-v3-base).

| SVM parameter | Values/search | Meaning |
|---|---|---|
| Feature mixes | Balanced, character-heavy, word-heavy | Changes word/character weights and n-gram lengths |
| Word n-grams | 1–2 or 1–3 | Single words and short phrases |
| Character n-grams | 3–5 or 2–5 | Unicode fragments and spelling/transliteration patterns |
| Minimum document frequency | 1 or 2, tied to feature mix | Filters very rare features |
| Feature caps | 80,000 word + 160,000 character | Bounds sparse feature count |
| `C` | 0.25, 1.0, 4.0 | Smaller values mean stronger regularization |
| Class weighting | None or balanced | Tests compensation for category imbalance |
| Primary confidence | 3-fold sigmoid calibration inside fitting rows | Maps SVM margins to probabilities |
| Main search | 18 combinations | 3 feature mixes × 3 `C` values × 2 weighting choices |
| Secondary and urgency | 6 additional settings each | Reuse selected features; tune macro F1 / urgent F1 |

The selected vocabulary is fitted only on the fitting rows; internal calibration shares that fitting vocabulary. Comparison rows never affect vocabulary, calibration or hyperparameter choice. SVM confidence is calibrated inside training; neural softmax probabilities are not separately calibrated. SVM and other sparse text classifiers are standard baselines in [scikit-learn's text-classification comparison](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html).

## Fair comparison and existing results

All scripts produce identical category-stratified partitions from the original 4,000 training rows:

1. **2,800 fitting rows:** fit trial parameters.
2. **600 tuning rows:** select learning rate/epoch or SVM parameters.
3. Refit each candidate on those **3,400 rows** with its selected configuration.
4. **600 comparison rows:** compare the three candidate models.
5. Refit the winner on **all 4,000**, then confirm on supplied **800 validation rows**.

Exact duplicate inputs are rejected before splitting. Template/near-duplicate overlap in the synthetic data is not eliminated. Internal scores can therefore be much higher than the supplied validation score.

The completed SVM run scored **99.0% primary accuracy and 0.9892 macro F1 on the internal 600-row comparison split**. It has **not** been evaluated on the official validation set in this tournament, so this is not evidence of improvement from the original 57.5% validation accuracy. Its primary best configuration is balanced TF-IDF features, `C=1.0`, no class weighting. See `results_svm/result.json` for all metrics and settings.

Neural paths were exercised with tiny randomly initialized versions of both architectures: forward/backward, epoch selection, saving, offline reloading and repeatable inference passed. Those tests are software checks, not accuracy or 6-GB VRAM benchmarks. The full pretrained neural runs have not been performed. See `verification.md`.

## Use the saved model

For label-free ticket JSONL, with `channel`, `subject` (optional), `text` and optional `ticket_id`:

```powershell
.\.venv\Scripts\python computer_1_xlmr.py predict --model best_model/model --input tickets.jsonl --out predictions.jsonl
```

Inference loads the saved neural tokenizer/encoder from disk with `local_files_only=True`; no hosted inference service is used. Copy the whole model directory, not just the heads file.

If SVM wins, `best_model/model/model.joblib` uses the original API's bundle format and can be selected with `MODEL_PATH`. If a neural model wins, its encoder/head loader must be integrated into the API; it cannot replace the original joblib file directly. Neural CPU latency and RAM must also be tested against the competition's 2-vCPU/4-GB deployment target. This package focuses on finding the best model first.
