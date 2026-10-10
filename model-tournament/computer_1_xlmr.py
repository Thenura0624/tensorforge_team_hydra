"""Standalone TensorForge model tournament runner. See --help and README.md.

This file contains its own shared functions so it can be copied to another PC.
Only channel, subject and text are model inputs. Train/finalize never call an LLM.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

FAMILY = 'xlmr'
BACKBONE = 'FacebookAI/xlm-roberta-base'
PROTOCOL = 'tensorforge-tournament-v1'
SPLIT_SEED = 20261009
SEED = 42
ROOT = Path(__file__).resolve().parent
CATEGORIES = ['payment_refund', 'ride_trip_issue', 'lost_item', 'order_missing_wrong',
              'delivery_delay', 'food_quality', 'account_promo', 'safety_conduct',
              'app_technical', 'general_inquiry', 'spam_irrelevant']
TEAMS = ['Payments & Refunds', 'Ride Operations', 'Lost & Found', 'Food Operations',
         'Delivery Operations', 'Restaurant Quality', 'Account Services', 'Trust & Safety',
         'Tech Support', 'Front-line Support', 'Auto-close / Spam Filter']
SECONDARY = CATEGORIES + ['none']

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'No rows in {path}')
    ids = [r['ticket_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate ticket IDs in input dataset.')
    for r in rows:
        if r['category'] not in CATEGORIES or (r['secondary_category'] or 'none') not in SECONDARY:
            raise ValueError('Unknown category label.')
        if type(r['is_urgent']) is not bool:
            raise ValueError('is_urgent must be a JSON Boolean.')
    return rows

def document(r):
    return f"channel_{r['channel']} subject: {r.get('subject', '')}\nbody: {r['text']}"

def targets(rows):
    return np.asarray([[CATEGORIES.index(r['category']), SECONDARY.index(r['secondary_category'] or 'none'),
                        int(r['is_urgent'])] for r in rows], dtype=np.int64)

def data_splits(data_dir):
    path = Path(data_dir) / 'train.jsonl'
    rows = read_rows(path)
    # Reject exact model-input duplicates rather than allowing cross-split leakage.
    if len({document(r) for r in rows}) != len(rows):
        raise ValueError('Exact duplicate inputs found; group/deduplicate them before splitting.')
    indices = np.arange(len(rows))
    labels = [r['category'] for r in rows]
    fit_tune, selection = train_test_split(indices, test_size=0.15, stratify=labels, random_state=SPLIT_SEED)
    fit, tune = train_test_split(fit_tune, test_size=0.15 / 0.85,
                               stratify=[labels[i] for i in fit_tune], random_state=SPLIT_SEED + 1)
    groups = {'fit': sorted(fit.tolist()), 'tune': sorted(tune.tolist()), 'selection': sorted(selection.tolist())}
    assert not set(groups['fit']) & set(groups['tune'])
    assert not set(groups['fit']) & set(groups['selection'])
    assert not set(groups['tune']) & set(groups['selection'])
    manifest = {'protocol': PROTOCOL, 'train_sha256': digest(path), 'split_seed': SPLIT_SEED,
                'ticket_ids': {k: [rows[i]['ticket_id'] for i in v] for k, v in groups.items()}}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return rows, {k: [rows[i] for i in v] for k, v in groups.items()}, manifest

def constrain(predictions):
    p = np.asarray(predictions).copy()
    p[p[:, 1] == p[:, 0], 1] = len(CATEGORIES)
    spam = p[:, 0] == CATEGORIES.index('spam_irrelevant')
    p[spam, 1] = len(CATEGORIES)
    p[spam, 2] = 0
    return p

def scores(rows, predictions):
    y = targets(rows)
    p = constrain(predictions)
    reports = {}
    for k, names in enumerate([CATEGORIES, SECONDARY, ['not_urgent', 'urgent']]):
        # Fixed label sets make macro-F1 comparable across machines and partitions.
        reports[['primary', 'secondary', 'urgency'][k]] = {
            'accuracy': float(accuracy_score(y[:, k], p[:, k])),
            'macro_f1': float(f1_score(y[:, k], p[:, k], labels=list(range(len(names))), average='macro', zero_division=0)),
            'classes': classification_report(y[:, k], p[:, k], labels=list(range(len(names))),
                                             target_names=names, output_dict=True, zero_division=0)}
    reports['joint_accuracy'] = float(np.mean(np.all(y == p, axis=1)))
    reports['by_language'] = {}
    for language in sorted({r['language'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['language'] == language]
        reports['by_language'][language] = dict(count=len(idx), primary_accuracy=float(accuracy_score(y[idx, 0], p[idx, 0])))
    return reports

def rank(metrics):
    return (metrics['primary']['accuracy'], metrics['primary']['macro_f1'], metrics['joint_accuracy'])

def versions():
    result = {'python': sys.version}
    for name in ['numpy', 'scipy', 'scikit-learn', 'torch', 'transformers', 'tokenizers', 'sentencepiece', 'safetensors']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result

def make_features(kind):
    configurations = {
        'balanced': dict(word=(1, 2), char=(3, 5), minimum=1, weights={'word': 1.0, 'char': 1.0}),
        'char_heavy': dict(word=(1, 2), char=(2, 5), minimum=2, weights={'word': 1.0, 'char': 1.5}),
        'word_heavy': dict(word=(1, 3), char=(3, 5), minimum=1, weights={'word': 1.5, 'char': 0.5})}
    conf = configurations[kind]
    return FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=conf['word'], min_df=conf['minimum'], sublinear_tf=True,
                                max_features=80000, dtype=np.float32)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=conf['char'], min_df=conf['minimum'],
                                sublinear_tf=True, max_features=160000, dtype=np.float32))],
        transformer_weights=conf['weights'])

def svm(c, balanced=False, seed=SEED):
    return LinearSVC(C=c, class_weight='balanced' if balanced else None, max_iter=10000, random_state=seed)

def primary_svm(c, balanced=False, seed=SEED):
    # The entire fitting split may define the vocabulary. Calibration fits only
    # within that fitting split; neither tune nor selection labels enter fitting.
    return CalibratedClassifierCV(svm(c, balanced, seed), method='sigmoid', cv=3, ensemble=False)

def svm_search(fit, tune, out, smoke=False):
    fy, ty = targets(fit), targets(tune)
    records, best = [], None
    kinds = ['balanced'] if smoke else ['balanced', 'char_heavy', 'word_heavy']
    cs = [1.0] if smoke else [0.25, 1.0, 4.0]
    balances = [False] if smoke else [False, True]
    for kind in kinds:
        features = make_features(kind)
        x = features.fit_transform([document(r) for r in fit])
        tx = features.transform([document(r) for r in tune])
        for c in cs:
            for balanced in balances:
                classifier = primary_svm(c, balanced)
                classifier.fit(x, fy[:, 0])
                pred = classifier.predict(tx)
                key = (float(accuracy_score(ty[:, 0], pred)), float(f1_score(ty[:, 0], pred, labels=range(11), average='macro', zero_division=0)))
                record = dict(features=kind, C=c, balanced=balanced, accuracy=key[0], macro_f1=key[1])
                records.append(record)
                write_json(out / 'search_progress.json', records)
                print('SVM trial:', json.dumps(record), flush=True)
                if best is None or key > best[0]:
                    best = (key, record)
    config = dict(features=best[1]['features'], primary_C=best[1]['C'], primary_balanced=best[1]['balanced'])
    features = make_features(config['features'])
    x = features.fit_transform([document(r) for r in fit])
    tx = features.transform([document(r) for r in tune])
    for task, index in [('secondary', 1), ('urgent', 2)]:
        winner = None
        for c in cs:
            for balanced in balances:
                classifier = svm(c, balanced).fit(x, fy[:, index])
                p = classifier.predict(tx)
                score = float(f1_score(ty[:, index], p, average='macro' if index == 1 else 'binary', zero_division=0))
                if winner is None or score > winner[0]:
                    winner = score, c, balanced
        config[task + '_C'], config[task + '_balanced'] = winner[1:]
    return config, records

def fit_svm(rows, config, destination, seed=SEED):
    destination.mkdir(parents=True, exist_ok=True)
    features = make_features(config['features'])
    x = features.fit_transform([document(r) for r in rows])
    y = targets(rows)
    # String labels deliberately match the original FastAPI model bundle.
    primary = primary_svm(config['primary_C'], config['primary_balanced'], seed).fit(x, [CATEGORIES[i] for i in y[:, 0]])
    secondary = svm(config['secondary_C'], config['secondary_balanced'], seed).fit(x, [SECONDARY[i] for i in y[:, 1]])
    urgent = svm(config['urgent_C'], config['urgent_balanced'], seed).fit(x, y[:, 2].astype(bool))
    bundle = dict(features=features, primary=primary, secondary=secondary, urgent=urgent)
    joblib.dump(bundle, destination / 'model.joblib', compress=3)
    write_json(destination / 'metadata.json', dict(family='svm', config=config, seed=seed))
    return bundle

def predict_svm(bundle, rows):
    x = bundle['features'].transform([document(r) for r in rows])
    probabilities = bundle['primary'].predict_proba(x)
    ordered = np.zeros((len(rows), len(CATEGORIES)))
    for j, label in enumerate(bundle['primary'].classes_):
        ordered[:, CATEGORIES.index(label)] = probabilities[:, j]
    p = np.column_stack([ordered.argmax(1),
                         [SECONDARY.index(s) for s in bundle['secondary'].predict(x)],
                         bundle['urgent'].predict(x).astype(int)])
    return constrain(p), ordered

def torch_imports():
    global torch, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    return torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def make_network(backbone):
    class MultiHead(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(0.1)
            self.heads = torch.nn.ModuleList([torch.nn.Linear(encoder.config.hidden_size, n) for n in [11, 12, 2]])

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # Masked mean pooling works across both model families.
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return [head(self.dropout(pooled)) for head in self.heads]
    return MultiHead(backbone)

def encode_rows(tokenizer, rows, max_length):
    encoded, truncated = [], 0
    for row in rows:
        ids = tokenizer.encode(document(row), add_special_tokens=True, truncation=False)
        if len(ids) > max_length:
            truncated += 1
            # Retain beginning/end special tokens and evidence; supported by
            # both legacy tokenizers and Transformers 5 TokenizersBackend.
            front = (3 * max_length) // 4
            ids = ids[:front] + ids[-(max_length - front):]
        encoded.append(dict(input_ids=ids, attention_mask=[1] * len(ids)))
    return encoded, truncated

def batches(tokenizer, encoded, labels, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    records = list(zip(encoded, labels))
    def collate(batch):
        features, y = zip(*batch)
        padded = tokenizer.pad(list(features), padding=True, pad_to_multiple_of=8, return_tensors='pt')
        padded = {k: v for k, v in padded.items() if k in ('input_ids', 'attention_mask')}
        return padded, torch.as_tensor(np.asarray(y), dtype=torch.long)
    return torch.utils.data.DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                                       num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator)

def neural_predict(network, loader, device):
    network.eval()
    logits = [[], [], []]
    with torch.inference_mode():
        for inputs, _ in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                values = network(**inputs)
            for k in range(3):
                logits[k].append(values[k].float().cpu().numpy())
    logits = [np.concatenate(v) for v in logits]
    p = constrain(np.column_stack([v.argmax(axis=1) for v in logits]))
    z = logits[0] - logits[0].max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return p, probs

def save_neural(network, tokenizer, destination, meta):
    from safetensors.torch import save_file
    destination.mkdir(parents=True, exist_ok=True)
    network.encoder.save_pretrained(destination / 'encoder', safe_serialization=True)
    tokenizer.save_pretrained(destination / 'encoder')
    save_file({k: v.detach().cpu().contiguous() for k, v in network.heads.state_dict().items()}, str(destination / 'heads.safetensors'))
    write_json(destination / 'metadata.json', meta)

def neural_fit(rows, config, args, destination=None, tune=None, trial_name='refit'):
    torch_imports()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('CUDA unavailable. Install CUDA-enabled PyTorch and check torch.cuda.is_available(); use --allow-cpu only for small tests.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.threads)
    seed_everything(config.get('seed', SEED))
    tokenizer = AutoTokenizer.from_pretrained(config['backbone'], revision=config.get('revision'), use_fast=False)
    # AMP scales FP32 parameter gradients; checkpoint storage may be FP16.
    # Explicit loading dtype avoids inheriting half-precision trainable weights.
    encoder = AutoModel.from_pretrained(config['backbone'], revision=config.get('revision'), dtype=torch.float32)
    resolved_revision = getattr(encoder.config, '_commit_hash', None)
    network = make_network(encoder)
    # Frozen token embeddings save ~2 GB of gradient/optimizer state on 6 GB GPUs.
    # Encoder layers and all classification heads are fine-tuned.
    network.encoder.get_input_embeddings().weight.requires_grad_(False)
    network.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(network.encoder.config, 'use_cache'):
        network.encoder.config.use_cache = False
    network.to(device=device, dtype=torch.float32)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    encoded, truncated = encode_rows(tokenizer, rows, config['max_length'])
    loader = batches(tokenizer, encoded, targets(rows), args.batch_size, True, config.get('seed', SEED))
    val_loader = None
    if tune is not None:
        val_encoded, _ = encode_rows(tokenizer, tune, config['max_length'])
        val_loader = batches(tokenizer, val_encoded, targets(tune), args.batch_size, False, SEED)
    parameters = [(n, p) for n, p in network.named_parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for _, p in parameters):
        raise RuntimeError('AMP requires FP32 trainable parameters before optimizer creation.')
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in parameters if 'bias' not in n and 'LayerNorm' not in n and 'layer_norm' not in n], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters if 'bias' in n or 'LayerNorm' in n or 'layer_norm' in n], 'weight_decay': 0.0}],
        lr=config['learning_rate'], foreach=False)
    steps = math.ceil(len(loader) / args.accumulation) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(steps * 0.1)), num_training_steps=steps)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    y = targets(rows)
    losses = [torch.nn.CrossEntropyLoss()]
    for column, count in [(1, 12), (2, 2)]:
        weights = np.sqrt(len(y) / np.maximum(np.bincount(y[:, column], minlength=count), 1))
        weights = np.minimum(weights / weights.mean(), 5.0)
        losses.append(torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device)))
    history, best, stale = [], None, 0
    print(json.dumps(dict(trial=trial_name, device=str(device), total_parameters=sum(p.numel() for p in network.parameters()),
                          trainable_parameters=sum(p.numel() for p in network.parameters() if p.requires_grad),
                          truncated_tickets=truncated, learning_rate=config['learning_rate'])), flush=True)
    for epoch in range(1, config['epochs'] + 1):
        started = time.perf_counter()
        network.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for index, (inputs, labels) in enumerate(loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                outputs = network(**inputs)
                loss = losses[0](outputs[0].float(), labels[:, 0]) + 0.5 * losses[1](outputs[1].float(), labels[:, 1]) + 0.5 * losses[2](outputs[2].float(), labels[:, 2])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss. Lower learning rate or disable mixed precision after investigating.')
            group_start = (index // args.accumulation) * args.accumulation
            denominator = min(args.accumulation, len(loader) - group_start)
            scaler.scale(loss / denominator).backward()
            total_loss += float(loss.detach())
            if (index + 1) % args.accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 200 == 0:
                print(f'{trial_name}: epoch {epoch}, batch {index + 1}/{len(loader)}', flush=True)
        record = dict(epoch=epoch, loss=total_loss / len(loader), seconds=time.perf_counter() - started)
        if tune is not None:
            p, _ = neural_predict(network, val_loader, device)
            report = scores(tune, p)
            record['metrics'] = report
            if best is None or rank(report) > rank(best['metrics']):
                best, stale = record, 0
            else:
                stale += 1
        history.append(record)
        print(f'{trial_name}: epoch {epoch}, loss {record["loss"]:.4f}, seconds {record["seconds"]:.1f}' +
              (f', tune accuracy {record["metrics"]["primary"]["accuracy"]:.4f}' if tune is not None else ''), flush=True)
        if tune is not None and stale >= args.patience:
            break
    meta = dict(family=FAMILY, backbone=config['backbone'], revision=resolved_revision, config=config,
                training_rows=len(rows), truncated_training_rows=truncated, versions=versions(),
                peak_cuda_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else None)
    if destination is not None:
        save_neural(network, tokenizer, destination, meta)
    del network, encoder, optimizer, scheduler, parameters, losses, loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best, history, resolved_revision

def predict_saved(model_dir, rows, batch_size=2, allow_cpu=True):
    meta = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
    if meta['family'] == 'svm':
        return predict_svm(joblib.load(model_dir / 'model.joblib'), rows)
    torch_imports()
    from safetensors.torch import load_file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and not allow_cpu:
        raise RuntimeError('CUDA is required for this inference run.')
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'encoder', local_files_only=True, use_fast=False)
    network = make_network(AutoModel.from_pretrained(model_dir / 'encoder', local_files_only=True))
    network.heads.load_state_dict(load_file(str(model_dir / 'heads.safetensors')))
    network.to(device=device, dtype=torch.float32)
    encoded, _ = encode_rows(tokenizer, rows, meta['config']['max_length'])
    loader = batches(tokenizer, encoded, np.zeros((len(rows), 3), dtype=np.int64), batch_size, False, SEED)
    result = neural_predict(network, loader, device)
    del network
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result

def train(args):
    rows, splits, manifest = data_splits(args.data)
    out = Path(args.out or ROOT / ('results_' + FAMILY))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise ValueError('Result already exists. Use a new --out directory to preserve the experiment.')
    write_json(out / 'split_manifest.json', manifest)
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        if FAMILY == 'svm':
            config, history = svm_search(splits['fit'], splits['tune'], out, args.smoke)
            fit_svm(splits['fit'] + splits['tune'], config, out / 'model')
        else:
            learning_rates = [float(x) for x in args.learning_rates.split(',')]
            history, best = [], None
            for i, lr in enumerate(learning_rates):
                config = dict(backbone=BACKBONE, learning_rate=lr, max_length=args.max_length, epochs=args.epochs,
                              seed=SEED, batch_size=args.batch_size, accumulation=args.accumulation)
                trial, records, revision = neural_fit(splits['fit'], config, args, tune=splits['tune'], trial_name=f'trial-{i + 1}')
                history.append(dict(config=config, best=trial, epochs=records))
                write_json(out / 'search_progress.json', history)
                if best is None or rank(trial['metrics']) > rank(best['metrics']):
                    best = trial
                    winning_config = {**config, 'epochs': trial['epoch'], 'revision': revision}
            config = winning_config
            neural_fit(splits['fit'] + splits['tune'], config, args, out / 'model')
        p, probs = predict_saved(out / 'model', splits['selection'], args.batch_size)
    metrics = scores(splits['selection'], p)
    metrics['primary_log_loss'] = float(log_loss(targets(splits['selection'])[:, 0], probs, labels=range(11)))
    np.savez_compressed(out / 'selection_predictions.npz', predictions=p, probabilities=probs,
                        ticket_ids=np.array([r['ticket_id'] for r in splits['selection']]))
    result = dict(protocol=PROTOCOL, family=FAMILY, complete=True, smoke=args.smoke, config=config,
                  manifest_fingerprint=manifest['fingerprint'], train_sha256=manifest['train_sha256'],
                  selection_metrics=metrics, versions=versions(), elapsed_seconds=time.perf_counter() - started)
    write_json(out / 'result.json', result)
    print(f'SELECTION primary accuracy: {metrics["primary"]["accuracy"]:.4%}; macro F1: {metrics["primary"]["macro_f1"]:.4f}', flush=True)
    print('Finished. Copy the entire result folder to the comparison computer. Final validation has NOT been read.', flush=True)

def compare(args):
    if not args.runs or len(args.runs) != 3:
        raise ValueError('--runs requires exactly three completed result directories.')
    _, splits, manifest = data_splits(args.data)
    records = []
    for directory in args.runs:
        path = Path(directory).resolve()
        result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        if not result.get('complete') or result.get('smoke'):
            raise ValueError(f'{path}: incomplete/smoke run cannot compete.')
        if result['manifest_fingerprint'] != manifest['fingerprint'] or result['protocol'] != PROTOCOL:
            raise ValueError('Different data/splits/protocol detected. Do not compare these runs.')
        saved = np.load(path / 'selection_predictions.npz', allow_pickle=False)
        if saved['ticket_ids'].tolist() != [r['ticket_id'] for r in splits['selection']]:
            raise ValueError('Selection row order/IDs differ.')
        report = scores(splits['selection'], saved['predictions'])
        records.append(dict(family=result['family'], run_dir=str(path), config=result['config'], metrics=report))
    if {r['family'] for r in records} != {'xlmr', 'mdeberta', 'svm'}:
        raise ValueError('Expected one XLM-R, one mDeBERTa and one SVM result.')
    records.sort(key=lambda r: rank(r['metrics']), reverse=True)
    out = Path(args.out or ROOT / 'comparison')
    winner = dict(protocol=PROTOCOL, manifest_fingerprint=manifest['fingerprint'], train_sha256=manifest['train_sha256'],
                  ranking_rule='primary accuracy, then fixed-label primary macro F1, then joint accuracy',
                  **records[0])
    write_json(out / 'leaderboard.json', records)
    write_json(out / 'winner.json', winner)
    for r in records:
        print(f'{r["family"]:12s} accuracy={r["metrics"]["primary"]["accuracy"]:.4%} macroF1={r["metrics"]["primary"]["macro_f1"]:.4f}')
    print('Winner:', records[0]['family'], '\nSaved:', out / 'winner.json')

def finalize(args):
    global FAMILY
    if not args.winner:
        raise ValueError('Supply --winner comparison/winner.json after comparing all three runs.')
    winner = json.loads(Path(args.winner).read_text(encoding='utf-8'))
    rows, _, manifest = data_splits(args.data)
    if winner['protocol'] != PROTOCOL or winner['manifest_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Winner does not match this dataset and split protocol.')
    FAMILY = winner['family']
    config = copy.deepcopy(winner['config'])
    out = Path(args.out or ROOT / 'best_model')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'confirmation.json').exists() or (out / 'model').exists():
        raise ValueError('Final output already exists. Do not repeatedly optimize on confirmation results; use a fresh test set for further tuning.')
    # Commit the choice before opening the validation file.
    write_json(out / 'locked_choice.json', winner)
    with threadpool_limits(limits=args.threads):
        if FAMILY == 'svm':
            fit_svm(rows, config, out / 'model')
        else:
            args.batch_size = config['batch_size']
            args.accumulation = config['accumulation']
            neural_fit(rows, config, args, out / 'model', trial_name='final-full-train')
        validation_path = Path(args.data) / 'validation.jsonl'
        validation = read_rows(validation_path)
        if set(document(r) for r in rows) & set(document(r) for r in validation):
            raise ValueError('Exact input overlap between train and validation.')
        p, probs = predict_saved(out / 'model', validation, args.batch_size)
    report = scores(validation, p)
    report['primary_log_loss'] = float(log_loss(targets(validation)[:, 0], probs, labels=range(11)))
    write_json(out / 'confirmation.json', dict(family=FAMILY, config=config, metrics=report,
               train_count=len(rows), validation_count=len(validation), validation_sha256=digest(validation_path),
               caveat='Supplied validation was previously evaluated for the original baseline. It was excluded from this tournament, but is not a new external blind test.'))
    np.savez_compressed(out / 'validation_predictions.npz', predictions=p, probabilities=probs,
                        ticket_ids=np.array([r['ticket_id'] for r in validation]))
    write_json(out / 'artifact_manifest.json', {str(p.relative_to(out / 'model')): digest(p) for p in (out / 'model').rglob('*') if p.is_file()})
    print(f'FINAL {FAMILY}: primary accuracy {report["primary"]["accuracy"]:.4%}, macro F1 {report["primary"]["macro_f1"]:.4f}')

def predict(args):
    if not args.model or not args.input:
        raise ValueError('Supply --model best_model/model and --input tickets.jsonl.')
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Input file is empty.')
    p, probabilities = predict_saved(Path(args.model), rows, args.batch_size)
    result = []
    for row, labels, prob in zip(rows, p, probabilities):
        item = dict(category=CATEGORIES[labels[0]], secondary_category=None if labels[1] == 11 else SECONDARY[labels[1]],
                    is_urgent=bool(labels[2]), team=TEAMS[labels[0]], confidence=float(prob.max()))
        if 'ticket_id' in row:
            item['ticket_id'] = row['ticket_id']
        result.append(item)
    path = Path(args.out or ROOT / 'predictions.jsonl')
    path.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in result) + '\n', encoding='utf-8')
    print('Saved:', path)

def parser():
    p = argparse.ArgumentParser(description=f'{FAMILY}: standalone model search, compare, finalize and inference')
    p.add_argument('mode', choices=['train', 'compare', 'finalize', 'predict', 'check'])
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--out', type=Path)
    p.add_argument('--runs', nargs='+')
    p.add_argument('--winner', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--input', type=Path)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--learning-rates', default='0.00001,0.00002,0.00003')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--smoke', action='store_true', help='SVM only: minimal grid; excluded from tournament comparisons')
    return p

def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.patience, args.threads) < 1 or not 32 <= args.max_length <= 512:
        raise ValueError('Counts must be positive and max-length must be 32..512.')
    if args.mode == 'check':
        print(json.dumps(versions(), indent=2))
        if FAMILY != 'svm':
            torch_imports()
            print('CUDA available:', torch.cuda.is_available())
            if torch.cuda.is_available():
                print('GPU:', torch.cuda.get_device_name(0))
                print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 1024**3)
            else:
                raise SystemExit('Install CUDA-enabled PyTorch before a full neural run.')
    else:
        globals()[args.mode](args)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Interrupted. Completed reports are preserved; rerunning starts training from scratch.')
