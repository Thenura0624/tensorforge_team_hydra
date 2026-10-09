"""Offline adapter for the tournament's XLM-R and mDeBERTa artifacts."""
import hashlib
import os
import threading
from pathlib import Path
from app.model import TEAMS, documents

class NeuralModel:
    def __init__(self, path, metadata):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from safetensors.torch import load_file
        self.torch = torch
        path = Path(path)
        artifact_hash = hashlib.sha256()
        for file in sorted(p for p in path.rglob('*') if p.is_file()):
            artifact_hash.update(file.relative_to(path).as_posix().encode())
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    artifact_hash.update(chunk)
        self.version = 'hydra-' + metadata['family'] + '-' + artifact_hash.hexdigest()[:16]
        requested_device = os.environ.get('MODEL_DEVICE', 'cpu')
        self.device = torch.device(requested_device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('MODEL_DEVICE=cuda requires CUDA-enabled PyTorch and an available GPU.')
        torch.set_num_threads(int(os.environ.get('MODEL_THREADS', '2')))
        self.batch_size = int(os.environ.get('MODEL_BATCH_SIZE', '2'))
        if self.batch_size < 1:
            raise ValueError('MODEL_BATCH_SIZE must be positive.')
        self.max_length = int(metadata['config']['max_length'])
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'encoder', local_files_only=True, use_fast=False)
        self.encoder = AutoModel.from_pretrained(path / 'encoder', local_files_only=True, dtype=torch.float32)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(self.encoder.config.hidden_size, count) for count in [11, 12, 2]])
        self.heads.load_state_dict(load_file(str(path / 'heads.safetensors')))
        self.encoder.to(self.device).eval()
        self.heads.to(self.device).eval()
        self.lock = threading.Lock()
        self.categories = list(TEAMS)

    def predict(self, tickets):
        torch = self.torch
        predictions = []
        # Bound GPU memory independently of HTTP concurrency or 5,000-item jobs.
        # Release the lock between microbatches so live traffic can interleave.
        for offset in range(0, len(tickets), self.batch_size):
            chunk = tickets[offset:offset + self.batch_size]
            with self.lock, torch.inference_mode():
                encoded = []
                for text in documents(chunk):
                    ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
                    if len(ids) > self.max_length:
                        front = 3 * self.max_length // 4
                        ids = ids[:front] + ids[-(self.max_length - front):]
                    encoded.append(dict(input_ids=ids, attention_mask=[1] * len(ids)))
                padded = self.tokenizer.pad(encoded, padding=True, pad_to_multiple_of=8, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in padded.items() if k in ('input_ids', 'attention_mask')}
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                    hidden = self.encoder(**inputs).last_hidden_state
                    mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                    logits = [head(pooled).float() for head in self.heads]
                probabilities = logits[0].softmax(-1).cpu().numpy()
                labels = [values.argmax(-1).cpu().tolist() for values in logits]
            for i, ticket in enumerate(chunk):
                category = self.categories[labels[0][i]]
                secondary = self.categories[labels[1][i]] if labels[1][i] < 11 else None
                spam = category == 'spam_irrelevant'
                result = dict(category=category, secondary_category=None if spam or secondary == category else secondary,
                              team=TEAMS[category], is_urgent=bool(labels[2][i]) and not spam,
                              confidence=float(probabilities[i].max()), model_version=self.version)
                if 'ticket_id' in ticket:
                    result['ticket_id'] = ticket['ticket_id']
                predictions.append(result)
        return predictions
