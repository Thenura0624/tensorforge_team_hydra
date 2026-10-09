"""Train on the supplied train split; evaluate once on untouched validation."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import hashlib
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss, confusion_matrix
from threadpoolctl import threadpool_limits
from app.model import Model, documents

ROOT = Path(__file__).resolve().parent

def read(name):
    return [json.loads(line) for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines()]

def main():
    train, valid = read('train.jsonl'), read('validation.jsonl')
    features = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2,
                                sublinear_tf=True, max_features=100000))])
    print('Fitting multilingual word/character TF-IDF...', flush=True)
    x = features.fit_transform(documents(train))
    bundle = {'features': features}
    for name, key in [('primary', 'category'), ('secondary', 'secondary_category'), ('urgent', 'is_urgent')]:
        y = [r[key] if r[key] is not None else 'none' for r in train]
        classifier = LogisticRegression(C=4.0, max_iter=1000, random_state=42,
                                        class_weight='balanced' if name == 'urgent' else None)
        with threadpool_limits(limits=2):
            classifier.fit(x, y)
        bundle[name] = classifier
        print(f'Trained {name}; iterations={classifier.n_iter_.tolist()}', flush=True)
    path = ROOT / 'models' / 'baseline.joblib'
    joblib.dump(bundle, path, compress=3)
    model = Model(path)
    predictions = model.predict(valid)
    metrics = {'model_version': model.version, 'train_count': len(train), 'validation_count': len(valid),
               'feature_count': x.shape[1], 'seed': 42, 'source_sha256': {
                   n: hashlib.sha256((ROOT / 'data' / n).read_bytes()).hexdigest()
                   for n in ['train.jsonl', 'validation.jsonl']},
               'train_category_counts': dict(Counter(r['category'] for r in train)),
               'exact_input_overlap': len(set(documents(train)) & set(documents(valid)))}
    for key in ['category', 'secondary_category', 'is_urgent']:
        y = [str(r[key]) if r[key] is not None else 'none' for r in valid]
        p = [str(r[key]) if r[key] is not None else 'none' for r in predictions]
        metrics[key] = classification_report(y, p, output_dict=True, zero_division=0)
        print(key, 'accuracy:', accuracy_score(y, p), 'macro F1:', f1_score(y, p, average='macro'), flush=True)
    metrics['by_language'] = {}
    for language in sorted(set(r['language'] for r in valid)):
        indices = [i for i, r in enumerate(valid) if r['language'] == language]
        y = [valid[i]['category'] for i in indices]
        p = [predictions[i]['category'] for i in indices]
        metrics['by_language'][language] = dict(count=len(indices), accuracy=accuracy_score(y, p),
                                               macro_f1=f1_score(y, p, average='macro', zero_division=0))
    metrics['joint_exact_match'] = np.mean([all(a[k] == b[k] for k in ['category', 'secondary_category', 'is_urgent'])
                                          for a, b in zip(valid, predictions)])
    probs = bundle['primary'].predict_proba(features.transform(documents(valid)))
    metrics['primary_log_loss'] = log_loss([r['category'] for r in valid], probs, labels=bundle['primary'].classes_)
    metrics['confusion_matrix_labels'] = bundle['primary'].classes_.tolist()
    metrics['confusion_matrix'] = confusion_matrix([r['category'] for r in valid], [r['category'] for r in predictions],
                                                   labels=bundle['primary'].classes_).tolist()
    (ROOT / 'reports' / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (ROOT / 'reports' / 'validation_predictions.jsonl').write_text(
        '\n'.join(json.dumps(p, ensure_ascii=False) for p in predictions) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
