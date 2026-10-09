"""Offline multilingual baseline; IDs and language tags are never features."""
from pathlib import Path
import hashlib
import json
import joblib

TEAMS = dict(zip(
    ['payment_refund', 'ride_trip_issue', 'lost_item', 'order_missing_wrong',
     'delivery_delay', 'food_quality', 'account_promo', 'safety_conduct',
     'app_technical', 'general_inquiry', 'spam_irrelevant'],
    ['Payments & Refunds', 'Ride Operations', 'Lost & Found', 'Food Operations',
     'Delivery Operations', 'Restaurant Quality', 'Account Services', 'Trust & Safety',
     'Tech Support', 'Front-line Support', 'Auto-close / Spam Filter']))

def documents(tickets):
    return [f"channel_{t['channel']} subject: {t.get('subject', '')}\nbody: {t['text']}" for t in tickets]

class Model:
    def __init__(self, path):
        path = Path(path)
        self.neural = None
        if path.is_dir():
            metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
            if metadata['family'] != 'svm':
                from app.neural_model import NeuralModel
                self.neural = NeuralModel(path, metadata)
                self.version = self.neural.version
                return
            path = path / 'model.joblib'
        self.version = 'tf-baseline-' + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        self.bundle = joblib.load(path)  # Only load locally trained, trusted artifacts.

    def predict(self, tickets):
        if self.neural is not None:
            return self.neural.predict(tickets)
        x = self.bundle['features'].transform(documents(tickets))
        probabilities = self.bundle['primary'].predict_proba(x)
        categories = self.bundle['primary'].classes_[probabilities.argmax(axis=1)]
        secondary = self.bundle['secondary'].predict(x)
        urgent = self.bundle['urgent'].predict(x)
        result = []
        for i, ticket in enumerate(tickets):
            category = str(categories[i])
            second = str(secondary[i])
            item = dict(category=category,
                        secondary_category=None if second in ('none', category) or category == 'spam_irrelevant' else second,
                        team=TEAMS[category],
                        is_urgent=bool(urgent[i]) and category != 'spam_irrelevant',
                        confidence=float(probabilities[i].max()), model_version=self.version)
            if 'ticket_id' in ticket:
                item['ticket_id'] = ticket['ticket_id']
            result.append(item)
        return result
