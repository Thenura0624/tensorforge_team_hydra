"""Rebuild README figures from saved result JSON without changing experiments."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'images'
OUT.mkdir(parents=True, exist_ok=True)
TEAL, GREEN, AMBER, BG = '#17463F', '#6D987C', '#BD683F', '#F5F6EF'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.facecolor': BG, 'figure.facecolor': BG,
                     'axes.labelcolor': TEAL, 'text.color': TEAL,
                     'xtick.color': TEAL, 'ytick.color': TEAL})

def save(fig, name, caption):
    fig.text(.08, .035, caption, fontsize=10, color='#52665D')
    fig.subplots_adjust(left=.09, right=.97, top=.80, bottom=.20)
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)

board = json.loads((ROOT / 'reports/model-comparison/leaderboard.json').read_text(encoding='utf-8'))
names = {'xlmr': 'XLM-R', 'svm': 'Linear SVM', 'mdeberta': 'mDeBERTa'}
fig, ax = plt.subplots(figsize=(10, 5.5))
errors = [round(600 * (1-r['metrics']['primary']['accuracy'])) for r in board]
bars = ax.bar([names[r['family']] for r in board], errors,
              color=[TEAL, GREEN, AMBER], width=.5)
ax.set_ylim(0, 8.4)
ax.set_ylabel('Primary classification errors')
ax.set_yticks(range(0, 9, 2))
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=.2)
for bar, row, error in zip(bars, board, errors):
    ax.text(bar.get_x()+bar.get_width()/2, error+.2,
            f"{error} error{'s' if error != 1 else ''}\n{row['metrics']['primary']['accuracy']:.2%} accuracy",
            ha='center', fontsize=12)
fig.suptitle('Candidate comparison: 600 internal tickets', fontsize=20, fontweight='bold', y=.96)
save(fig, 'internal_comparison.png',
     'Source: reports/model-comparison/leaderboard.json. Internal selection scores, not final validation.')

winner = json.loads((ROOT / 'reports/model-comparison/winner.json').read_text(encoding='utf-8'))
groups = winner['metrics']['by_language']
order = ['en', 'si', 'ta', 'singlish', 'tanglish', 'mixed']
labels = dict(en='English', si='Sinhala', ta='Tamil', singlish='Singlish', tanglish='Tanglish', mixed='Mixed')
fig, ax = plt.subplots(figsize=(11, 5.5))
values = [groups[k]['primary_accuracy']*100 for k in order]
bars = ax.bar([f"{labels[k]}\nn={groups[k]['count']}" for k in order], values,
              color=[AMBER if k == 'tanglish' else TEAL for k in order], width=.6)
ax.set_ylim(0, 115)
ax.set_ylabel('Primary accuracy (%)')
ax.set_yticks(range(0, 101, 20))
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, value+2, f'{value:.2f}%', ha='center', fontsize=11)
fig.suptitle(f"{names[winner['family']]} language results with sample sizes",
             fontsize=20, fontweight='bold', y=.96)
save(fig, 'language_accuracy.png',
     'Source: reports/model-comparison/winner.json. Small groups limit conclusions; all examples are synthetic.')

baseline = ROOT / 'reports/metrics.json'
if baseline.exists():
    metrics = json.loads(baseline.read_text(encoding='utf-8'))
    values = [metrics[k]['accuracy']*100 for k in ['category', 'secondary_category', 'is_urgent']]
    values.append(metrics['joint_exact_match']*100)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(['Primary', 'Secondary', 'Urgency', 'Joint'], values,
                  color=[TEAL, GREEN, GREEN, AMBER], width=.55)
    ax.set_ylim(0, 112)
    ax.set_ylabel('Accuracy (%)')
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+2, f'{value:.2f}%', ha='center', fontsize=12)
    fig.suptitle('Original baseline: 800 supplied validation tickets', fontsize=19, fontweight='bold', y=.96)
    save(fig, 'baseline_validation.png',
         'Source: companion API reports/metrics.json. Different split from the candidate comparison.')
else:
    print('Companion metrics absent; keeping any existing baseline chart unchanged.')
print('Charts saved to', OUT)
