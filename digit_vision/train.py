"""Select on validation only; evaluate the selected model once on test."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import dataset, split_indices


def train(output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y = dataset()
    tr, va, te = split_indices(y, seed)
    candidates = {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1)),
        'rbf_svm': make_pipeline(StandardScaler(), SVC(C=10, gamma='scale')),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(x[tr], y[tr])
        pred = model.predict(x[va])
        validation[name] = {'accuracy': accuracy_score(y[va], pred),
                            'macro_f1': f1_score(y[va], pred, average='macro')}
    selected = max(validation, key=lambda name: validation[name]['macro_f1'])
    model = clone(candidates[selected])
    train_val = np.concatenate([tr, va])
    model.fit(x[train_val], y[train_val])
    pred = model.predict(x[te])
    report = {
        'seed': seed, 'dataset': 'sklearn.datasets.load_digits',
        'split_sizes': {'train': len(tr), 'validation': len(va), 'test': len(te)},
        'selection_metric': 'validation macro_f1', 'validation': validation,
        'selected_model': selected, 'test_accuracy': accuracy_score(y[te], pred),
        'test_macro_f1': f1_score(y[te], pred, average='macro'),
        'classification_report': classification_report(y[te], pred, output_dict=True, zero_division=0),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'splits.json').write_text(json.dumps({'train': tr.tolist(), 'validation': va.tolist(), 'test': te.tolist()}, indent=2) + '\n')
    joblib.dump({'model': model, 'input_shape': [8, 8], 'sklearn_version': sklearn.__version__}, output / 'model.joblib')
    ConfusionMatrixDisplay.from_predictions(y[te], pred, labels=np.arange(10), cmap='Blues')
    plt.title('Held-out test set')
    plt.tight_layout()
    plt.savefig(output / 'confusion_matrix.png', dpi=140)
    plt.close()
    errors = np.flatnonzero(pred != y[te])
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, errors[:10]):
        ax.imshow(x[te[index]].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'True {y[te[index]]} / pred {pred[index]}')
    fig.suptitle(f'Test errors: {len(errors)} / {len(te)} (up to 10 shown)')
    fig.tight_layout()
    fig.savefig(output / 'errors.png', dpi=140)
    plt.close(fig)
    # This sample is held out, but is still in-distribution, not a real photo.
    Image.fromarray(np.rint(x[te[0]].reshape(8, 8) * 255).astype('uint8')).save(output / 'sample.png')
    (output / 'sample.json').write_text(json.dumps({'dataset_index': int(te[0]), 'label': int(y[te[0]])}) + '\n')
    print(json.dumps({key: report[key] for key in ['selected_model', 'test_accuracy', 'test_macro_f1']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.output, args.seed)
