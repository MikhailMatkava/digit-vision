"""Compare validation scores across a few reproducible splits."""
import argparse
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from digit_vision.data import dataset, split_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 21, 42])
    parser.add_argument('--c', type=float, default=1.0)
    args = parser.parse_args()

    x, y = dataset()
    results = []
    for seed in args.seeds:
        train, validation, test = split_indices(y, seed=seed)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=args.c))
        model.fit(x[train], y[train])
        predicted = model.predict(x[validation])
        results.append({
            'seed': seed,
            'train_size': len(train),
            'validation_size': len(validation),
            'held_out_test_size': len(test),
            'validation_accuracy': round(float(accuracy_score(y[validation], predicted)), 4),
            'validation_macro_f1': round(float(f1_score(y[validation], predicted, average='macro')), 4),
        })
    print(json.dumps({'model': 'scaled logistic regression', 'C': args.c,
                      'results': results}, indent=2))


if __name__ == '__main__':
    main()
