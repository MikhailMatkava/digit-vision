"""Inference for a single image. Load only your own trusted joblib artifacts."""
import argparse
import json
import joblib
from .data import image_features


def predict(model_path, image_path, invert=False):
    artifact = joblib.load(model_path)
    features = image_features(image_path, invert=invert)
    return int(artifact['model'].predict(features)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='artifacts/model.joblib')
    parser.add_argument('--image', required=True)
    parser.add_argument('--invert', action='store_true', help='Use for a dark digit on a light background.')
    args = parser.parse_args()
    try:
        print(json.dumps({'digit': predict(args.model, args.image, args.invert)}))
    except (OSError, ValueError) as error:
        parser.exit(2, f'Prediction failed: {error}\n')
