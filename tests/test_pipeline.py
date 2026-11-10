import tempfile
import unittest
import json
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from digit_vision.data import dataset, image_features, split_indices
from digit_vision.predict import predict
from digit_vision.train import train as train_project


class PipelineTests(unittest.TestCase):
    def test_dataset_has_normalized_pixels_and_ten_classes(self):
        x, y = dataset()
        self.assertEqual(x.shape, (len(y), 64))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.isfinite(x).all())
        self.assertGreaterEqual(float(x.min()), 0)
        self.assertLessEqual(float(x.max()), 1)
        np.testing.assert_array_equal(np.unique(y), np.arange(10))

    def test_no_split_overlap_and_repeatability(self):
        x, y = dataset()
        splits = split_indices(y)
        self.assertEqual(len(set(np.concatenate(splits))), len(y))
        self.assertEqual(sum(map(len, splits)), len(y))
        for a, b in zip(splits, split_indices(y)):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(set(y[a]), set(range(10)))

    def test_image_contract_and_inversion(self):
        x, _ = dataset()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'digit.png'
            pixels = np.rint(x[0].reshape(8, 8) * 255).astype('uint8')
            Image.fromarray(pixels).save(path)
            np.testing.assert_allclose(image_features(path), x[:1], atol=1/255)
            Image.fromarray(255 - pixels).save(path)
            np.testing.assert_allclose(image_features(path, invert=True), x[:1], atol=1/255)

    def test_blank_image_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'blank.png'
            Image.new('L', (20, 20)).save(path)
            with self.assertRaises(ValueError):
                image_features(path)

    def test_low_contrast_image_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'low-contrast.png'
            pixels = np.full((8, 8), 100, dtype=np.uint8)
            pixels[3, 3] = 110
            Image.fromarray(pixels).save(path)
            with self.assertRaises(ValueError):
                image_features(path)

    def test_training_artifacts_and_sample_inference_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = train_project(root)
            saved_report = json.loads((root / 'metrics.json').read_text())
            splits = json.loads((root / 'splits.json').read_text())
            sample = json.loads((root / 'sample.json').read_text())
            artifact = joblib.load(root / 'model.joblib')
            self.assertEqual(saved_report['selected_model'], report['selected_model'])
            self.assertEqual(sample['dataset_index'], splits['test'][0])
            self.assertEqual(sum(map(len, splits.values())), len(dataset()[1]))
            self.assertEqual(artifact['input_shape'], [8, 8])
            expected = int(artifact['model'].predict(image_features(root / 'sample.png'))[0])
            self.assertEqual(predict(root / 'model.joblib', root / 'sample.png'), expected)

    def test_saved_model_inference_on_held_out_image(self):
        x, y = dataset()
        train, _, test = split_indices(y)
        model = LogisticRegression(max_iter=2000).fit(x[train], y[train])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            joblib.dump({'model': model}, root / 'model.joblib')
            image = np.rint(x[test[0]].reshape(8, 8) * 255).astype('uint8')
            Image.fromarray(image).save(root / 'digit.png')
            actual = predict(root / 'model.joblib', root / 'digit.png')
            expected = int(model.predict(image_features(root / 'digit.png'))[0])
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
