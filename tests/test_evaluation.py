import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd

from src.classifier.data_splits import load_data, message_keys
from src.utils.data_preprocessing import split_and_save_data


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = pd.DataFrame({'message': [f'train {i}' for i in range(80)], 'label': [i % 2 for i in range(80)]})
        self.val = pd.DataFrame({'message': ['val ham', 'val spam'], 'label': [0, 1]})
        self.test = pd.DataFrame({'message': ['test ham', 'test spam'], 'label': [0, 1]})
        self.train.to_csv(self.root / 'train.csv', index=False)
        self.test.to_csv(self.root / 'test.csv', index=False)

    def load(self, **kwargs):
        return load_data(self.root / 'train.csv', self.root / 'test.csv', **kwargs)

    def test_fallback_is_reproducible_disjoint_and_preserves_test(self):
        first = self.load()
        second = self.load()
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(first[2], self.test)
        keys = [set(message_keys(frame)) for frame in first]
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        self.assertEqual(len(first[0]) + len(first[1]), 80)

    def test_validation_test_overlap_removed_in_shared_loader(self):
        val = pd.concat([self.val, pd.DataFrame({'message': [' TEST   HAM '], 'label': [0]})])
        val.to_csv(self.root / 'val.csv', index=False)
        train = pd.concat([self.train, self.test, self.val])
        train.to_csv(self.root / 'train.csv', index=False)
        actual_train, actual_val, actual_test = self.load()
        pd.testing.assert_frame_equal(actual_val, self.val)
        pd.testing.assert_frame_equal(actual_train, self.train)
        pd.testing.assert_frame_equal(actual_test, self.test)

    def test_explicit_missing_validation_path_fails(self):
        self.val.to_csv(self.root / 'val.csv', index=False)
        with self.assertRaises(FileNotFoundError):
            self.load(val_path=self.root / 'missing.csv')

    def test_validation_discovery_uses_training_directory(self):
        self.val.to_csv(self.root / 'val.csv', index=False)
        _, val, _ = self.load()
        pd.testing.assert_frame_equal(val, self.val)

    def test_duplicates_are_removed_before_random_split(self):
        pd.concat([self.train, self.train]).to_csv(self.root / 'train.csv', index=False)
        train, val, _ = self.load()
        self.assertEqual(len(train) + len(val), 80)
        self.assertFalse(set(message_keys(train)) & set(message_keys(val)))

    def test_conflicting_labels_fail(self):
        conflict = pd.DataFrame({'message': [' TEST HAM '], 'label': [1]})
        pd.concat([self.train, conflict]).to_csv(self.root / 'train.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            self.load()

    def test_empty_after_cleanup_fails(self):
        self.test.to_csv(self.root / 'val.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'validation'):
            self.load()

    def test_invalid_data_fails(self):
        for invalid in [pd.DataFrame({'text': ['x']}), pd.DataFrame({'message': [None], 'label': [0]}),
                        pd.DataFrame({'message': ['x'], 'label': [2]})]:
            with self.subTest(columns=list(invalid.columns)):
                invalid.to_csv(self.root / 'train.csv', index=False)
                with self.assertRaises(ValueError):
                    self.load()

    def test_preprocessor_produces_three_disjoint_files(self):
        data = self.train.rename(columns={'label': 'label_binary'})
        data = pd.concat([data, data.iloc[:4]])
        splits = split_and_save_data(data, output_dir=self.root)
        self.assertEqual(sum(map(len, splits)), 80)
        self.assertTrue((self.root / 'val.csv').is_file())
        keys = [set(message_keys(frame)) for frame in splits]
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])

    def test_preprocessor_rejects_invalid_split_sizes(self):
        with self.assertRaises(ValueError):
            split_and_save_data(self.train, val_size=0.6, test_size=0.6)


class TrainingContractTests(unittest.TestCase):
    def test_both_bert_trainers_select_using_validation(self):
        for name, function, trainer_name in [
            ('train_bert', 'fine_tune_bert', 'Trainer'),
            ('train_bert_enhanced', 'fine_tune_bert_enhanced', 'WeightedTrainer'),
        ]:
            with self.subTest(runner=name), tempfile.TemporaryDirectory() as output:
                module = importlib.import_module(f'src.classifier.{name}')
                train, val = object(), object()
                with patch.object(module.AutoTokenizer, 'from_pretrained'), \
                     patch.object(module.AutoModelForSequenceClassification, 'from_pretrained'), \
                     patch.object(module, trainer_name) as trainer:
                    getattr(module, function)(train, val, output_dir=output)
                self.assertIs(trainer.call_args.kwargs['train_dataset'], train)
                self.assertIs(trainer.call_args.kwargs['eval_dataset'], val)
                self.assertEqual(trainer.call_args.kwargs['args'].metric_for_best_model, 'f1_spam')
                trainer.return_value.train.assert_called_once()
                trainer.return_value.evaluate.assert_called_once_with()

    def test_all_runners_share_loader(self):
        for name in ['train_bert', 'train_bert_enhanced', 'train_baseline']:
            module = importlib.import_module(f'src.classifier.{name}')
            self.assertIs(module.load_data, load_data)

    def test_baseline_selection_precedes_final_holdout_evaluation(self):
        from src.classifier import train_baseline as module
        train = pd.DataFrame({'message': ['train ham', 'train spam'], 'label': [0, 1]})
        val = pd.DataFrame({'message': ['val ham', 'val spam'], 'label': [0, 1]})
        test = pd.DataFrame({'message': ['test ham', 'test spam'], 'label': [0, 1]})
        with tempfile.TemporaryDirectory() as output, \
             patch('sys.argv', ['train_baseline', '--plots', output]), \
             patch.object(module, 'load_data', return_value=(train, val, test)), \
             patch.object(module, 'train_naive_bayes'), patch.object(module, 'train_logistic_regression'), \
             patch.object(module, 'train_svm'), patch.object(module, 'evaluate_model') as evaluate, \
             patch.object(module, 'compare_models', return_value=('SVM', pd.DataFrame())) as compare, \
             patch.object(module, 'save_best_model') as save:
            module.main()
        self.assertEqual(evaluate.call_count, 4)
        for call in evaluate.call_args_list[:3]:
            self.assertEqual(call.args[1].tolist(), val['message'].tolist())
        self.assertEqual(evaluate.call_args_list[3].args[1].tolist(), test['message'].tolist())
        self.assertEqual(evaluate.call_args_list[3].args[3], 'SVM')
        compare.assert_called_once()
        save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
