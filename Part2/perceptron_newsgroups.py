"""Perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from features import make_featurize
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


@dataclass(frozen=True)
class DataPointWithFeatures(DataPoint):
    features: Dict[str, float]


def featurize_data(
    data: List[DataPoint], feature_types: Set[str]
) -> List[DataPointWithFeatures]:
    """Add features to each datapoint based on feature types"""
    # TODO: Implement this!
    # raise NotImplementedError
    featurizer = make_featurize(feature_types)
    
    labeled_data = []
    for dp in data:
        features = featurizer(dp.text)
        labeled_data.append(DataPointWithFeatures(dp.id, dp.text, dp.label, features))
    return labeled_data


class PerceptronModel:
    """Perceptron model for classification."""

    def __init__(self):
        self.weights: Dict[str, float] = defaultdict(float)
        self.labels: Set[str] = set()

    def _get_weight_key(self, feature: str, label: str) -> str:
        """An internal hash function to build keys of self.weights (needed for tests)"""
        return feature + "#" + str(label)

    def score(self, datapoint: DataPointWithFeatures, label: str) -> float:
        """Compute the score of a class given the input.

        Inputs:
            datapoint (Datapoint): a single datapoint with features populated
            label (str): label

        Returns:
            The output score.
        """
        # TODO: Implement this! Expected # of lines: <10
        # raise NotImplementedError
        score_val = 0.0
        for feat_name, feat_value in datapoint.features.items():
            key = self._get_weight_key(feat_name, label)
            # self.weights[key] returns 0.0 if key is missing (defaultdict)
            score_val += self.weights[key] * feat_value
        return score_val
    
    def predict(self, datapoint: DataPointWithFeatures) -> str:
        """Predicts a label for an input.

        Inputs:
            datapoint: Input data point.

        Returns:
            The predicted class.
        """
        # TODO: Implement this! Expected # of lines: <5
        # raise NotImplementedError
        if not self.labels:
            return ""
        
        # Sort labels to ensure deterministic tie-breaking.
        # This prevents random failures in tests if scores are identical.
        sorted_labels = sorted(list(self.labels))
        
        # Returns the label with the highest score
        return max(sorted_labels, key=lambda l: self.score(datapoint, l))

    def update_parameters(
        self, datapoint: DataPointWithFeatures, prediction: str, lr: float
    ) -> None:
        """Update the model weights of the model using the perceptron update rule.

        Inputs:
            datapoint: The input example, including its label.
            prediction: The predicted label.
            lr: Learning rate.
        """
        # TODO: Implement this! Expected # of lines: <10
        # raise NotImplementedError
        # If prediction is correct, do nothing
        if prediction == datapoint.label:
            return

        # Perceptron Update Rule:
        # Weights for CORRECT label increase by (lr * feature_value)
        # Weights for WRONG (predicted) label decrease by (lr * feature_value)
        for feat_name, feat_value in datapoint.features.items():
            correct_key = self._get_weight_key(feat_name, datapoint.label)
            wrong_key = self._get_weight_key(feat_name, prediction)
            
            self.weights[correct_key] += lr * feat_value
            self.weights[wrong_key] -= lr * feat_value
        

    def train(
        self,
        training_data: List[DataPointWithFeatures],
        val_data: List[DataPointWithFeatures],
        num_epochs: int,
        lr: float,
    ) -> None:
        """Perceptron model training. Updates self.weights and self.labels
        We greedily learn about new labels.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        # TODO: Implement this!
        # raise NotImplementedError
        # 1. Register all unique labels from the training data
        for dp in training_data:
            if dp.label:
                self.labels.add(dp.label)

        # 2. Training Loop
        # The instructions mention picking examples "in sequence", so we iterate directly.
        for epoch in range(num_epochs):
            # Using tqdm for progress tracking
            for dp in tqdm(training_data, desc=f"Epoch {epoch+1}"):
                prediction = self.predict(dp)
                self.update_parameters(dp, prediction, lr)

    def save_weights(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps(self.weights, indent=2, sort_keys=True))
        print(f"Model weights saved to {path}")

    def evaluate(
        self,
        data: List[DataPointWithFeatures],
        save_path: str = None,
    ) -> float:
        """Evaluates the model on the given data.

        Inputs:
            data (list of Datapoint): The data to evaluate on.
            save_path: The path to save the predictions.

        Returns:
            accuracy (float): The accuracy of the model on the data.
        """
        # TODO: Implement this!
        # raise NotImplementedError
        predictions = []
        targets = []
        
        for dp in data:
            pred = self.predict(dp)
            predictions.append(pred)
            # For test data, label might be None, but accuracy requires targets
            if dp.label is not None:
                targets.append(dp.label)

        if save_path:
            save_results([DataPoint(d.id, d.text, d.label) for d in data], predictions, save_path)

        # If we have targets, compute accuracy, otherwise return 0.0
        if targets:
            return accuracy(predictions, targets)
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="newsgroups",
        help="Data source, one of ('sst2', 'newsgroups')",
    )

    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default="bow+bigram+sentiment",
        help="Feature type, e.g., bow+len",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=20, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.005, help="Learning rate"
    )
    args = parser.parse_args()

    data_type = DataType(args.data)
    feature_types: Set[str] = set(args.features.split("+"))
    num_epochs: int = args.epochs
    lr: float = args.learning_rate

    train_data, val_data, dev_data, test_data = load_data(data_type)
    train_data = featurize_data(train_data, feature_types)
    val_data = featurize_data(val_data, feature_types)
    dev_data = featurize_data(dev_data, feature_types)
    test_data = featurize_data(test_data, feature_types)

    model = PerceptronModel()
    print("Training the model...")
    model.train(train_data, val_data, num_epochs, lr)

    # Predict on the development set.
    dev_acc = model.evaluate(
        dev_data,
        save_path=os.path.join(
            "results",
            f"perceptron_{args.data}_{args.features}_dev_predictions.csv",
        ),
    )
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on the test set
    _ = model.evaluate(
        test_data,
        save_path=os.path.join(
            "results",
            f"perceptron_{args.data}_test_predictions.csv",
        ),
    )

    model.save_weights(
        os.path.join(
            "results", f"perceptron_{args.data}_{args.features}_model.json"
        )
    )
