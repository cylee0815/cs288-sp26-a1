"""Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.


We adapt shape suffixes style when working with tensors.
See https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd.

Dimension key:

b: batch size
l: max sequence length
c: number of classes
v: vocabulary size

For example,

feature_b_l means a tensor of shape (b, l) == (batch_size, max_sequence_length).
length_1 means a tensor of shape (1) == (1,).
loss means a tensor of shape (). You can retrieve the loss value with loss.item().
"""

import argparse
import os
import re
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


class Tokenizer:
    # The index of the padding embedding.
    # This is used to pad variable length sequences.
    TOK_PADDING_INDEX = 0
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    def _pre_process_text(self, text: str) -> List[str]:
        # TODO: Implement this! Expected # of lines: 5~10
        # raise NotImplementedError
        # 1. Convert to lower case
        text = text.lower()
        
        # 2. Remove punctuation/non-alphanumeric chars
        #    This ensures "movie." and "movie" are treated the same.
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # 3. Split by whitespace
        tokens = text.split()
        
        # 4. Remove stop words
        valid_tokens = [t for t in tokens if t not in self.STOP_WORDS]
        
        return valid_tokens

    def __init__(self, data: List[DataPoint], max_vocab_size: int = None):
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # offset because padding index is 0
        self.token2id = {t: (i + 1) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, text: str) -> List[int]:
        # TODO: Implement this! Expected # of lines: 5~10
        # raise NotImplementedError
        # 1. Preprocess the text (lower, strip punctuation, remove stop words)
        tokens = self._pre_process_text(text)
        
        # 2. Map tokens to IDs, skipping any words not in our vocabulary
        #    This effectively handles "unknown" words by ignoring them.
        ids = [self.token2id[t] for t in tokens if t in self.token2id]
        
        return ids


def get_label_mappings(
    data: List[DataPoint],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Reads the labels file and returns the mapping."""
    labels = list(set([d.label for d in data]))
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return label2id, id2label


class BOWDataset(Dataset):
    def __init__(
        self,
        data: List[DataPoint],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_length: int = 100,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a single example as a tuple of torch.Tensors.
        features_l: The tokenized text of example, shaped (max_length,)
        length: The length of the text, shaped ()
        label: The label of the example, shaped ()

        All of have type torch.int64.
        """
        dp: DataPoint = self.data[idx]
        # TODO: Implement this! Expected # of lines: ~20
        # raise NotImplementedError
        
        # 1. Convert text to integers
        token_ids = self.tokenizer.tokenize(dp.text)
        
        # 2. Handle Truncation (if too long)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            
        # 3. Handle Padding (if too short)
        #    We calculate explicit length needed to pad
        current_len = len(token_ids)
        pad_len = self.max_length - current_len
        if pad_len > 0:
            token_ids += [self.tokenizer.TOK_PADDING_INDEX] * pad_len
            
        # 4. Handle Label (None check for test set)
        if dp.label is not None:
            label_id = self.label2id[dp.label]
        else:
            label_id = -1  # Placeholder for test data

        # 5. Convert to tensors
        #    Input length is the original length (capped at max_length) before padding
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(current_len, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long),
        )


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    def __init__(self, vocab_size: int, num_classes: int, padding_index: int):
        """Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.padding_index = padding_index
        # TODO: Implement this!
        # raise NotImplementedError
        # Hyperparameters
        embed_dim = 128
        hidden_dim = 128
        
        # Efficient NBOW: EmbeddingBag calculates mean of embeddings directly
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, padding_idx=padding_index, mode='mean'
        )
        
        # Multi-layer structure with nonlinearity
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Inputs:
            input_features_b_l (tensor): Input data for an example or a batch of examples.
            input_length (tensor): The length of the input data.

        Returns:
            output_b_c: The output of the model.
        """
        # TODO: Implement this!
        # raise NotImplementedError
        embeds = self.embedding(input_features_b_l) # Output: (Batch, Embed_Dim)
        
        out = self.fc1(embeds)
        out = self.activation(out)
        out = self.fc2(out)
        return out # Return logits


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, data: BOWDataset) -> List[int]:
        """Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.

        """
        all_predictions = []
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        # TODO: Implement this!
        # raise NotImplementedError
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for features, lengths, _ in dataloader:
                features = features.to(device)
                lengths = lengths.to(device)
                
                logits = self.model(features, lengths)
                preds = torch.argmax(logits, dim=1)
                all_predictions.extend(preds.tolist())
                
        return all_predictions
    

    def evaluate(self, data: BOWDataset) -> float:
        """Evaluates the model on a dataset.

        Inputs:
            data: The dataset to evaluate on.

        Returns:
            The accuracy of the model.
        """
        # TODO: Implement this!
        # raise NotImplementedError
        device = next(self.model.parameters()).device
        self.model.eval()
        correct = 0
        total = 0
        dataloader = DataLoader(data, batch_size=32, shuffle=False)

        with torch.no_grad():
            for features, lengths, labels in dataloader:
                features = features.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                logits = self.model(features, lengths)
                preds = torch.argmax(logits, dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return correct / total if total > 0 else 0.0


    def train(
        self,
        training_data: BOWDataset,
        val_data: BOWDataset,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        label_smoothing: float,
        lr_scheduler_name: str,
    ) -> None:
        """Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        torch.manual_seed(0)

        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch
        
        # ---- Scheduler factories (NOT instantiated yet) ----
        SCHEDULER_MAP = {
            "no": lambda: None,
    
            "cos": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-5
            ),
    
            "step": lambda: torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, num_epochs // 3),
                gamma=0.5
            ),
    
            "multistep": lambda: torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[num_epochs // 2, int(0.75 * num_epochs)],
                gamma=0.1
            ),
    
            "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2
            ),
    
            # Batch-based
            "onecycle": lambda: torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=3e-3,          # 3x base lr
                total_steps=total_steps,
                pct_start=0.1,         # 10% warmup
                anneal_strategy="cos",
                div_factor=10.0,     # start lr = 3e-4
                final_div_factor=1e3 # end lr = 3e-6
            ),
        }
    
        # ---- Create scheduler ONCE ----
        if lr_scheduler_name not in SCHEDULER_MAP:
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler_name}")
    
        scheduler = SCHEDULER_MAP[lr_scheduler_name]()

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0. #0
            # dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader):
                # TODO: Implement this!
                # raise NotImplementedError
                # Move data to the configured device (GPU/CPU)
                inputs_b_l = inputs_b_l.to(device)
                lengths_b = lengths_b.to(device)
                labels_b = labels_b.to(device)
                
                # 1. Zero Gradients
                optimizer.zero_grad()

                # 2. Forward Pass
                logits = self.model(inputs_b_l, lengths_b)

                # 3. Compute Loss
                loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                loss = loss_fn(logits, labels_b)

                # 4. Backward Pass
                loss.backward()

                # 5. Update Weights
                optimizer.step()
                
                # ---- OneCycle steps per batch ----
                if scheduler is not None and lr_scheduler_name == "onecycle":
                    scheduler.step()

                # 6. Accumulate Loss
                total_loss += loss.item()
            per_dp_loss = total_loss / len(dataloader)#0

            self.model.eval()
            val_acc = self.evaluate(val_data)

            # ---- Epoch-based scheduler step ----
            if scheduler is not None and lr_scheduler_name != "onecycle":
                if lr_scheduler_name == "plateau":
                    # Plateau expects a metric, usually val loss
                    scheduler.step(per_dp_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
    
            print(
                f"Epoch: {epoch + 1:<2} | "
                f"Loss: {per_dp_loss:.2f} | "
                f"Val accuracy: {100 * val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            # print(
            #     f"Epoch: {epoch + 1:<2} | Loss: {per_dp_loss:.2f} | Val accuracy: {100 * val_acc:.2f}%"
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=30, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "-ls", "--label_smoothing", type=float, default=0.0, help="Label smoothing for training loss"
    )
    parser.add_argument(
        "-lrs", "--lr_scheduler", type=str, default="no", help="Learning rate scheduler"
    )
    
    args = parser.parse_args()

    num_epochs = args.epochs
    lr = args.learning_rate
    label_smoothing = args.label_smoothing
    lr_scheduler_name = args.lr_scheduler
    data_type = DataType(args.data)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    print("Id to label mapping:")
    pprint(id2label)

    max_length = 100
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
    )

    trainer = Trainer(model)

    print("Training the model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trainer.train(train_ds, val_ds, optimizer, num_epochs, label_smoothing, lr_scheduler_name)

    # Evaluate on dev
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on test
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
    )
