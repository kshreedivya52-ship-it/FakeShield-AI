import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import settings and data loader
from app.config.settings import MODEL_DIR
from app.utils.data_loader import DataLoader as ProjectDataLoader


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train_model(
    model_name="bert-base-uncased",
    epochs=1,
    batch_size=8,
    limit_data=1000,
    learning_rate=2e-5,
    max_len=128,
):
    print("=" * 50)
    print(f"Starting BERT Training using model: {model_name}")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}, limit_data={limit_data}")
    print("=" * 50)

    # 1. Load Data
    loader = ProjectDataLoader()
    df = loader.load_dataset()

    # Preprocessing text to avoid empty strings
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.strip() != ""]

    if limit_data and limit_data > 0:
        # Subsample class-wise to maintain balance
        fake_samples = df[df["label"] == 0].sample(n=min(len(df[df["label"] == 0]), limit_data // 2), random_state=42)
        true_samples = df[df["label"] == 1].sample(n=min(len(df[df["label"] == 1]), limit_data // 2), random_state=42)
        df = pd.concat([fake_samples, true_samples], ignore_index=True)
        print(f"Subsampled dataset to {len(df)} records for training speed.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"].tolist()
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )

    print(f"Split sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # 2. Tokenizer and DataLoaders
    print("Loading BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading pretrained BERT Model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # 4. Training Loop
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc="Training Batches"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation Loop
        model.eval()
        total_val_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation Batches"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions.double() / len(val_dataset)
        print(f"Average Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # 5. Save Model and Tokenizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "bert_model.pt"
    
    # Save the PyTorch model state dict
    torch.save(model.state_dict(), model_path)
    print(f"Trained BERT model state dict saved to: {model_path}")

    # Save tokenizer and config inside MODEL_DIR as well for deployment convenience
    tokenizer.save_pretrained(MODEL_DIR)
    model.config.save_pretrained(MODEL_DIR)
    
    # Keep track of test split for evaluate.py
    test_df = pd.DataFrame({"text": test_texts, "label": test_labels})
    test_df.to_csv(MODEL_DIR / "test_split.csv", index=False)
    print("Test split saved for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT for Fake News Detection")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--limit-data", type=int, default=1000, help="Limit dataset size (0 for full dataset)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length")

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit_data=args.limit_data,
        learning_rate=args.lr,
        max_len=args.max_len,
    )
