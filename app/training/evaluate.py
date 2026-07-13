import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader

from app.config.settings import MODEL_DIR
from app.training.bert_train import NewsDataset


def evaluate_model(model_name="bert-base-uncased", max_len=128, batch_size=16):
    print("=" * 50)
    print("Evaluating Trained BERT Model")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model and tokenizer
    model_path = MODEL_DIR / "bert_model.pt"
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load test set
    test_path = MODEL_DIR / "test_split.csv"
    if not test_path.exists():
        print(f"Error: Test split not found at {test_path}.")
        return
    test_df = pd.read_csv(test_path)
    print(f"Loaded test split of size {len(test_df)}.")

    test_dataset = NewsDataset(
        texts=test_df["text"].values,
        labels=test_df["label"].values,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 3. Predict
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm_loader(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])  # probability of class 1 (Real)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 4. Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    
    # In case there's only one class present in small subsets, handle ROC-AUC exception
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)

    print("\nEvaluation Summary:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics to JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics JSON saved to: {metrics_path}")


def tqdm_loader(loader):
    # Quick helper to handle progress bar without adding hard import issues
    from tqdm import tqdm
    return tqdm(loader, desc="Evaluating Batches")


if __name__ == "__main__":
    evaluate_model()
