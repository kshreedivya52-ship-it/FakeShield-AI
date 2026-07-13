import sys
import torch
import numpy as np
import shap
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from pathlib import Path
from typing import Dict, Any, List

from app.config.settings import MODEL_DIR

class SHAPExplainer:
    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = MODEL_DIR / "bert_model.pt"
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please train the model first."
            )

        print("Loading model and tokenizer for SHAP...")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        
        # Load HuggingFace model architecture and load state dict
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Create pipelines for text classification
        self.pred_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if self.device.type == "cpu" else self.device.index,
        )

        # Define custom predictor function for SHAP
        def predictor(texts):
            if isinstance(texts, str):
                texts = [texts]
            else:
                texts = [str(t) for t in texts]
                
            results = self.pred_pipeline(texts, top_k=None)
            
            probs = []
            for res in results:
                # Sort by label ('LABEL_0' is Fake, 'LABEL_1' is Real)
                sorted_res = sorted(res, key=lambda x: x["label"])
                probs.append([r["score"] for r in sorted_res])
            return np.array(probs)

        self.predictor = predictor
        # Partition explainer is the most robust and standard for text models
        self.explainer = shap.Explainer(self.predictor, self.tokenizer)
        self._initialized = True

    def explain(self, text: str) -> Dict[str, Any]:
        """
        Explain the model's prediction on a text and return word-level importances.
        """
        self._initialize()

        if not text or not text.strip():
            return {
                "prediction": "Unknown",
                "probability": 0.0,
                "words": [],
                "scores": [],
            }

        # Get prediction
        probs = self.predictor([text])[0]
        pred_class = int(np.argmax(probs))
        prob = float(probs[pred_class])
        label = "REAL" if pred_class == 1 else "FAKE"

        # Compute SHAP values
        # Since it's binary classification, shape of shap_values is (1, num_tokens, 2)
        shap_values = self.explainer([text])

        # Get tokens and values for the predicted class
        tokens = shap_values.data[0]
        # In some versions of SHAP, values might have shape (num_tokens, 2) or (num_tokens,)
        # Let's inspect shape and get values for the specific class
        values = shap_values.values[0]
        
        if len(values.shape) > 1 and values.shape[1] == 2:
            # It has shape (num_tokens, 2), select the predicted class values
            values = values[:, pred_class]
        elif len(values.shape) > 1:
            # Handle other cases
            values = values[:, 0]

        # Clean up tokens (remove tokenizer-specific markers like '##' or spaces)
        word_importances = []
        for token, val in zip(tokens, values):
            # Skip empty tokens
            if not token.strip():
                continue
            word_importances.append({
                "word": token,
                "score": float(val)
            })

        return {
            "prediction": label,
            "probability": prob,
            "importances": word_importances,
            "base_value": float(shap_values.base_values[0][pred_class]) if len(shap_values.base_values.shape) > 1 else float(shap_values.base_values[0])
        }

if __name__ == "__main__":
    explainer = SHAPExplainer()
    try:
        res = explainer.explain("NASA discovers aliens in a secret base.")
        print("Prediction:", res["prediction"], "Conf:", res["probability"])
        print("Top importances:", res["importances"][:10])
    except Exception as e:
        import traceback
        print("Error during SHAP execution:")
        traceback.print_exc()
